"""
Test Result REST API
Provides REST API endpoints for querying test results programmatically
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from flask import Flask, jsonify, request
from flask_cors import CORS
import sys

# Add parent directory to path to import test_database
sys.path.insert(0, str(Path(__file__).parent))

try:
    from core.test_database import TestResultDatabase
except ImportError:
    # Fallback if database module not available
    TestResultDatabase = None

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend integration

# Global database instance
db: Optional[TestResultDatabase] = None


def init_database(db_path: Path = None):
    """Initialize database connection"""
    global db
    if TestResultDatabase:
        db = TestResultDatabase(db_path)
    else:
        print("Warning: TestResultDatabase not available")


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'database_available': db is not None,
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/runs', methods=['GET'])
def get_runs():
    """Get all test runs with optional filtering"""
    if not db:
        return jsonify({'error': 'Database not available'}), 500
    
    try:
        limit = request.args.get('limit', 50, type=int)
        category = request.args.get('category', None)
        status = request.args.get('status', None)  # 'passed', 'failed', 'partial'
        
        runs = db.get_recent_runs(limit)
        
        # Apply filters
        if category:
            runs = [r for r in runs if r.get('test_category') == category]
        
        if status:
            if status == 'passed':
                runs = [r for r in runs if r.get('success_rate', 0) == 100.0]
            elif status == 'failed':
                runs = [r for r in runs if r.get('success_rate', 0) < 100.0]
            elif status == 'partial':
                runs = [r for r in runs if 0 < r.get('success_rate', 0) < 100.0]
        
        return jsonify({
            'runs': runs,
            'count': len(runs)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/runs/<int:run_id>', methods=['GET'])
def get_run(run_id: int):
    """Get specific test run details"""
    if not db:
        return jsonify({'error': 'Database not available'}), 500
    
    try:
        run = db.get_test_run(run_id)
        if not run:
            return jsonify({'error': 'Run not found'}), 404
        
        # Get test results for this run
        with db._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM test_results WHERE run_id = ?
            """, (run_id,))
            results = [dict(row) for row in cursor.fetchall()]
        
        run['test_results'] = results
        
        return jsonify(run)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/runs/<run_name>', methods=['GET'])
def get_run_by_name(run_name: str):
    """Get test run by name"""
    if not db:
        return jsonify({'error': 'Database not available'}), 500
    
    try:
        with db._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM test_runs WHERE run_name = ?", (run_name,))
            row = cursor.fetchone()
            
            if not row:
                return jsonify({'error': 'Run not found'}), 404
            
            run = dict(row)
            run_id = run.get('id')
            
            if run_id:
                cursor.execute("SELECT * FROM test_results WHERE run_id = ?", (run_id,))
                results = [dict(row) for row in cursor.fetchall()]
                run['test_results'] = results
        
        return jsonify(run)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/statistics', methods=['GET'])
def get_statistics():
    """Get overall test statistics"""
    if not db:
        return jsonify({'error': 'Database not available'}), 500
    
    try:
        stats = db.get_test_statistics()
        
        # Add additional computed statistics
        recent_runs = db.get_recent_runs(30)
        if recent_runs:
            stats['recent_trend'] = {
                'last_30_runs': len(recent_runs),
                'avg_success_rate': sum(r.get('success_rate', 0) for r in recent_runs) / len(recent_runs),
                'total_tests_last_30': sum(r.get('total_tests', 0) for r in recent_runs)
            }
        
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/tests/failing', methods=['GET'])
def get_failing_tests():
    """Get most frequently failing tests"""
    if not db:
        return jsonify({'error': 'Database not available'}), 500
    
    try:
        limit = request.args.get('limit', 20, type=int)
        failing = db.get_failing_tests(limit)
        return jsonify({
            'failing_tests': failing,
            'count': len(failing)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/tests/<test_name>', methods=['GET'])
def get_test_history(test_name: str):
    """Get history of a specific test"""
    if not db:
        return jsonify({'error': 'Database not available'}), 500
    
    try:
        # Query test results by name
        with db._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT tr.*, trun.run_name, trun.timestamp, trun.test_category
                FROM test_results tr
                JOIN test_runs trun ON tr.run_id = trun.id
                WHERE tr.test_name = ?
                ORDER BY trun.timestamp DESC
                LIMIT 50
            """, (test_name,))
            
            results = []
            for row in cursor.fetchall():
                row_dict = dict(row)
                results.append({
                    'run_id': row_dict.get('run_id'),
                    'run_name': row_dict.get('run_name'),
                    'timestamp': row_dict.get('timestamp'),
                    'status': row_dict.get('status'),
                    'duration': row_dict.get('duration'),
                    'error_message': row_dict.get('error_message'),
                    'category': row_dict.get('test_category')
                })
        
        return jsonify({
            'test_name': test_name,
            'history': results,
            'count': len(results)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/compare', methods=['POST'])
def compare_runs():
    """Compare two test runs"""
    if not db:
        return jsonify({'error': 'Database not available'}), 500
    
    try:
        data = request.json
        run1_id = data.get('run1_id')
        run2_id = data.get('run2_id')
        
        if not run1_id or not run2_id:
            return jsonify({'error': 'Both run1_id and run2_id required'}), 400
        
        run1 = db.get_test_run(run1_id)
        run2 = db.get_test_run(run2_id)
        
        if not run1 or not run2:
            return jsonify({'error': 'One or both runs not found'}), 404
        
        comparison = {
            'run1': run1,
            'run2': run2,
            'differences': {
                'total_tests': run2.get('total_tests', 0) - run1.get('total_tests', 0),
                'success_rate_change': run2.get('success_rate', 0) - run1.get('success_rate', 0),
                'execution_time_change': run2.get('execution_time', 0) - run1.get('execution_time', 0)
            }
        }
        
        return jsonify(comparison)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/search', methods=['GET'])
def search_tests():
    """Search tests by name or pattern"""
    if not db:
        return jsonify({'error': 'Database not available'}), 500
    
    try:
        query = request.args.get('q', '')
        if not query:
            return jsonify({'error': 'Query parameter required'}), 400
        
        limit = request.args.get('limit', 50, type=int)
        
        with db._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT DISTINCT test_name
                FROM test_results
                WHERE test_name LIKE ?
                LIMIT ?
            """, (f'%{query}%', limit))
            
            results = [row[0] for row in cursor.fetchall()]
        
        return jsonify({
            'query': query,
            'results': results,
            'count': len(results)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def main():
    """Run the API server"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Result REST API')
    parser.add_argument('--host', default='127.0.0.1', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to')
    parser.add_argument('--db-path', type=str, help='Path to test results database')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    # Initialize database
    db_path = Path(args.db_path) if args.db_path else None
    init_database(db_path)
    
    print(f"🚀 Starting Test Result API on http://{args.host}:{args.port}")
    print(f"📊 Database: {'Available' if db else 'Not available'}")
    print(f"📖 API Documentation:")
    print(f"   GET  /api/health - Health check")
    print(f"   GET  /api/runs - List test runs")
    print(f"   GET  /api/runs/<id> - Get run details")
    print(f"   GET  /api/statistics - Get statistics")
    print(f"   GET  /api/tests/failing - Get failing tests")
    print(f"   GET  /api/tests/<name> - Get test history")
    print(f"   POST /api/compare - Compare runs")
    print(f"   GET  /api/search?q=<query> - Search tests")
    
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == '__main__':
    main()

