"""
Test Results API
RESTful API for accessing test results programmatically
"""

from flask import Flask, jsonify, request
from pathlib import Path
from typing import Dict, Optional
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from tests.test_database import TestResultDatabase
    from tests.test_history import TestHistory
    from tests.test_metrics import TestMetricsTracker
except ImportError as e:
    print(f"⚠️  Warning: Some dependencies unavailable: {e}")

app = Flask(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True

# Initialize database
db = TestResultDatabase(project_root / "test_results.db")
history = TestHistory()
metrics = TestMetricsTracker()

@app.route('/api/v1/runs', methods=['GET'])
def get_runs():
    """Get list of test runs"""
    limit = request.args.get('limit', 10, type=int)
    runs = db.get_recent_runs(limit=limit)
    return jsonify({
        'runs': [dict(run) for run in runs],
        'count': len(runs)
    })

@app.route('/api/v1/runs/<int:run_id>', methods=['GET'])
def get_run(run_id: int):
    """Get specific test run"""
    run = db.get_test_run(run_id)
    if not run:
        return jsonify({'error': 'Run not found'}), 404
    return jsonify(dict(run))

@app.route('/api/v1/statistics', methods=['GET'])
def get_statistics():
    """Get test statistics"""
    stats = db.get_test_statistics()
    history_stats = history.get_statistics()
    metrics_summary = metrics.get_summary()
    
    return jsonify({
        'database': stats,
        'history': history_stats,
        'metrics': metrics_summary
    })

@app.route('/api/v1/tests/search', methods=['GET'])
def search_tests():
    """Search for tests"""
    query = request.args.get('q', '')
    limit = request.args.get('limit', 100, type=int)
    
    if not query:
        return jsonify({'error': 'Query parameter required'}), 400
    
    results = db.search_tests(query, limit=limit)
    return jsonify({
        'query': query,
        'results': [dict(r) for r in results],
        'count': len(results)
    })

@app.route('/api/v1/tests/failing', methods=['GET'])
def get_failing_tests():
    """Get most frequently failing tests"""
    limit = request.args.get('limit', 20, type=int)
    failing = db.get_failing_tests(limit=limit)
    return jsonify({
        'tests': [dict(t) for t in failing],
        'count': len(failing)
    })

@app.route('/api/v1/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'database': 'connected',
        'timestamp': __import__('datetime').datetime.now().isoformat()
    })

@app.route('/api/v1/metrics/trends', methods=['GET'])
def get_trends():
    """Get test metrics trends"""
    trends = metrics.get_trends()
    history_trends = history.get_trends()
    
    return jsonify({
        'metrics_trends': trends,
        'history_trends': history_trends
    })

@app.route('/api/v1/compare/<run1>/<run2>', methods=['GET'])
def compare_runs(run1: str, run2: str):
    """Compare two test runs"""
    try:
        from tests.test_comparator import TestResultComparator
        comparator = TestResultComparator(project_root)
        comparison = comparator.compare(run1, run2)
        return jsonify(comparison)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def run_api_server(host='127.0.0.1', port=5000, debug=False):
    """Run the API server"""
    print(f"🚀 Starting Test Results API server on http://{host}:{port}")
    print(f"📊 API Documentation:")
    print(f"   GET /api/v1/runs - List test runs")
    print(f"   GET /api/v1/runs/<id> - Get specific run")
    print(f"   GET /api/v1/statistics - Get statistics")
    print(f"   GET /api/v1/tests/search?q=<query> - Search tests")
    print(f"   GET /api/v1/tests/failing - Get failing tests")
    print(f"   GET /api/v1/metrics/trends - Get trends")
    print(f"   GET /api/v1/compare/<run1>/<run2> - Compare runs")
    print(f"   GET /api/v1/health - Health check")
    print()
    
    app.run(host=host, port=port, debug=debug)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Results API Server')
    parser.add_argument('--host', default='127.0.0.1', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    run_api_server(host=args.host, port=args.port, debug=args.debug)







