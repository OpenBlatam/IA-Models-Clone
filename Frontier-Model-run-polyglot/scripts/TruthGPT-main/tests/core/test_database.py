"""
Test Result Database
Stores and queries test results in a structured database
"""

import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from contextlib import contextmanager

class TestResultDatabase:
    """SQLite database for storing test results"""
    
    def __init__(self, db_path: Path = None):
        if db_path is None:
            db_path = Path(__file__).parent.parent / "test_results.db"
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize database schema"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Test runs table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS test_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_name TEXT UNIQUE,
                    timestamp TEXT NOT NULL,
                    total_tests INTEGER,
                    passed INTEGER,
                    failed INTEGER,
                    errors INTEGER,
                    skipped INTEGER,
                    success_rate REAL,
                    execution_time REAL,
                    test_category TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Test results table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS test_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER,
                    test_name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    duration REAL,
                    error_message TEXT,
                    traceback TEXT,
                    FOREIGN KEY (run_id) REFERENCES test_runs(id)
                )
            """)
            
            # Test metrics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS test_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER,
                    metric_name TEXT NOT NULL,
                    metric_value REAL,
                    FOREIGN KEY (run_id) REFERENCES test_runs(id)
                )
            """)
            
            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_run_timestamp ON test_runs(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_result_run ON test_results(run_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_result_status ON test_results(status)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_result_name ON test_results(test_name)")
            
            conn.commit()
    
    @contextmanager
    def _get_connection(self):
        """Get database connection"""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def save_test_run(self, test_results: Dict, run_name: str = None) -> int:
        """Save test run to database"""
        if run_name is None:
            run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Insert test run
            cursor.execute("""
                INSERT OR REPLACE INTO test_runs 
                (run_name, timestamp, total_tests, passed, failed, errors, skipped, 
                 success_rate, execution_time, test_category)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                run_name,
                datetime.now().isoformat(),
                test_results.get('total_tests', 0),
                test_results.get('total_tests', 0) - test_results.get('failures', 0) - 
                test_results.get('errors', 0) - test_results.get('skipped', 0),
                test_results.get('failures', 0),
                test_results.get('errors', 0),
                test_results.get('skipped', 0),
                test_results.get('success_rate', 0),
                test_results.get('execution_time', 0),
                test_results.get('test_category')
            ))
            
            run_id = cursor.lastrowid
            
            # Insert test results
            if 'failures' in test_results:
                for test, traceback in test_results.get('failures', []):
                    cursor.execute("""
                        INSERT INTO test_results (run_id, test_name, status, error_message, traceback)
                        VALUES (?, ?, ?, ?, ?)
                    """, (run_id, str(test), 'failed', None, traceback[:1000]))
            
            if 'errors' in test_results:
                for test, traceback in test_results.get('errors', []):
                    cursor.execute("""
                        INSERT INTO test_results (run_id, test_name, status, error_message, traceback)
                        VALUES (?, ?, ?, ?, ?)
                    """, (run_id, str(test), 'error', None, traceback[:1000]))
            
            if 'skipped' in test_results:
                for test, reason in test_results.get('skipped', []):
                    cursor.execute("""
                        INSERT INTO test_results (run_id, test_name, status, error_message)
                        VALUES (?, ?, ?, ?)
                    """, (run_id, str(test), 'skipped', str(reason)))
            
            conn.commit()
            return run_id
    
    def get_test_run(self, run_id: int) -> Optional[Dict]:
        """Get test run by ID"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM test_runs WHERE id = ?", (run_id,))
            row = cursor.fetchone()
            
            if not row:
                return None
            
            return dict(row)
    
    def get_recent_runs(self, limit: int = 10) -> List[Dict]:
        """Get recent test runs"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM test_runs 
                ORDER BY timestamp DESC 
                LIMIT ?
            """, (limit,))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def search_tests(self, query: str, limit: int = 100) -> List[Dict]:
        """Search for tests by name"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT DISTINCT test_name, status, COUNT(*) as count
                FROM test_results
                WHERE test_name LIKE ?
                GROUP BY test_name, status
                ORDER BY count DESC
                LIMIT ?
            """, (f'%{query}%', limit))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def get_test_statistics(self) -> Dict:
        """Get overall test statistics"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Total runs
            cursor.execute("SELECT COUNT(*) as count FROM test_runs")
            total_runs = cursor.fetchone()['count']
            
            # Average success rate
            cursor.execute("SELECT AVG(success_rate) as avg FROM test_runs")
            avg_success = cursor.fetchone()['avg'] or 0
            
            # Total tests run
            cursor.execute("SELECT SUM(total_tests) as total FROM test_runs")
            total_tests = cursor.fetchone()['total'] or 0
            
            # Failed tests count
            cursor.execute("SELECT COUNT(*) as count FROM test_results WHERE status IN ('failed', 'error')")
            total_failures = cursor.fetchone()['count']
            
            return {
                'total_runs': total_runs,
                'average_success_rate': avg_success,
                'total_tests_run': total_tests,
                'total_failures': total_failures
            }
    
    def get_failing_tests(self, limit: int = 20) -> List[Dict]:
        """Get most frequently failing tests"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT test_name, COUNT(*) as failure_count
                FROM test_results
                WHERE status IN ('failed', 'error')
                GROUP BY test_name
                ORDER BY failure_count DESC
                LIMIT ?
            """, (limit,))
            
            return [dict(row) for row in cursor.fetchall()]

def main():
    """Example usage"""
    from pathlib import Path
    project_root = Path(__file__).parent.parent
    
    db = TestResultDatabase(project_root / "test_results.db")
    
    # Example: Save test run
    test_results = {
        'total_tests': 204,
        'passed': 200,
        'failures': 2,
        'errors': 0,
        'skipped': 2,
        'success_rate': 98.0,
        'execution_time': 45.3
    }
    
    run_id = db.save_test_run(test_results, "example_run")
    print(f"✅ Saved test run with ID: {run_id}")
    
    # Get statistics
    stats = db.get_test_statistics()
    print(f"\n📊 Statistics:")
    print(f"  Total Runs: {stats['total_runs']}")
    print(f"  Avg Success Rate: {stats['average_success_rate']:.1f}%")
    
    # Get recent runs
    recent = db.get_recent_runs(5)
    print(f"\n📋 Recent Runs: {len(recent)}")

if __name__ == "__main__":
    main()







