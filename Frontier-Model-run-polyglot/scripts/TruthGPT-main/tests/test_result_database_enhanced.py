"""
Enhanced Test Result Database
Advanced database operations with indexing, full-text search, and analytics
"""

import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from contextlib import contextmanager
from collections import defaultdict
import statistics


class EnhancedTestResultDatabase:
    """Enhanced SQLite database with advanced features"""
    
    def __init__(self, db_path: Path = None):
        if db_path is None:
            db_path = Path(__file__).parent / "test_results_enhanced.db"
        self.db_path = db_path
        self._init_database()
        self._create_indexes()
    
    def _init_database(self):
        """Initialize enhanced database schema"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Test runs table (enhanced)
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
                    environment TEXT,
                    branch TEXT,
                    commit_hash TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT
                )
            """)
            
            # Test results table (enhanced)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS test_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER,
                    test_name TEXT NOT NULL,
                    test_file TEXT,
                    test_class TEXT,
                    status TEXT NOT NULL,
                    duration REAL,
                    error_message TEXT,
                    traceback TEXT,
                    error_type TEXT,
                    tags TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (run_id) REFERENCES test_runs(id)
                )
            """)
            
            # Test metrics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS test_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    test_name TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value REAL,
                    timestamp TEXT NOT NULL,
                    run_id INTEGER,
                    FOREIGN KEY (run_id) REFERENCES test_runs(id)
                )
            """)
            
            # Test tags table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS test_tags (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    test_name TEXT NOT NULL,
                    tag TEXT NOT NULL,
                    UNIQUE(test_name, tag)
                )
            """)
            
            # Performance history table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS performance_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    test_name TEXT NOT NULL,
                    duration REAL NOT NULL,
                    timestamp TEXT NOT NULL,
                    run_id INTEGER,
                    FOREIGN KEY (run_id) REFERENCES test_runs(id)
                )
            """)
            
            conn.commit()
    
    def _create_indexes(self):
        """Create indexes for better query performance"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            indexes = [
                "CREATE INDEX IF NOT EXISTS idx_test_results_run_id ON test_results(run_id)",
                "CREATE INDEX IF NOT EXISTS idx_test_results_test_name ON test_results(test_name)",
                "CREATE INDEX IF NOT EXISTS idx_test_results_status ON test_results(status)",
                "CREATE INDEX IF NOT EXISTS idx_test_runs_timestamp ON test_runs(timestamp)",
                "CREATE INDEX IF NOT EXISTS idx_test_runs_category ON test_runs(test_category)",
                "CREATE INDEX IF NOT EXISTS idx_performance_test_name ON performance_history(test_name)",
                "CREATE INDEX IF NOT EXISTS idx_performance_timestamp ON performance_history(timestamp)",
                "CREATE INDEX IF NOT EXISTS idx_test_tags_test_name ON test_tags(test_name)",
            ]
            
            for index_sql in indexes:
                cursor.execute(index_sql)
            
            conn.commit()
    
    @contextmanager
    def _get_connection(self):
        """Get database connection with row factory"""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def save_test_run(
        self,
        test_results: Dict,
        run_name: str = None,
        environment: str = None,
        branch: str = None,
        commit_hash: str = None,
        metadata: Dict = None
    ) -> int:
        """Save test run with enhanced metadata"""
        if run_name is None:
            run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            metadata_json = json.dumps(metadata) if metadata else None
            
            cursor.execute("""
                INSERT OR REPLACE INTO test_runs 
                (run_name, timestamp, total_tests, passed, failed, errors, skipped, 
                 success_rate, execution_time, test_category, environment, branch, commit_hash, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                run_name,
                datetime.now().isoformat(),
                test_results.get('total_tests', 0),
                test_results.get('passed', 0),
                test_results.get('failed', 0),
                test_results.get('errors', 0),
                test_results.get('skipped', 0),
                test_results.get('success_rate', 0),
                test_results.get('execution_time', 0),
                test_results.get('test_category'),
                environment,
                branch,
                commit_hash,
                metadata_json
            ))
            
            run_id = cursor.lastrowid
            
            # Save test results
            for result in test_results.get('test_details', []):
                self._save_test_result(cursor, run_id, result)
            
            conn.commit()
            return run_id
    
    def _save_test_result(self, cursor, run_id: int, result: Dict):
        """Save individual test result"""
        cursor.execute("""
            INSERT INTO test_results 
            (run_id, test_name, test_file, test_class, status, duration, 
             error_message, traceback, error_type, tags)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            run_id,
            result.get('test_name'),
            result.get('test_file'),
            result.get('test_class'),
            result.get('status', 'unknown'),
            result.get('duration', 0),
            result.get('error_message'),
            result.get('traceback'),
            result.get('error_type'),
            ','.join(result.get('tags', [])) if result.get('tags') else None
        ))
        
        # Save performance history
        if result.get('duration', 0) > 0:
            cursor.execute("""
                INSERT INTO performance_history (test_name, duration, timestamp, run_id)
                VALUES (?, ?, ?, ?)
            """, (
                result.get('test_name'),
                result.get('duration'),
                datetime.now().isoformat(),
                run_id
            ))
    
    def search_tests(
        self,
        query: str = None,
        status: str = None,
        tags: List[str] = None,
        date_from: datetime = None,
        date_to: datetime = None,
        limit: int = 100
    ) -> List[Dict]:
        """Advanced test search with multiple filters"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            conditions = []
            params = []
            
            if query:
                conditions.append("(test_name LIKE ? OR error_message LIKE ?)")
                params.extend([f'%{query}%', f'%{query}%'])
            
            if status:
                conditions.append("status = ?")
                params.append(status)
            
            if tags:
                placeholders = ','.join(['?'] * len(tags))
                conditions.append(f"""
                    test_name IN (
                        SELECT test_name FROM test_tags WHERE tag IN ({placeholders})
                    )
                """)
                params.extend(tags)
            
            if date_from:
                conditions.append("created_at >= ?")
                params.append(date_from.isoformat())
            
            if date_to:
                conditions.append("created_at <= ?")
                params.append(date_to.isoformat())
            
            where_clause = " AND ".join(conditions) if conditions else "1=1"
            
            cursor.execute(f"""
                SELECT DISTINCT tr.*, trun.run_name, trun.timestamp
                FROM test_results tr
                JOIN test_runs trun ON tr.run_id = trun.id
                WHERE {where_clause}
                ORDER BY trun.timestamp DESC
                LIMIT ?
            """, params + [limit])
            
            return [dict(row) for row in cursor.fetchall()]
    
    def get_performance_trends(
        self,
        test_name: str,
        days: int = 30
    ) -> Dict:
        """Get performance trends for a test"""
        cutoff = datetime.now() - timedelta(days=days)
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT duration, timestamp
                FROM performance_history
                WHERE test_name = ? AND timestamp >= ?
                ORDER BY timestamp
            """, (test_name, cutoff.isoformat()))
            
            data = cursor.fetchall()
            
            if not data:
                return {}
            
            durations = [row['duration'] for row in data]
            timestamps = [row['timestamp'] for row in data]
            
            return {
                'test_name': test_name,
                'period_days': days,
                'data_points': len(durations),
                'current': durations[-1] if durations else 0,
                'average': statistics.mean(durations),
                'median': statistics.median(durations),
                'min': min(durations),
                'max': max(durations),
                'stdev': statistics.stdev(durations) if len(durations) > 1 else 0,
                'trend': self._calculate_trend(durations),
                'timeline': [
                    {'timestamp': ts, 'duration': dur}
                    for ts, dur in zip(timestamps, durations)
                ]
            }
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction"""
        if len(values) < 2:
            return 'stable'
        
        first_half = values[:len(values)//2]
        second_half = values[len(values)//2:]
        
        first_avg = statistics.mean(first_half)
        second_avg = statistics.mean(second_half)
        
        change = (second_avg - first_avg) / first_avg * 100 if first_avg > 0 else 0
        
        if abs(change) < 5:
            return 'stable'
        elif change > 0:
            return 'increasing'
        else:
            return 'decreasing'
    
    def get_test_statistics_advanced(
        self,
        test_name: str = None,
        days: int = 30
    ) -> Dict:
        """Get advanced statistics for tests"""
        cutoff = datetime.now() - timedelta(days=days)
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            if test_name:
                cursor.execute("""
                    SELECT status, COUNT(*) as count, AVG(duration) as avg_duration
                    FROM test_results
                    WHERE test_name = ? AND created_at >= ?
                    GROUP BY status
                """, (test_name, cutoff.isoformat()))
            else:
                cursor.execute("""
                    SELECT status, COUNT(*) as count, AVG(duration) as avg_duration
                    FROM test_results
                    WHERE created_at >= ?
                    GROUP BY status
                """, (cutoff.isoformat(),))
            
            stats_by_status = {row['status']: dict(row) for row in cursor.fetchall()}
            
            # Get failure rate
            total = sum(s['count'] for s in stats_by_status.values())
            failures = stats_by_status.get('failed', {}).get('count', 0) + \
                      stats_by_status.get('error', {}).get('count', 0)
            
            return {
                'period_days': days,
                'total_runs': total,
                'by_status': stats_by_status,
                'failure_rate': (failures / total * 100) if total > 0 else 0,
                'success_rate': ((total - failures) / total * 100) if total > 0 else 0
            }
    
    def get_most_changed_tests(
        self,
        days: int = 7,
        limit: int = 20
    ) -> List[Dict]:
        """Get tests with most status changes"""
        cutoff = datetime.now() - timedelta(days=days)
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT 
                    test_name,
                    COUNT(DISTINCT status) as status_count,
                    COUNT(*) as total_runs,
                    GROUP_CONCAT(DISTINCT status) as statuses
                FROM test_results
                WHERE created_at >= ?
                GROUP BY test_name
                HAVING status_count > 1
                ORDER BY status_count DESC, total_runs DESC
                LIMIT ?
            """, (cutoff.isoformat(), limit))
            
            return [dict(row) for row in cursor.fetchall()]


def main():
    """CLI interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced Test Result Database')
    parser.add_argument('--search', type=str, help='Search tests')
    parser.add_argument('--performance', type=str, help='Get performance trends for test')
    parser.add_argument('--stats', type=str, help='Get statistics for test')
    parser.add_argument('--changed', action='store_true', help='Get most changed tests')
    parser.add_argument('--db-path', type=str, help='Database path')
    
    args = parser.parse_args()
    
    db_path = Path(args.db_path) if args.db_path else None
    db = EnhancedTestResultDatabase(db_path)
    
    if args.search:
        print(f"🔍 Searching for: {args.search}")
        results = db.search_tests(query=args.search)
        print(f"Found {len(results)} results")
        for r in results[:10]:
            print(f"  {r['test_name']}: {r['status']}")
    elif args.performance:
        print(f"📊 Performance trends for: {args.performance}")
        trends = db.get_performance_trends(args.performance)
        print(f"  Current: {trends.get('current', 0):.2f}s")
        print(f"  Average: {trends.get('average', 0):.2f}s")
        print(f"  Trend: {trends.get('trend', 'unknown')}")
    elif args.stats:
        print(f"📈 Statistics for: {args.stats}")
        stats = db.get_test_statistics_advanced(args.stats)
        print(f"  Total Runs: {stats['total_runs']}")
        print(f"  Success Rate: {stats['success_rate']:.1f}%")
        print(f"  Failure Rate: {stats['failure_rate']:.1f}%")
    elif args.changed:
        print("🔄 Most changed tests:")
        changed = db.get_most_changed_tests()
        for test in changed[:10]:
            print(f"  {test['test_name']}: {test['status_count']} statuses")
    else:
        print("Use --help to see available options")


if __name__ == '__main__':
    main()

