"""
Analytics System
Ported from backend/bulk/analytics_dashboard.py and adapted for Unified AI Model.
"""

import sqlite3
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime, timedelta
import psutil

logger = logging.getLogger(__name__)

class AnalyticsSystem:
    """Real-time analytics system for Unified AI Model."""
    
    def __init__(self, db_path: str = "data/unified_ai/analytics.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_database()
    
    def init_database(self):
        """Initialize analytics database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                description TEXT,
                priority INTEGER,
                status TEXT,
                processing_time REAL,
                tokens_used INTEGER,
                result_length INTEGER
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                cpu_percent REAL,
                memory_percent REAL,
                disk_percent REAL,
                active_workers INTEGER,
                queue_size INTEGER
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def log_task(self, task_id: str, description: str, priority: int, 
                 status: str, processing_time: float, tokens_used: int, 
                 result_length: int):
        """Log task execution to analytics database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO tasks (task_id, description, priority, status, 
                                 processing_time, tokens_used, result_length)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (task_id, description, priority, status, processing_time, 
                  tokens_used, result_length))
            
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Error logging task analytics: {e}")
    
    def log_performance(self, active_workers: int, queue_size: int):
        """Log system performance metrics."""
        try:
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO performance (cpu_percent, memory_percent, disk_percent,
                                       active_workers, queue_size)
                VALUES (?, ?, ?, ?, ?)
            ''', (cpu_percent, memory.percent, (disk.used / disk.total) * 100, 
                  active_workers, queue_size))
            
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Error logging performance analytics: {e}")
    
    def get_task_analytics(self, days: int = 7) -> Dict[str, Any]:
        """Get task analytics for specified period."""
        try:
            conn = sqlite3.connect(self.db_path)
            
            start_date = datetime.now() - timedelta(days=days)
            
            # Total tasks
            total_tasks = conn.execute('''
                SELECT COUNT(*) FROM tasks 
                WHERE timestamp >= ?
            ''', (start_date,)).fetchone()[0]
            
            # Tasks by status
            status_counts = dict(conn.execute('''
                SELECT status, COUNT(*) FROM tasks 
                WHERE timestamp >= ?
                GROUP BY status
            ''', (start_date,)).fetchall())
            
            # Average processing time
            avg_time = conn.execute('''
                SELECT AVG(processing_time) FROM tasks 
                WHERE timestamp >= ? AND processing_time IS NOT NULL
            ''', (start_date,)).fetchone()[0] or 0
            
            # Total tokens
            total_tokens = conn.execute('''
                SELECT SUM(tokens_used) FROM tasks 
                WHERE timestamp >= ?
            ''', (start_date,)).fetchone()[0] or 0
            
            conn.close()
            
            return {
                'total_tasks': total_tasks,
                'status_distribution': status_counts,
                'average_processing_time': round(avg_time, 2),
                'total_tokens_used': total_tokens
            }
        except Exception as e:
            logger.error(f"Error getting task analytics: {e}")
            return {}
