"""
BUL Analytics Dashboard
======================

Real-time analytics dashboard for the BUL system.
"""

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import sqlite3
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict, Counter

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from modules.api_handler import APIHandler
from modules.document_processor import DocumentProcessor
from modules.query_analyzer import QueryAnalyzer
from modules.business_agents import BusinessAgentManager
from config_optimized import BULConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AnalyticsDashboard:
    """Real-time analytics dashboard for BUL system."""
    
    def __init__(self, db_path: str = "analytics.db"):
        self.db_path = db_path
        self.config = BULConfig()
        self.init_database()
        
        # Initialize components
        self.processor = DocumentProcessor(self.config.to_dict())
        self.analyzer = QueryAnalyzer()
        self.agent_manager = BusinessAgentManager(self.config.to_dict())
        self.api_handler = APIHandler(
            self.processor, self.analyzer, self.agent_manager
        )
    
    def init_database(self):
        """Initialize analytics database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS requests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                query TEXT,
                business_area TEXT,
                document_type TEXT,
                priority INTEGER,
                response_time REAL,
                status TEXT,
                user_agent TEXT,
                ip_address TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                document_id TEXT,
                business_area TEXT,
                document_type TEXT,
                word_count INTEGER,
                processing_time REAL,
                file_size INTEGER
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                cpu_percent REAL,
                memory_percent REAL,
                disk_percent REAL,
                active_connections INTEGER,
                queue_size INTEGER
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def log_request(self, query: str, business_area: str, document_type: str, 
                   priority: int, response_time: float, status: str, 
                   user_agent: str = None, ip_address: str = None):
        """Log API request to analytics database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO requests (query, business_area, document_type, priority, 
                                response_time, status, user_agent, ip_address)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (query, business_area, document_type, priority, response_time, 
              status, user_agent, ip_address))
        
        conn.commit()
        conn.close()
    
    def log_document(self, document_id: str, business_area: str, document_type: str,
                    word_count: int, processing_time: float, file_size: int):
        """Log document generation to analytics database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO documents (document_id, business_area, document_type,
                                 word_count, processing_time, file_size)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (document_id, business_area, document_type, word_count, 
              processing_time, file_size))
        
        conn.commit()
        conn.close()
    
    def log_performance(self, cpu_percent: float, memory_percent: float, 
                       disk_percent: float, active_connections: int, queue_size: int):
        """Log system performance metrics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO performance (cpu_percent, memory_percent, disk_percent,
                                   active_connections, queue_size)
            VALUES (?, ?, ?, ?, ?)
        ''', (cpu_percent, memory_percent, disk_percent, active_connections, queue_size))
        
        conn.commit()
        conn.close()
    
    def get_request_analytics(self, days: int = 7) -> Dict[str, Any]:
        """Get request analytics for specified period."""
        conn = sqlite3.connect(self.db_path)
        
        # Get date range
        start_date = datetime.now() - timedelta(days=days)
        
        # Total requests
        total_requests = conn.execute('''
            SELECT COUNT(*) FROM requests 
            WHERE timestamp >= ?
        ''', (start_date,)).fetchone()[0]
        
        # Requests by status
        status_counts = dict(conn.execute('''
            SELECT status, COUNT(*) FROM requests 
            WHERE timestamp >= ?
            GROUP BY status
        ''', (start_date,)).fetchall())
        
        # Requests by business area
        area_counts = dict(conn.execute('''
            SELECT business_area, COUNT(*) FROM requests 
            WHERE timestamp >= ?
            GROUP BY business_area
        ''', (start_date,)).fetchall())
        
        # Average response time
        avg_response_time = conn.execute('''
            SELECT AVG(response_time) FROM requests 
            WHERE timestamp >= ? AND response_time IS NOT NULL
        ''', (start_date,)).fetchone()[0] or 0
        
        # Requests by hour
        hourly_requests = dict(conn.execute('''
            SELECT strftime('%H', timestamp) as hour, COUNT(*) 
            FROM requests 
            WHERE timestamp >= ?
            GROUP BY hour
            ORDER BY hour
        ''', (start_date,)).fetchall())
        
        # Top queries
        top_queries = conn.execute('''
            SELECT query, COUNT(*) as count 
            FROM requests 
            WHERE timestamp >= ?
            GROUP BY query
            ORDER BY count DESC
            LIMIT 10
        ''', (start_date,)).fetchall()
        
        conn.close()
        
        return {
            'total_requests': total_requests,
            'status_distribution': status_counts,
            'business_area_distribution': area_counts,
            'average_response_time': round(avg_response_time, 3),
            'hourly_distribution': hourly_requests,
            'top_queries': top_queries
        }
    
    def get_document_analytics(self, days: int = 7) -> Dict[str, Any]:
        """Get document generation analytics."""
        conn = sqlite3.connect(self.db_path)
        
        start_date = datetime.now() - timedelta(days=days)
        
        # Total documents
        total_documents = conn.execute('''
            SELECT COUNT(*) FROM documents 
            WHERE timestamp >= ?
        ''', (start_date,)).fetchone()[0]
        
        # Documents by business area
        area_counts = dict(conn.execute('''
            SELECT business_area, COUNT(*) FROM documents 
            WHERE timestamp >= ?
            GROUP BY business_area
        ''', (start_date,)).fetchall())
        
        # Documents by type
        type_counts = dict(conn.execute('''
            SELECT document_type, COUNT(*) FROM documents 
            WHERE timestamp >= ?
            GROUP BY document_type
        ''', (start_date,)).fetchall())
        
        # Average processing time
        avg_processing_time = conn.execute('''
            SELECT AVG(processing_time) FROM documents 
            WHERE timestamp >= ? AND processing_time IS NOT NULL
        ''', (start_date,)).fetchone()[0] or 0
        
        # Average word count
        avg_word_count = conn.execute('''
            SELECT AVG(word_count) FROM documents 
            WHERE timestamp >= ? AND word_count IS NOT NULL
        ''', (start_date,)).fetchone()[0] or 0
        
        # Total words generated
        total_words = conn.execute('''
            SELECT SUM(word_count) FROM documents 
            WHERE timestamp >= ? AND word_count IS NOT NULL
        ''', (start_date,)).fetchone()[0] or 0
        
        conn.close()
        
        return {
            'total_documents': total_documents,
            'business_area_distribution': area_counts,
            'document_type_distribution': type_counts,
            'average_processing_time': round(avg_processing_time, 3),
            'average_word_count': round(avg_word_count, 0),
            'total_words_generated': total_words
        }
    
    def get_performance_analytics(self, days: int = 7) -> Dict[str, Any]:
        """Get system performance analytics."""
        conn = sqlite3.connect(self.db_path)
        
        start_date = datetime.now() - timedelta(days=days)
        
        # Average CPU usage
        avg_cpu = conn.execute('''
            SELECT AVG(cpu_percent) FROM performance 
            WHERE timestamp >= ?
        ''', (start_date,)).fetchone()[0] or 0
        
        # Average memory usage
        avg_memory = conn.execute('''
            SELECT AVG(memory_percent) FROM performance 
            WHERE timestamp >= ?
        ''', (start_date,)).fetchone()[0] or 0
        
        # Average disk usage
        avg_disk = conn.execute('''
            SELECT AVG(disk_percent) FROM performance 
            WHERE timestamp >= ?
        ''', (start_date,)).fetchone()[0] or 0
        
        # Peak CPU usage
        peak_cpu = conn.execute('''
            SELECT MAX(cpu_percent) FROM performance 
            WHERE timestamp >= ?
        ''', (start_date,)).fetchone()[0] or 0
        
        # Peak memory usage
        peak_memory = conn.execute('''
            SELECT MAX(memory_percent) FROM performance 
            WHERE timestamp >= ?
        ''', (start_date,)).fetchone()[0] or 0
        
        # Performance over time (hourly averages)
        hourly_performance = conn.execute('''
            SELECT strftime('%H', timestamp) as hour, 
                   AVG(cpu_percent) as avg_cpu,
                   AVG(memory_percent) as avg_memory
            FROM performance 
            WHERE timestamp >= ?
            GROUP BY hour
            ORDER BY hour
        ''', (start_date,)).fetchall()
        
        conn.close()
        
        return {
            'average_cpu_usage': round(avg_cpu, 2),
            'average_memory_usage': round(avg_memory, 2),
            'average_disk_usage': round(avg_disk, 2),
            'peak_cpu_usage': round(peak_cpu, 2),
            'peak_memory_usage': round(peak_memory, 2),
            'hourly_performance': hourly_performance
        }
    
    def generate_charts(self, output_dir: str = "analytics_charts"):
        """Generate analytics charts."""
        print("📊 Generating analytics charts...")
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Get analytics data
        request_analytics = self.get_request_analytics()
        document_analytics = self.get_document_analytics()
        performance_analytics = self.get_performance_analytics()
        
        # Set up matplotlib
        plt.style.use('seaborn-v0_8')
        
        # 1. Request Status Distribution
        if request_analytics['status_distribution']:
            plt.figure(figsize=(10, 6))
            statuses = list(request_analytics['status_distribution'].keys())
            counts = list(request_analytics['status_distribution'].values())
            
            plt.pie(counts, labels=statuses, autopct='%1.1f%%', startangle=90)
            plt.title('Request Status Distribution (Last 7 Days)')
            plt.savefig(output_path / 'request_status_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 2. Business Area Distribution
        if request_analytics['business_area_distribution']:
            plt.figure(figsize=(12, 6))
            areas = list(request_analytics['business_area_distribution'].keys())
            counts = list(request_analytics['business_area_distribution'].values())
            
            plt.bar(areas, counts)
            plt.title('Requests by Business Area (Last 7 Days)')
            plt.xlabel('Business Area')
            plt.ylabel('Number of Requests')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(output_path / 'business_area_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. Hourly Request Distribution
        if request_analytics['hourly_distribution']:
            plt.figure(figsize=(14, 6))
            hours = [f"{h:02d}:00" for h in range(24)]
            counts = [request_analytics['hourly_distribution'].get(str(h), 0) for h in range(24)]
            
            plt.plot(hours, counts, marker='o')
            plt.title('Requests by Hour (Last 7 Days)')
            plt.xlabel('Hour')
            plt.ylabel('Number of Requests')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_path / 'hourly_requests.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 4. Document Type Distribution
        if document_analytics['document_type_distribution']:
            plt.figure(figsize=(10, 6))
            types = list(document_analytics['document_type_distribution'].keys())
            counts = list(document_analytics['document_type_distribution'].values())
            
            plt.bar(types, counts)
            plt.title('Documents by Type (Last 7 Days)')
            plt.xlabel('Document Type')
            plt.ylabel('Number of Documents')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(output_path / 'document_type_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 5. Performance Metrics
        if performance_analytics['hourly_performance']:
            plt.figure(figsize=(14, 8))
            
            hours = [f"{h:02d}:00" for h in range(24)]
            cpu_data = [0] * 24
            memory_data = [0] * 24
            
            for hour, avg_cpu, avg_memory in performance_analytics['hourly_performance']:
                hour_idx = int(hour)
                cpu_data[hour_idx] = avg_cpu
                memory_data[hour_idx] = avg_memory
            
            plt.subplot(2, 1, 1)
            plt.plot(hours, cpu_data, marker='o', color='red', label='CPU')
            plt.title('Average CPU Usage by Hour (Last 7 Days)')
            plt.ylabel('CPU Usage (%)')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            plt.subplot(2, 1, 2)
            plt.plot(hours, memory_data, marker='o', color='blue', label='Memory')
            plt.title('Average Memory Usage by Hour (Last 7 Days)')
            plt.xlabel('Hour')
            plt.ylabel('Memory Usage (%)')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(output_path / 'performance_metrics.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"✅ Charts generated in: {output_path}")
        return str(output_path)
    
    def generate_dashboard_report(self) -> str:
        """Generate comprehensive dashboard report."""
        print("📊 Generating dashboard report...")
        
        request_analytics = self.get_request_analytics()
        document_analytics = self.get_document_analytics()
        performance_analytics = self.get_performance_analytics()
        
        report = f"""
BUL Analytics Dashboard Report
=============================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

REQUEST ANALYTICS (Last 7 Days)
-------------------------------
Total Requests: {request_analytics['total_requests']}
Average Response Time: {request_analytics['average_response_time']}s

Status Distribution:
{chr(10).join(f"  {status}: {count}" for status, count in request_analytics['status_distribution'].items())}

Business Area Distribution:
{chr(10).join(f"  {area}: {count}" for area, count in request_analytics['business_area_distribution'].items())}

Top Queries:
{chr(10).join(f"  {count}x: {query[:50]}..." for query, count in request_analytics['top_queries'][:5])}

DOCUMENT ANALYTICS (Last 7 Days)
--------------------------------
Total Documents Generated: {document_analytics['total_documents']}
Total Words Generated: {document_analytics['total_words_generated']:,}
Average Processing Time: {document_analytics['average_processing_time']}s
Average Word Count: {document_analytics['average_word_count']}

Document Type Distribution:
{chr(10).join(f"  {doc_type}: {count}" for doc_type, count in document_analytics['document_type_distribution'].items())}

PERFORMANCE ANALYTICS (Last 7 Days)
-----------------------------------
Average CPU Usage: {performance_analytics['average_cpu_usage']}%
Average Memory Usage: {performance_analytics['average_memory_usage']}%
Average Disk Usage: {performance_analytics['average_disk_usage']}%
Peak CPU Usage: {performance_analytics['peak_cpu_usage']}%
Peak Memory Usage: {performance_analytics['peak_memory_usage']}%

SYSTEM INSIGHTS
---------------
"""
        
        # Add insights
        if request_analytics['total_requests'] > 0:
            success_rate = (request_analytics['status_distribution'].get('completed', 0) / 
                          request_analytics['total_requests']) * 100
            report += f"Success Rate: {success_rate:.1f}%\n"
        
        if document_analytics['total_documents'] > 0:
            avg_docs_per_day = document_analytics['total_documents'] / 7
            report += f"Average Documents per Day: {avg_docs_per_day:.1f}\n"
        
        if performance_analytics['average_cpu_usage'] > 80:
            report += "⚠️  High CPU usage detected\n"
        
        if performance_analytics['average_memory_usage'] > 80:
            report += "⚠️  High memory usage detected\n"
        
        return report
    
    def start_analytics_collection(self, interval: int = 60):
        """Start collecting analytics data."""
        print(f"📊 Starting analytics collection (interval: {interval}s)")
        
        import psutil
        
        while True:
            try:
                # Collect performance metrics
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                
                # Log performance data
                self.log_performance(
                    cpu_percent=cpu_percent,
                    memory_percent=memory.percent,
                    disk_percent=(disk.used / disk.total) * 100,
                    active_connections=0,  # Would need to implement connection tracking
                    queue_size=0  # Would need to implement queue tracking
                )
                
                print(f"📊 Collected metrics: CPU {cpu_percent}%, Memory {memory.percent}%")
                
                # Wait for next collection
                import time
                time.sleep(interval)
                
            except KeyboardInterrupt:
                print("\n⏹️  Analytics collection stopped")
                break
            except Exception as e:
                print(f"❌ Error collecting analytics: {e}")
                import time
                time.sleep(interval)

def main():
    """Main analytics dashboard function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="BUL Analytics Dashboard")
    parser.add_argument("--report", action="store_true", help="Generate analytics report")
    parser.add_argument("--charts", action="store_true", help="Generate analytics charts")
    parser.add_argument("--collect", action="store_true", help="Start data collection")
    parser.add_argument("--interval", type=int, default=60, help="Collection interval in seconds")
    parser.add_argument("--days", type=int, default=7, help="Analytics period in days")
    parser.add_argument("--output", default="analytics_charts", help="Output directory for charts")
    
    args = parser.parse_args()
    
    dashboard = AnalyticsDashboard()
    
    print("📊 BUL Analytics Dashboard")
    print("=" * 40)
    
    if args.report:
        report = dashboard.generate_dashboard_report()
        print(report)
        
        # Save report
        report_file = f"analytics_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        print(f"📄 Report saved to: {report_file}")
    
    if args.charts:
        charts_dir = dashboard.generate_charts(args.output)
        print(f"📊 Charts generated in: {charts_dir}")
    
    if args.collect:
        dashboard.start_analytics_collection(args.interval)
    
    if not any([args.report, args.charts, args.collect]):
        # Show quick analytics
        request_analytics = dashboard.get_request_analytics(args.days)
        document_analytics = dashboard.get_document_analytics(args.days)
        performance_analytics = dashboard.get_performance_analytics(args.days)
        
        print(f"\n📊 Quick Analytics (Last {args.days} days):")
        print(f"Total Requests: {request_analytics['total_requests']}")
        print(f"Total Documents: {document_analytics['total_documents']}")
        print(f"Average CPU: {performance_analytics['average_cpu_usage']}%")
        print(f"Average Memory: {performance_analytics['average_memory_usage']}%")
        
        print(f"\n💡 Use --report for detailed analytics")
        print(f"💡 Use --charts to generate visualizations")
        print(f"💡 Use --collect to start data collection")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
