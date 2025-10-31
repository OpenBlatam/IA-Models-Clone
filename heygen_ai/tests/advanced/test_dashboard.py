"""
Advanced test monitoring dashboard for HeyGen AI system.
Real-time monitoring, analytics, and reporting capabilities.
"""

import json
import time
import threading
import asyncio
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import psutil
import sqlite3
from collections import defaultdict, deque
import logging

@dataclass
class TestEvent:
    """Represents a test event for monitoring."""
    event_id: str
    test_name: str
    event_type: str  # started, completed, failed, error
    timestamp: datetime
    duration: Optional[float] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TestMetrics:
    """Test execution metrics."""
    test_name: str
    total_runs: int = 0
    successful_runs: int = 0
    failed_runs: int = 0
    error_runs: int = 0
    total_duration: float = 0.0
    average_duration: float = 0.0
    min_duration: float = float('inf')
    max_duration: float = 0.0
    last_run: Optional[datetime] = None
    success_rate: float = 0.0
    trend: str = "stable"  # improving, declining, stable

class TestEventLogger:
    """Logs and stores test events."""
    
    def __init__(self, db_path: str = "test_monitoring.db"):
        self.db_path = db_path
        self.connection = None
        self._init_database()
        self.event_queue = deque(maxlen=10000)  # Keep last 10k events in memory
        self.lock = threading.Lock()
    
    def _init_database(self):
        """Initialize SQLite database for event storage."""
        self.connection = sqlite3.connect(self.db_path, check_same_thread=False)
        cursor = self.connection.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS test_events (
                event_id TEXT PRIMARY KEY,
                test_name TEXT NOT NULL,
                event_type TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                duration REAL,
                result TEXT,
                error TEXT,
                metadata TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_test_name ON test_events(test_name)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_timestamp ON test_events(timestamp)
        ''')
        
        self.connection.commit()
    
    def log_event(self, event: TestEvent):
        """Log a test event."""
        with self.lock:
            # Add to memory queue
            self.event_queue.append(event)
            
            # Store in database
            cursor = self.connection.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO test_events 
                (event_id, test_name, event_type, timestamp, duration, result, error, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                event.event_id,
                event.test_name,
                event.event_type,
                event.timestamp.isoformat(),
                event.duration,
                json.dumps(event.result) if event.result else None,
                event.error,
                json.dumps(event.metadata)
            ))
            self.connection.commit()
    
    def get_recent_events(self, limit: int = 100) -> List[TestEvent]:
        """Get recent test events."""
        with self.lock:
            return list(self.event_queue)[-limit:]
    
    def get_events_by_test(self, test_name: str, limit: int = 100) -> List[TestEvent]:
        """Get events for a specific test."""
        cursor = self.connection.cursor()
        cursor.execute('''
            SELECT event_id, test_name, event_type, timestamp, duration, result, error, metadata
            FROM test_events
            WHERE test_name = ?
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (test_name, limit))
        
        events = []
        for row in cursor.fetchall():
            event = TestEvent(
                event_id=row[0],
                test_name=row[1],
                event_type=row[2],
                timestamp=datetime.fromisoformat(row[3]),
                duration=row[4],
                result=json.loads(row[5]) if row[5] else None,
                error=row[6],
                metadata=json.loads(row[7]) if row[7] else {}
            )
            events.append(event)
        
        return events
    
    def get_events_by_timeframe(self, start_time: datetime, end_time: datetime) -> List[TestEvent]:
        """Get events within a time frame."""
        cursor = self.connection.cursor()
        cursor.execute('''
            SELECT event_id, test_name, event_type, timestamp, duration, result, error, metadata
            FROM test_events
            WHERE timestamp BETWEEN ? AND ?
            ORDER BY timestamp DESC
        ''', (start_time.isoformat(), end_time.isoformat()))
        
        events = []
        for row in cursor.fetchall():
            event = TestEvent(
                event_id=row[0],
                test_name=row[1],
                event_type=row[2],
                timestamp=datetime.fromisoformat(row[3]),
                duration=row[4],
                result=json.loads(row[5]) if row[5] else None,
                error=row[6],
                metadata=json.loads(row[7]) if row[7] else {}
            )
            events.append(event)
        
        return events

class TestMetricsCalculator:
    """Calculates test metrics from events."""
    
    def __init__(self, event_logger: TestEventLogger):
        self.event_logger = event_logger
    
    def calculate_metrics(self, test_name: str, timeframe_hours: int = 24) -> TestMetrics:
        """Calculate metrics for a specific test."""
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=timeframe_hours)
        
        events = self.event_logger.get_events_by_timeframe(start_time, end_time)
        test_events = [e for e in events if e.test_name == test_name]
        
        if not test_events:
            return TestMetrics(test_name=test_name)
        
        # Group events by test run
        test_runs = defaultdict(list)
        for event in test_events:
            # Use a simple grouping based on timestamp proximity
            run_key = event.timestamp.strftime("%Y%m%d_%H%M%S")
            test_runs[run_key].append(event)
        
        metrics = TestMetrics(test_name=test_name)
        
        for run_events in test_runs.values():
            metrics.total_runs += 1
            
            # Find start and end events
            start_event = next((e for e in run_events if e.event_type == "started"), None)
            end_event = next((e for e in run_events if e.event_type in ["completed", "failed", "error"]), None)
            
            if start_event and end_event:
                duration = (end_event.timestamp - start_event.timestamp).total_seconds()
                metrics.total_duration += duration
                metrics.min_duration = min(metrics.min_duration, duration)
                metrics.max_duration = max(metrics.max_duration, duration)
                metrics.last_run = end_event.timestamp
                
                if end_event.event_type == "completed":
                    metrics.successful_runs += 1
                elif end_event.event_type == "failed":
                    metrics.failed_runs += 1
                elif end_event.event_type == "error":
                    metrics.error_runs += 1
        
        # Calculate derived metrics
        if metrics.total_runs > 0:
            metrics.average_duration = metrics.total_duration / metrics.total_runs
            metrics.success_rate = (metrics.successful_runs / metrics.total_runs) * 100
        
        # Calculate trend (simplified)
        metrics.trend = self._calculate_trend(test_events)
        
        return metrics
    
    def _calculate_trend(self, events: List[TestEvent]) -> str:
        """Calculate trend based on recent performance."""
        if len(events) < 10:
            return "stable"
        
        # Simple trend calculation based on success rate over time
        recent_events = sorted(events, key=lambda e: e.timestamp)[-10:]
        recent_successes = sum(1 for e in recent_events if e.event_type == "completed")
        recent_total = len([e for e in recent_events if e.event_type in ["completed", "failed", "error"]])
        
        if recent_total == 0:
            return "stable"
        
        recent_success_rate = recent_successes / recent_total
        
        # Compare with earlier period
        earlier_events = sorted(events, key=lambda e: e.timestamp)[-20:-10]
        earlier_successes = sum(1 for e in earlier_events if e.event_type == "completed")
        earlier_total = len([e for e in earlier_events if e.event_type in ["completed", "failed", "error"]])
        
        if earlier_total == 0:
            return "stable"
        
        earlier_success_rate = earlier_successes / earlier_total
        
        if recent_success_rate > earlier_success_rate + 0.1:
            return "improving"
        elif recent_success_rate < earlier_success_rate - 0.1:
            return "declining"
        else:
            return "stable"

class SystemResourceMonitor:
    """Monitors system resources during test execution."""
    
    def __init__(self):
        self.cpu_samples = deque(maxlen=1000)
        self.memory_samples = deque(maxlen=1000)
        self.monitoring = False
        self.monitor_thread = None
    
    def start_monitoring(self):
        """Start resource monitoring."""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_resources, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop resource monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
    
    def _monitor_resources(self):
        """Monitor system resources."""
        while self.monitoring:
            try:
                cpu_percent = psutil.cpu_percent(interval=1.0)
                memory_percent = psutil.virtual_memory().percent
                
                self.cpu_samples.append({
                    'timestamp': datetime.now(),
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory_percent
                })
                
            except Exception as e:
                logging.error(f"Resource monitoring error: {e}")
    
    def get_current_stats(self) -> Dict[str, Any]:
        """Get current resource statistics."""
        if not self.cpu_samples:
            return {}
        
        recent_samples = list(self.cpu_samples)[-10:]  # Last 10 samples
        
        cpu_values = [s['cpu_percent'] for s in recent_samples]
        memory_values = [s['memory_percent'] for s in recent_samples]
        
        return {
            'cpu': {
                'current': cpu_values[-1] if cpu_values else 0,
                'average': sum(cpu_values) / len(cpu_values) if cpu_values else 0,
                'max': max(cpu_values) if cpu_values else 0
            },
            'memory': {
                'current': memory_values[-1] if memory_values else 0,
                'average': sum(memory_values) / len(memory_values) if memory_values else 0,
                'max': max(memory_values) if memory_values else 0
            }
        }

class TestDashboard:
    """Main test monitoring dashboard."""
    
    def __init__(self, db_path: str = "test_monitoring.db"):
        self.event_logger = TestEventLogger(db_path)
        self.metrics_calculator = TestMetricsCalculator(self.event_logger)
        self.resource_monitor = SystemResourceMonitor()
        self.dashboard_data = {}
        self.update_interval = 5.0  # seconds
        self.running = False
        self.update_thread = None
    
    def start_dashboard(self):
        """Start the monitoring dashboard."""
        print("🚀 Starting Test Monitoring Dashboard")
        print("=" * 50)
        
        self.running = True
        self.resource_monitor.start_monitoring()
        
        # Start dashboard update thread
        self.update_thread = threading.Thread(target=self._update_dashboard, daemon=True)
        self.update_thread.start()
        
        # Start web interface (if available)
        self._start_web_interface()
    
    def stop_dashboard(self):
        """Stop the monitoring dashboard."""
        self.running = False
        self.resource_monitor.stop_monitoring()
        
        if self.update_thread:
            self.update_thread.join()
    
    def _update_dashboard(self):
        """Update dashboard data periodically."""
        while self.running:
            try:
                self._refresh_dashboard_data()
                time.sleep(self.update_interval)
            except Exception as e:
                logging.error(f"Dashboard update error: {e}")
                time.sleep(self.update_interval)
    
    def _refresh_dashboard_data(self):
        """Refresh dashboard data."""
        # Get recent events
        recent_events = self.event_logger.get_recent_events(100)
        
        # Get unique test names
        test_names = list(set(e.test_name for e in recent_events))
        
        # Calculate metrics for each test
        test_metrics = {}
        for test_name in test_names:
            metrics = self.metrics_calculator.calculate_metrics(test_name)
            test_metrics[test_name] = metrics
        
        # Get system resources
        resource_stats = self.resource_monitor.get_current_stats()
        
        # Update dashboard data
        self.dashboard_data = {
            'timestamp': datetime.now().isoformat(),
            'recent_events': [
                {
                    'test_name': e.test_name,
                    'event_type': e.event_type,
                    'timestamp': e.timestamp.isoformat(),
                    'duration': e.duration
                }
                for e in recent_events[-20:]  # Last 20 events
            ],
            'test_metrics': {
                name: {
                    'total_runs': metrics.total_runs,
                    'successful_runs': metrics.successful_runs,
                    'failed_runs': metrics.failed_runs,
                    'error_runs': metrics.error_runs,
                    'success_rate': metrics.success_rate,
                    'average_duration': metrics.average_duration,
                    'trend': metrics.trend,
                    'last_run': metrics.last_run.isoformat() if metrics.last_run else None
                }
                for name, metrics in test_metrics.items()
            },
            'system_resources': resource_stats,
            'summary': {
                'total_tests': len(test_names),
                'active_tests': len([e for e in recent_events if e.event_type == "started"]),
                'recent_failures': len([e for e in recent_events if e.event_type in ["failed", "error"]]),
                'overall_success_rate': self._calculate_overall_success_rate(test_metrics)
            }
        }
    
    def _calculate_overall_success_rate(self, test_metrics: Dict[str, TestMetrics]) -> float:
        """Calculate overall success rate."""
        if not test_metrics:
            return 0.0
        
        total_runs = sum(m.total_runs for m in test_metrics.values())
        successful_runs = sum(m.successful_runs for m in test_metrics.values())
        
        return (successful_runs / total_runs * 100) if total_runs > 0 else 0.0
    
    def _start_web_interface(self):
        """Start web interface for dashboard."""
        try:
            import http.server
            import socketserver
            import webbrowser
            from urllib.parse import urlparse
            
            # Create simple HTTP server
            class DashboardHandler(http.server.SimpleHTTPRequestHandler):
                def do_GET(self):
                    if self.path == '/api/dashboard':
                        self.send_response(200)
                        self.send_header('Content-type', 'application/json')
                        self.send_header('Access-Control-Allow-Origin', '*')
                        self.end_headers()
                        self.wfile.write(json.dumps(self.server.dashboard_data).encode())
                    else:
                        super().do_GET()
            
            # Start server
            port = 8080
            with socketserver.TCPServer(("", port), DashboardHandler) as httpd:
                httpd.dashboard_data = self.dashboard_data
                print(f"🌐 Dashboard available at: http://localhost:{port}")
                print("   API endpoint: http://localhost:8080/api/dashboard")
                
                # Start server in background
                server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
                server_thread.start()
                
        except ImportError:
            print("⚠️  Web interface not available (missing http.server)")
        except Exception as e:
            print(f"⚠️  Web interface error: {e}")
    
    def log_test_start(self, test_name: str, metadata: Dict[str, Any] = None):
        """Log test start event."""
        event = TestEvent(
            event_id=f"{test_name}_{int(time.time() * 1000)}",
            test_name=test_name,
            event_type="started",
            timestamp=datetime.now(),
            metadata=metadata or {}
        )
        self.event_logger.log_event(event)
    
    def log_test_completion(self, test_name: str, duration: float, result: Any = None, metadata: Dict[str, Any] = None):
        """Log test completion event."""
        event = TestEvent(
            event_id=f"{test_name}_{int(time.time() * 1000)}",
            test_name=test_name,
            event_type="completed",
            timestamp=datetime.now(),
            duration=duration,
            result=result,
            metadata=metadata or {}
        )
        self.event_logger.log_event(event)
    
    def log_test_failure(self, test_name: str, duration: float, error: str, metadata: Dict[str, Any] = None):
        """Log test failure event."""
        event = TestEvent(
            event_id=f"{test_name}_{int(time.time() * 1000)}",
            test_name=test_name,
            event_type="failed",
            timestamp=datetime.now(),
            duration=duration,
            error=error,
            metadata=metadata or {}
        )
        self.event_logger.log_event(event)
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get current dashboard data."""
        return self.dashboard_data
    
    def print_dashboard(self):
        """Print current dashboard status."""
        if not self.dashboard_data:
            print("📊 Dashboard data not available")
            return
        
        data = self.dashboard_data
        
        print("\n" + "=" * 60)
        print("📊 TEST MONITORING DASHBOARD")
        print("=" * 60)
        
        # Summary
        summary = data.get('summary', {})
        print(f"📈 Summary:")
        print(f"   Total Tests: {summary.get('total_tests', 0)}")
        print(f"   Active Tests: {summary.get('active_tests', 0)}")
        print(f"   Recent Failures: {summary.get('recent_failures', 0)}")
        print(f"   Overall Success Rate: {summary.get('overall_success_rate', 0):.1f}%")
        
        # System Resources
        resources = data.get('system_resources', {})
        if resources:
            print(f"\n💻 System Resources:")
            cpu = resources.get('cpu', {})
            memory = resources.get('memory', {})
            print(f"   CPU: {cpu.get('current', 0):.1f}% (avg: {cpu.get('average', 0):.1f}%)")
            print(f"   Memory: {memory.get('current', 0):.1f}% (avg: {memory.get('average', 0):.1f}%)")
        
        # Test Metrics
        test_metrics = data.get('test_metrics', {})
        if test_metrics:
            print(f"\n🧪 Test Metrics:")
            for test_name, metrics in test_metrics.items():
                trend_icon = {"improving": "📈", "declining": "📉", "stable": "➡️"}.get(metrics.get('trend', 'stable'), "➡️")
                print(f"   {trend_icon} {test_name}: {metrics.get('success_rate', 0):.1f}% success ({metrics.get('total_runs', 0)} runs)")
        
        # Recent Events
        recent_events = data.get('recent_events', [])
        if recent_events:
            print(f"\n📋 Recent Events:")
            for event in recent_events[-5:]:  # Last 5 events
                event_icon = {"started": "▶️", "completed": "✅", "failed": "❌", "error": "⚠️"}.get(event.get('event_type', ''), "❓")
                duration_str = f" ({event.get('duration', 0):.2f}s)" if event.get('duration') else ""
                print(f"   {event_icon} {event.get('test_name', 'Unknown')}: {event.get('event_type', 'Unknown')}{duration_str}")
        
        print("=" * 60)

# Example usage
def demo_test_dashboard():
    """Demonstrate test dashboard functionality."""
    print("🔄 Test Dashboard Demo")
    print("=" * 30)
    
    # Create dashboard
    dashboard = TestDashboard()
    
    # Start dashboard
    dashboard.start_dashboard()
    
    # Simulate some test events
    test_names = ["test_basic", "test_performance", "test_integration", "test_unit"]
    
    for i in range(10):
        test_name = test_names[i % len(test_names)]
        
        # Log test start
        dashboard.log_test_start(test_name, {"iteration": i})
        
        # Simulate test execution
        time.sleep(0.1)
        
        # Log test completion (with some failures)
        if i % 7 == 0:  # 1 in 7 tests fail
            dashboard.log_test_failure(test_name, 0.1, f"Test failed at iteration {i}")
        else:
            dashboard.log_test_completion(test_name, 0.1, f"Result {i}")
        
        # Print dashboard every few iterations
        if i % 3 == 0:
            dashboard.print_dashboard()
            time.sleep(1)
    
    # Stop dashboard
    dashboard.stop_dashboard()
    
    print("\n✅ Dashboard demo completed")

if __name__ == "__main__":
    # Run demo
    demo_test_dashboard()
