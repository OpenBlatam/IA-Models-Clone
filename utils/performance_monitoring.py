from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

# Constants
BUFFER_SIZE = 1024

import asyncio
import time
import logging
import psutil
import threading
from typing import Any, Dict, List, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
from datetime import datetime, timedelta
from enum import Enum
import json
import sqlite3
from pathlib import Path
import structlog
from pydantic import BaseModel, Field
import numpy as np
from prometheus_client import Counter, Histogram, Gauge, Summary, generate_latest
from typing import Any, List, Dict, Optional
"""
📊 Performance Monitoring System
================================

Comprehensive performance monitoring with:
- Real-time metrics collection
- Performance alerting
- Resource monitoring
- Database performance tracking
- API endpoint monitoring
- Custom metrics
- Performance dashboards
- Historical data analysis
"""



logger = structlog.get_logger(__name__)

class MetricType(Enum):
    """Metric types"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"

class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class MetricPoint:
    """Single metric data point"""
    timestamp: float
    value: float
    labels: Dict[str, str] = field(default_factory=dict)

@dataclass
class Alert:
    """Performance alert"""
    id: str
    severity: AlertSeverity
    message: str
    metric_name: str
    threshold: float
    current_value: float
    timestamp: float
    acknowledged: bool = False
    resolved: bool = False

class PerformanceMetric:
    """Base class for performance metrics"""
    
    def __init__(self, name: str, metric_type: MetricType, description: str = ""):
        
    """__init__ function."""
self.name = name
        self.metric_type = metric_type
        self.description = description
        self.data_points: deque = deque(maxlen=10000)  # Keep last 10k points
        self.labels: Dict[str, str] = {}
        
    def add_point(self, value: float, labels: Dict[str, str] = None):
        """Add a data point"""
        point = MetricPoint(
            timestamp=time.time(),
            value=value,
            labels=labels or {}
        )
        self.data_points.append(point)
    
    def get_latest_value(self) -> Optional[float]:
        """Get the latest value"""
        if self.data_points:
            return self.data_points[-1].value
        return None
    
    def get_average(self, window_seconds: int = 300) -> float:
        """Get average value over time window"""
        cutoff_time = time.time() - window_seconds
        recent_points = [
            point.value for point in self.data_points 
            if point.timestamp >= cutoff_time
        ]
        return np.mean(recent_points) if recent_points else 0.0
    
    def get_percentile(self, percentile: float, window_seconds: int = 300) -> float:
        """Get percentile value over time window"""
        cutoff_time = time.time() - window_seconds
        recent_points = [
            point.value for point in self.data_points 
            if point.timestamp >= cutoff_time
        ]
        if recent_points:
            return np.percentile(recent_points, percentile)
        return 0.0

class SystemMetrics:
    """System-level performance metrics"""
    
    def __init__(self) -> Any:
        self.cpu_usage = PerformanceMetric("cpu_usage", MetricType.GAUGE, "CPU usage percentage")
        self.memory_usage = PerformanceMetric("memory_usage", MetricType.GAUGE, "Memory usage in MB")
        self.disk_usage = PerformanceMetric("disk_usage", MetricType.GAUGE, "Disk usage percentage")
        self.network_io = PerformanceMetric("network_io", MetricType.COUNTER, "Network I/O in bytes")
        self.process_count = PerformanceMetric("process_count", MetricType.GAUGE, "Number of processes")
        
        # Prometheus metrics
        self.prometheus_metrics = {
            "cpu_usage": Gauge("cpu_usage_percent", "CPU usage percentage"),
            "memory_usage": Gauge("memory_usage_mb", "Memory usage in MB"),
            "disk_usage": Gauge("disk_usage_percent", "Disk usage percentage"),
            "network_io": Counter("network_io_bytes", "Network I/O in bytes"),
            "process_count": Gauge("process_count", "Number of processes")
        }
    
    def collect_metrics(self) -> Any:
        """Collect current system metrics"""
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        self.cpu_usage.add_point(cpu_percent)
        self.prometheus_metrics["cpu_usage"].set(cpu_percent)
        
        # Memory usage
        memory = psutil.virtual_memory()
        memory_mb = memory.used / 1024 / 1024
        self.memory_usage.add_point(memory_mb)
        self.prometheus_metrics["memory_usage"].set(memory_mb)
        
        # Disk usage
        disk = psutil.disk_usage('/')
        disk_percent = (disk.used / disk.total) * 100
        self.disk_usage.add_point(disk_percent)
        self.prometheus_metrics["disk_usage"].set(disk_percent)
        
        # Network I/O
        network = psutil.net_io_counters()
        network_bytes = network.bytes_sent + network.bytes_recv
        self.network_io.add_point(network_bytes)
        self.prometheus_metrics["network_io"].inc(network_bytes)
        
        # Process count
        process_count = len(psutil.pids())
        self.process_count.add_point(process_count)
        self.prometheus_metrics["process_count"].set(process_count)

class ApplicationMetrics:
    """Application-level performance metrics"""
    
    def __init__(self) -> Any:
        self.request_count = PerformanceMetric("request_count", MetricType.COUNTER, "Total requests")
        self.request_duration = PerformanceMetric("request_duration", MetricType.HISTOGRAM, "Request duration")
        self.error_count = PerformanceMetric("error_count", MetricType.COUNTER, "Total errors")
        self.active_connections = PerformanceMetric("active_connections", MetricType.GAUGE, "Active connections")
        self.cache_hit_rate = PerformanceMetric("cache_hit_rate", MetricType.GAUGE, "Cache hit rate")
        self.database_queries = PerformanceMetric("database_queries", MetricType.COUNTER, "Database queries")
        self.database_query_duration = PerformanceMetric("database_query_duration", MetricType.HISTOGRAM, "Database query duration")
        
        # Prometheus metrics
        self.prometheus_metrics = {
            "request_count": Counter("requests_total", "Total requests"),
            "request_duration": Histogram("request_duration_seconds", "Request duration"),
            "error_count": Counter("errors_total", "Total errors"),
            "active_connections": Gauge("active_connections", "Active connections"),
            "cache_hit_rate": Gauge("cache_hit_rate", "Cache hit rate"),
            "database_queries": Counter("database_queries_total", "Database queries"),
            "database_query_duration": Histogram("database_query_duration_seconds", "Database query duration")
        }
    
    def record_request(self, duration: float, status_code: int = 200):
        """Record a request"""
        self.request_count.add_point(1.0)
        self.request_duration.add_point(duration)
        self.prometheus_metrics["request_count"].inc()
        self.prometheus_metrics["request_duration"].observe(duration)
        
        if status_code >= 400:
            self.error_count.add_point(1.0)
            self.prometheus_metrics["error_count"].inc()
    
    def record_database_query(self, duration: float):
        """Record a database query"""
        self.database_queries.add_point(1.0)
        self.database_query_duration.add_point(duration)
        self.prometheus_metrics["database_queries"].inc()
        self.prometheus_metrics["database_query_duration"].observe(duration)
    
    def update_cache_hit_rate(self, hit_rate: float):
        """Update cache hit rate"""
        self.cache_hit_rate.add_point(hit_rate)
        self.prometheus_metrics["cache_hit_rate"].set(hit_rate)
    
    def update_active_connections(self, count: int):
        """Update active connections"""
        self.active_connections.add_point(count)
        self.prometheus_metrics["active_connections"].set(count)

class AlertManager:
    """Manages performance alerts"""
    
    def __init__(self) -> Any:
        self.alerts: Dict[str, Alert] = {}
        self.alert_rules: List[Dict[str, Any]] = []
        self.alert_handlers: List[Callable] = []
        
        # Default alert rules
        self._setup_default_rules()
    
    def _setup_default_rules(self) -> Any:
        """Setup default alert rules"""
        self.add_alert_rule(
            name="high_cpu_usage",
            metric_name="cpu_usage",
            threshold=80.0,
            severity=AlertSeverity.WARNING,
            message="CPU usage is high"
        )
        
        self.add_alert_rule(
            name="high_memory_usage",
            metric_name="memory_usage",
            threshold=85.0,
            severity=AlertSeverity.WARNING,
            message="Memory usage is high"
        )
        
        self.add_alert_rule(
            name="high_disk_usage",
            metric_name="disk_usage",
            threshold=90.0,
            severity=AlertSeverity.WARNING,
            message="Disk usage is high"
        )
        
        self.add_alert_rule(
            name="high_error_rate",
            metric_name="error_rate",
            threshold=0.05,
            severity=AlertSeverity.ERROR,
            message="Error rate is high"
        )
    
    def add_alert_rule(self, name: str, metric_name: str, threshold: float, 
                      severity: AlertSeverity, message: str):
        """Add an alert rule"""
        self.alert_rules.append({
            "name": name,
            "metric_name": metric_name,
            "threshold": threshold,
            "severity": severity,
            "message": message
        })
    
    def add_alert_handler(self, handler: Callable):
        """Add an alert handler"""
        self.alert_handlers.append(handler)
    
    def check_alerts(self, metrics: Dict[str, PerformanceMetric]):
        """Check for alerts based on current metrics"""
        for rule in self.alert_rules:
            metric = metrics.get(rule["metric_name"])
            if metric is None:
                continue
            
            current_value = metric.get_latest_value()
            if current_value is None:
                continue
            
            # Check if threshold is exceeded
            if current_value > rule["threshold"]:
                alert_id = f"{rule['name']}_{int(time.time())}"
                
                # Create alert if it doesn't exist
                if alert_id not in self.alerts:
                    alert = Alert(
                        id=alert_id,
                        severity=rule["severity"],
                        message=rule["message"],
                        metric_name=rule["metric_name"],
                        threshold=rule["threshold"],
                        current_value=current_value,
                        timestamp=time.time()
                    )
                    
                    self.alerts[alert_id] = alert
                    
                    # Trigger alert handlers
                    for handler in self.alert_handlers:
                        try:
                            handler(alert)
                        except Exception as e:
                            logger.error(f"Alert handler error: {e}")
                    
                    logger.warning(
                        f"Alert triggered: {alert.message}",
                        metric=alert.metric_name,
                        value=alert.current_value,
                        threshold=alert.threshold,
                        severity=alert.severity.value
                    )
    
    def acknowledge_alert(self, alert_id: str):
        """Acknowledge an alert"""
        if alert_id in self.alerts:
            self.alerts[alert_id].acknowledged = True
    
    def resolve_alert(self, alert_id: str):
        """Resolve an alert"""
        if alert_id in self.alerts:
            self.alerts[alert_id].resolved = True
    
    def get_active_alerts(self) -> List[Alert]:
        """Get active (unresolved) alerts"""
        return [alert for alert in self.alerts.values() if not alert.resolved]
    
    def get_alerts_by_severity(self, severity: AlertSeverity) -> List[Alert]:
        """Get alerts by severity"""
        return [alert for alert in self.alerts.values() if alert.severity == severity]

class MetricsStorage:
    """Stores metrics data persistently"""
    
    def __init__(self, db_path: str = "performance_metrics.db"):
        
    """__init__ function."""
self.db_path = db_path
        self._init_database()
    
    def _init_database(self) -> Any:
        """Initialize the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                timestamp REAL NOT NULL,
                value REAL NOT NULL,
                labels TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create alerts table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                id TEXT PRIMARY KEY,
                severity TEXT NOT NULL,
                message TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                threshold REAL NOT NULL,
                current_value REAL NOT NULL,
                timestamp REAL NOT NULL,
                acknowledged BOOLEAN DEFAULT FALSE,
                resolved BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_metrics_name_timestamp ON metrics(name, timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_alerts_severity ON alerts(severity)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_alerts_resolved ON alerts(resolved)")
        
        conn.commit()
        conn.close()
    
    def store_metric(self, metric: PerformanceMetric):
        """Store metric data points"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for point in metric.data_points:
            cursor.execute("""
                INSERT INTO metrics (name, timestamp, value, labels)
                VALUES (?, ?, ?, ?)
            """, (
                metric.name,
                point.timestamp,
                point.value,
                json.dumps(point.labels)
            ))
        
        conn.commit()
        conn.close()
    
    def store_alert(self, alert: Alert):
        """Store alert"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO alerts 
            (id, severity, message, metric_name, threshold, current_value, timestamp, acknowledged, resolved)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            alert.id,
            alert.severity.value,
            alert.message,
            alert.metric_name,
            alert.threshold,
            alert.current_value,
            alert.timestamp,
            alert.acknowledged,
            alert.resolved
        ))
        
        conn.commit()
        conn.close()
    
    def get_metrics(self, metric_name: str, start_time: float = None, end_time: float = None) -> List[MetricPoint]:
        """Get metrics from storage"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = "SELECT timestamp, value, labels FROM metrics WHERE name = ?"
        params = [metric_name]
        
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time)
        
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time)
        
        query += " ORDER BY timestamp"
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        return [
            MetricPoint(
                timestamp=row[0],
                value=row[1],
                labels=json.loads(row[2]) if row[2] else {}
            )
            for row in rows
        ]
    
    def get_alerts(self, severity: AlertSeverity = None, resolved: bool = None) -> List[Alert]:
        """Get alerts from storage"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = "SELECT * FROM alerts WHERE 1=1"
        params = []
        
        if severity:
            query += " AND severity = ?"
            params.append(severity.value)
        
        if resolved is not None:
            query += " AND resolved = ?"
            params.append(resolved)
        
        query += " ORDER BY timestamp DESC"
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        return [
            Alert(
                id=row[0],
                severity=AlertSeverity(row[1]),
                message=row[2],
                metric_name=row[3],
                threshold=row[4],
                current_value=row[5],
                timestamp=row[6],
                acknowledged=bool(row[7]),
                resolved=bool(row[8])
            )
            for row in rows
        ]

class PerformanceMonitor:
    """
    Main performance monitoring system that orchestrates all monitoring components.
    """
    
    def __init__(self, storage_path: str = "performance_metrics.db"):
        
    """__init__ function."""
self.system_metrics = SystemMetrics()
        self.application_metrics = ApplicationMetrics()
        self.alert_manager = AlertManager()
        self.storage = MetricsStorage(storage_path)
        
        # Monitoring state
        self.is_running = False
        self.monitoring_task = None
        self.collection_interval = 60  # seconds
        
        # Custom metrics
        self.custom_metrics: Dict[str, PerformanceMetric] = {}
        
        # Setup default alert handlers
        self._setup_default_handlers()
    
    def _setup_default_handlers(self) -> Any:
        """Setup default alert handlers"""
        def log_alert(alert: Alert):
            
    """log_alert function."""
logger.warning(
                f"Performance alert: {alert.message}",
                severity=alert.severity.value,
                metric=alert.metric_name,
                value=alert.current_value,
                threshold=alert.threshold
            )
        
        def store_alert(alert: Alert):
            
    """store_alert function."""
self.storage.store_alert(alert)
        
        self.alert_manager.add_alert_handler(log_alert)
        self.alert_manager.add_alert_handler(store_alert)
    
    async def start(self) -> Any:
        """Start performance monitoring"""
        if self.is_running:
            return
        
        self.is_running = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        logger.info("Performance monitoring started")
    
    async def stop(self) -> Any:
        """Stop performance monitoring"""
        if not self.is_running:
            return
        
        self.is_running = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Performance monitoring stopped")
    
    async def _monitoring_loop(self) -> Any:
        """Main monitoring loop"""
        while self.is_running:
            try:
                # Collect system metrics
                self.system_metrics.collect_metrics()
                
                # Store metrics
                self._store_all_metrics()
                
                # Check for alerts
                all_metrics = self._get_all_metrics()
                self.alert_manager.check_alerts(all_metrics)
                
                # Log summary
                self._log_monitoring_summary()
                
                await asyncio.sleep(self.collection_interval)
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(10)  # Wait 10 seconds on error
    
    def _store_all_metrics(self) -> Any:
        """Store all metrics to database"""
        # Store system metrics
        for metric_name in ["cpu_usage", "memory_usage", "disk_usage", "network_io", "process_count"]:
            metric = getattr(self.system_metrics, metric_name)
            self.storage.store_metric(metric)
        
        # Store application metrics
        for metric_name in ["request_count", "request_duration", "error_count", "active_connections", "cache_hit_rate", "database_queries", "database_query_duration"]:
            metric = getattr(self.application_metrics, metric_name)
            self.storage.store_metric(metric)
        
        # Store custom metrics
        for metric in self.custom_metrics.values():
            self.storage.store_metric(metric)
    
    def _get_all_metrics(self) -> Dict[str, PerformanceMetric]:
        """Get all metrics"""
        metrics = {}
        
        # System metrics
        for metric_name in ["cpu_usage", "memory_usage", "disk_usage", "network_io", "process_count"]:
            metrics[metric_name] = getattr(self.system_metrics, metric_name)
        
        # Application metrics
        for metric_name in ["request_count", "request_duration", "error_count", "active_connections", "cache_hit_rate", "database_queries", "database_query_duration"]:
            metrics[metric_name] = getattr(self.application_metrics, metric_name)
        
        # Custom metrics
        metrics.update(self.custom_metrics)
        
        return metrics
    
    def _log_monitoring_summary(self) -> Any:
        """Log monitoring summary"""
        cpu_usage = self.system_metrics.cpu_usage.get_latest_value() or 0
        memory_usage = self.system_metrics.memory_usage.get_latest_value() or 0
        request_count = self.application_metrics.request_count.get_latest_value() or 0
        error_count = self.application_metrics.error_count.get_latest_value() or 0
        
        logger.info(
            "Performance monitoring summary",
            cpu_usage=f"{cpu_usage:.1f}%",
            memory_usage=f"{memory_usage:.1f}MB",
            request_count=request_count,
            error_count=error_count,
            active_alerts=len(self.alert_manager.get_active_alerts())
        )
    
    def add_custom_metric(self, name: str, metric_type: MetricType, description: str = ""):
        """Add a custom metric"""
        self.custom_metrics[name] = PerformanceMetric(name, metric_type, description)
    
    def record_custom_metric(self, name: str, value: float, labels: Dict[str, str] = None):
        """Record a custom metric"""
        if name in self.custom_metrics:
            self.custom_metrics[name].add_point(value, labels)
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get metrics summary"""
        all_metrics = self._get_all_metrics()
        
        summary = {}
        for name, metric in all_metrics.items():
            latest_value = metric.get_latest_value()
            if latest_value is not None:
                summary[name] = {
                    "latest_value": latest_value,
                    "average_5min": metric.get_average(300),
                    "average_1hour": metric.get_average(3600),
                    "p95_5min": metric.get_percentile(95, 300),
                    "p99_5min": metric.get_percentile(99, 300)
                }
        
        return summary
    
    def get_prometheus_metrics(self) -> str:
        """Get Prometheus metrics"""
        return generate_latest()
    
    def get_alerts_summary(self) -> Dict[str, Any]:
        """Get alerts summary"""
        active_alerts = self.alert_manager.get_active_alerts()
        
        return {
            "total_alerts": len(active_alerts),
            "alerts_by_severity": {
                severity.value: len(self.alert_manager.get_alerts_by_severity(severity))
                for severity in AlertSeverity
            },
            "recent_alerts": [
                {
                    "id": alert.id,
                    "severity": alert.severity.value,
                    "message": alert.message,
                    "metric": alert.metric_name,
                    "value": alert.current_value,
                    "threshold": alert.threshold,
                    "timestamp": alert.timestamp
                }
                for alert in active_alerts[-10:]  # Last 10 alerts
            ]
        }

# Global monitor instance
_global_monitor = None

def get_monitor() -> PerformanceMonitor:
    """Get global monitor instance"""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = PerformanceMonitor()
    return _global_monitor

def monitor_performance(metric_name: str = None):
    """Decorator for monitoring function performance"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            monitor = get_monitor()
            start_time = time.time()
            
            try:
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                duration = time.time() - start_time
                
                # Record metrics
                monitor.application_metrics.record_request(duration)
                
                if metric_name:
                    monitor.record_custom_metric(metric_name, duration)
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                monitor.application_metrics.record_request(duration, status_code=500)
                raise
        
        return wrapper
    return decorator

# Example usage
async def example_usage():
    """Example usage of performance monitoring"""
    
    # Create monitor
    monitor = PerformanceMonitor()
    
    # Add custom metric
    monitor.add_custom_metric("custom_operation_duration", MetricType.HISTOGRAM, "Custom operation duration")
    
    # Start monitoring
    await monitor.start()
    
    # Example monitored function
    @monitor_performance("example_function")
    async def example_function():
        
    """example_function function."""
await asyncio.sleep(0.1)
        return "example result"
    
    # Execute function
    result = await example_function()
    
    # Get metrics summary
    summary = monitor.get_metrics_summary()
    logger.info("Metrics summary", summary=summary)
    
    # Get alerts summary
    alerts = monitor.get_alerts_summary()
    logger.info("Alerts summary", alerts=alerts)
    
    # Stop monitoring
    await monitor.stop()
    
    return result

match __name__:
    case "__main__":
    asyncio.run(example_usage()) 