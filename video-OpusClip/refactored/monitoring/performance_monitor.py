"""
Performance Monitor

Advanced performance monitoring and metrics collection for the Ultimate Opus Clip system.
Provides real-time monitoring, alerting, and performance optimization insights.
"""

from __future__ import annotations
from typing import Dict, Any, Optional, List, Union, Callable
import asyncio
import time
import psutil
import threading
from dataclasses import dataclass, field
from enum import Enum
import structlog
from datetime import datetime, timedelta
import json
from pathlib import Path
import sqlite3
from collections import defaultdict, deque
import numpy as np
from concurrent.futures import ThreadPoolExecutor

logger = structlog.get_logger("performance_monitor")

class MetricType(Enum):
    """Types of performance metrics."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"

class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class Metric:
    """Performance metric data point."""
    name: str
    value: float
    metric_type: MetricType
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Alert:
    """Performance alert."""
    alert_id: str
    name: str
    level: AlertLevel
    message: str
    metric_name: str
    threshold: float
    current_value: float
    timestamp: datetime
    resolved: bool = False
    resolved_at: Optional[datetime] = None

@dataclass
class PerformanceSnapshot:
    """System performance snapshot."""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: Dict[str, float]
    process_count: int
    active_connections: int
    queue_sizes: Dict[str, int]
    custom_metrics: Dict[str, float] = field(default_factory=dict)

class MetricsCollector:
    """Collects and stores performance metrics."""
    
    def __init__(self, db_path: str = "/tmp/metrics.db", retention_days: int = 30):
        self.db_path = db_path
        self.retention_days = retention_days
        self.metrics_buffer = deque(maxlen=10000)
        self._init_database()
        self._start_cleanup_task()
    
    def _init_database(self):
        """Initialize metrics database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    value REAL NOT NULL,
                    metric_type TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    tags TEXT,
                    metadata TEXT
                )
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_metrics_name ON metrics(name)
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON metrics(timestamp)
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS alerts (
                    alert_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    level TEXT NOT NULL,
                    message TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    threshold REAL NOT NULL,
                    current_value REAL NOT NULL,
                    timestamp DATETIME NOT NULL,
                    resolved BOOLEAN DEFAULT FALSE,
                    resolved_at DATETIME
                )
            ''')
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to initialize metrics database: {e}")
            raise
    
    def record_metric(self, 
                     name: str, 
                     value: float, 
                     metric_type: MetricType = MetricType.GAUGE,
                     tags: Dict[str, str] = None,
                     metadata: Dict[str, Any] = None):
        """Record a performance metric."""
        try:
            metric = Metric(
                name=name,
                value=value,
                metric_type=metric_type,
                timestamp=datetime.now(),
                tags=tags or {},
                metadata=metadata or {}
            )
            
            self.metrics_buffer.append(metric)
            
            # Flush buffer if it's full
            if len(self.metrics_buffer) >= 1000:
                self._flush_buffer()
                
        except Exception as e:
            logger.error(f"Failed to record metric {name}: {e}")
    
    def _flush_buffer(self):
        """Flush metrics buffer to database."""
        try:
            if not self.metrics_buffer:
                return
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            metrics_data = []
            for metric in self.metrics_buffer:
                metrics_data.append((
                    metric.name,
                    metric.value,
                    metric.metric_type.value,
                    metric.timestamp.isoformat(),
                    json.dumps(metric.tags),
                    json.dumps(metric.metadata)
                ))
            
            cursor.executemany('''
                INSERT INTO metrics (name, value, metric_type, timestamp, tags, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', metrics_data)
            
            conn.commit()
            conn.close()
            
            self.metrics_buffer.clear()
            
        except Exception as e:
            logger.error(f"Failed to flush metrics buffer: {e}")
    
    def get_metrics(self, 
                   name: Optional[str] = None,
                   start_time: Optional[datetime] = None,
                   end_time: Optional[datetime] = None,
                   limit: int = 1000) -> List[Metric]:
        """Get metrics with optional filtering."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            query = "SELECT * FROM metrics WHERE 1=1"
            params = []
            
            if name:
                query += " AND name = ?"
                params.append(name)
            
            if start_time:
                query += " AND timestamp >= ?"
                params.append(start_time.isoformat())
            
            if end_time:
                query += " AND timestamp <= ?"
                params.append(end_time.isoformat())
            
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            conn.close()
            
            metrics = []
            for row in rows:
                metric = Metric(
                    name=row[1],
                    value=row[2],
                    metric_type=MetricType(row[3]),
                    timestamp=datetime.fromisoformat(row[4]),
                    tags=json.loads(row[5]) if row[5] else {},
                    metadata=json.loads(row[6]) if row[6] else {}
                )
                metrics.append(metric)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to get metrics: {e}")
            return []
    
    def _start_cleanup_task(self):
        """Start background cleanup task."""
        def cleanup_old_metrics():
            while True:
                try:
                    time.sleep(3600)  # Run every hour
                    
                    cutoff_time = datetime.now() - timedelta(days=self.retention_days)
                    
                    conn = sqlite3.connect(self.db_path)
                    cursor = conn.cursor()
                    
                    cursor.execute('DELETE FROM metrics WHERE timestamp < ?', (cutoff_time.isoformat(),))
                    deleted_count = cursor.rowcount
                    
                    conn.commit()
                    conn.close()
                    
                    if deleted_count > 0:
                        logger.info(f"Cleaned up {deleted_count} old metrics")
                        
                except Exception as e:
                    logger.error(f"Metrics cleanup failed: {e}")
        
        cleanup_thread = threading.Thread(target=cleanup_old_metrics, daemon=True)
        cleanup_thread.start()

class AlertManager:
    """Manages performance alerts and thresholds."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.alert_rules = {}
        self.active_alerts = {}
        self.alert_callbacks = []
    
    def add_alert_rule(self, 
                      name: str,
                      metric_name: str,
                      threshold: float,
                      level: AlertLevel,
                      condition: str = "greater_than",
                      message_template: str = None):
        """Add an alert rule for a metric."""
        try:
            self.alert_rules[name] = {
                "metric_name": metric_name,
                "threshold": threshold,
                "level": level,
                "condition": condition,
                "message_template": message_template or f"{metric_name} is {condition} {threshold}",
                "enabled": True
            }
            
            logger.info(f"Added alert rule: {name}")
            
        except Exception as e:
            logger.error(f"Failed to add alert rule {name}: {e}")
    
    def add_alert_callback(self, callback: Callable[[Alert], None]):
        """Add callback for alert notifications."""
        self.alert_callbacks.append(callback)
    
    def check_alerts(self):
        """Check all alert rules and trigger alerts if needed."""
        try:
            for rule_name, rule in self.alert_rules.items():
                if not rule["enabled"]:
                    continue
                
                # Get latest metric value
                metrics = self.metrics_collector.get_metrics(
                    name=rule["metric_name"],
                    limit=1
                )
                
                if not metrics:
                    continue
                
                latest_metric = metrics[0]
                current_value = latest_metric.value
                threshold = rule["threshold"]
                condition = rule["condition"]
                
                # Check if alert should be triggered
                should_alert = False
                if condition == "greater_than" and current_value > threshold:
                    should_alert = True
                elif condition == "less_than" and current_value < threshold:
                    should_alert = True
                elif condition == "equals" and current_value == threshold:
                    should_alert = True
                
                if should_alert:
                    # Check if alert is already active
                    if rule_name not in self.active_alerts:
                        self._trigger_alert(rule_name, rule, current_value)
                else:
                    # Resolve alert if it was active
                    if rule_name in self.active_alerts:
                        self._resolve_alert(rule_name)
                        
        except Exception as e:
            logger.error(f"Alert checking failed: {e}")
    
    def _trigger_alert(self, rule_name: str, rule: Dict[str, Any], current_value: float):
        """Trigger an alert."""
        try:
            alert_id = f"{rule_name}_{int(time.time())}"
            
            alert = Alert(
                alert_id=alert_id,
                name=rule_name,
                level=rule["level"],
                message=rule["message_template"].format(
                    metric_name=rule["metric_name"],
                    threshold=rule["threshold"],
                    current_value=current_value
                ),
                metric_name=rule["metric_name"],
                threshold=rule["threshold"],
                current_value=current_value,
                timestamp=datetime.now()
            )
            
            self.active_alerts[rule_name] = alert
            
            # Store alert in database
            self._store_alert(alert)
            
            # Call alert callbacks
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    logger.error(f"Alert callback failed: {e}")
            
            logger.warning(f"Alert triggered: {rule_name} - {alert.message}")
            
        except Exception as e:
            logger.error(f"Failed to trigger alert {rule_name}: {e}")
    
    def _resolve_alert(self, rule_name: str):
        """Resolve an active alert."""
        try:
            if rule_name in self.active_alerts:
                alert = self.active_alerts[rule_name]
                alert.resolved = True
                alert.resolved_at = datetime.now()
                
                # Update alert in database
                self._update_alert(alert)
                
                del self.active_alerts[rule_name]
                
                logger.info(f"Alert resolved: {rule_name}")
                
        except Exception as e:
            logger.error(f"Failed to resolve alert {rule_name}: {e}")
    
    def _store_alert(self, alert: Alert):
        """Store alert in database."""
        try:
            conn = sqlite3.connect(self.metrics_collector.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO alerts 
                (alert_id, name, level, message, metric_name, threshold, current_value, timestamp, resolved, resolved_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                alert.alert_id,
                alert.name,
                alert.level.value,
                alert.message,
                alert.metric_name,
                alert.threshold,
                alert.current_value,
                alert.timestamp.isoformat(),
                alert.resolved,
                alert.resolved_at.isoformat() if alert.resolved_at else None
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to store alert: {e}")
    
    def _update_alert(self, alert: Alert):
        """Update alert in database."""
        try:
            conn = sqlite3.connect(self.metrics_collector.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE alerts 
                SET resolved = ?, resolved_at = ?
                WHERE alert_id = ?
            ''', (alert.resolved, alert.resolved_at.isoformat() if alert.resolved_at else None, alert.alert_id))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to update alert: {e}")

class SystemMonitor:
    """Monitors system resources and performance."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.monitoring_active = False
        self.monitoring_thread = None
    
    def start_monitoring(self, interval: float = 5.0):
        """Start system monitoring."""
        try:
            if self.monitoring_active:
                return
            
            self.monitoring_active = True
            self.monitoring_thread = threading.Thread(
                target=self._monitoring_loop,
                args=(interval,),
                daemon=True
            )
            self.monitoring_thread.start()
            
            logger.info("System monitoring started")
            
        except Exception as e:
            logger.error(f"Failed to start monitoring: {e}")
    
    def stop_monitoring(self):
        """Stop system monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        
        logger.info("System monitoring stopped")
    
    def _monitoring_loop(self, interval: float):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                snapshot = self._collect_system_snapshot()
                self._record_snapshot_metrics(snapshot)
                
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                time.sleep(interval)
    
    def _collect_system_snapshot(self) -> PerformanceSnapshot:
        """Collect current system performance snapshot."""
        try:
            # CPU usage
            cpu_usage = psutil.cpu_percent(interval=1.0)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_usage = memory.percent
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_usage = (disk.used / disk.total) * 100
            
            # Network I/O
            network = psutil.net_io_counters()
            network_io = {
                "bytes_sent": network.bytes_sent,
                "bytes_recv": network.bytes_recv,
                "packets_sent": network.packets_sent,
                "packets_recv": network.packets_recv
            }
            
            # Process count
            process_count = len(psutil.pids())
            
            # Active connections (placeholder)
            active_connections = 0
            
            # Queue sizes (placeholder - would get from actual queues)
            queue_sizes = {
                "job_queue": 0,
                "processing_queue": 0
            }
            
            return PerformanceSnapshot(
                timestamp=datetime.now(),
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                disk_usage=disk_usage,
                network_io=network_io,
                process_count=process_count,
                active_connections=active_connections,
                queue_sizes=queue_sizes
            )
            
        except Exception as e:
            logger.error(f"Failed to collect system snapshot: {e}")
            return PerformanceSnapshot(
                timestamp=datetime.now(),
                cpu_usage=0.0,
                memory_usage=0.0,
                disk_usage=0.0,
                network_io={},
                process_count=0,
                active_connections=0,
                queue_sizes={}
            )
    
    def _record_snapshot_metrics(self, snapshot: PerformanceSnapshot):
        """Record metrics from system snapshot."""
        try:
            # Record system metrics
            self.metrics_collector.record_metric(
                "system.cpu_usage",
                snapshot.cpu_usage,
                MetricType.GAUGE,
                tags={"component": "system"}
            )
            
            self.metrics_collector.record_metric(
                "system.memory_usage",
                snapshot.memory_usage,
                MetricType.GAUGE,
                tags={"component": "system"}
            )
            
            self.metrics_collector.record_metric(
                "system.disk_usage",
                snapshot.disk_usage,
                MetricType.GAUGE,
                tags={"component": "system"}
            )
            
            self.metrics_collector.record_metric(
                "system.process_count",
                snapshot.process_count,
                MetricType.GAUGE,
                tags={"component": "system"}
            )
            
            # Record network metrics
            for metric_name, value in snapshot.network_io.items():
                self.metrics_collector.record_metric(
                    f"system.network.{metric_name}",
                    value,
                    MetricType.COUNTER,
                    tags={"component": "system"}
                )
            
            # Record queue metrics
            for queue_name, size in snapshot.queue_sizes.items():
                self.metrics_collector.record_metric(
                    f"system.queue.{queue_name}.size",
                    size,
                    MetricType.GAUGE,
                    tags={"component": "queue", "queue": queue_name}
                )
            
        except Exception as e:
            logger.error(f"Failed to record snapshot metrics: {e}")

class PerformanceMonitor:
    """Main performance monitoring system."""
    
    def __init__(self, db_path: str = "/tmp/metrics.db"):
        self.metrics_collector = MetricsCollector(db_path)
        self.alert_manager = AlertManager(self.metrics_collector)
        self.system_monitor = SystemMonitor(self.metrics_collector)
        
        self.logger = structlog.get_logger("performance_monitor")
        
        # Setup default alert rules
        self._setup_default_alerts()
    
    def _setup_default_alerts(self):
        """Setup default alert rules."""
        try:
            # CPU usage alerts
            self.alert_manager.add_alert_rule(
                "high_cpu_usage",
                "system.cpu_usage",
                80.0,
                AlertLevel.WARNING,
                "greater_than",
                "CPU usage is {current_value:.1f}% (threshold: {threshold:.1f}%)"
            )
            
            self.alert_manager.add_alert_rule(
                "critical_cpu_usage",
                "system.cpu_usage",
                95.0,
                AlertLevel.CRITICAL,
                "greater_than",
                "CPU usage is critically high: {current_value:.1f}% (threshold: {threshold:.1f}%)"
            )
            
            # Memory usage alerts
            self.alert_manager.add_alert_rule(
                "high_memory_usage",
                "system.memory_usage",
                85.0,
                AlertLevel.WARNING,
                "greater_than",
                "Memory usage is {current_value:.1f}% (threshold: {threshold:.1f}%)"
            )
            
            self.alert_manager.add_alert_rule(
                "critical_memory_usage",
                "system.memory_usage",
                95.0,
                AlertLevel.CRITICAL,
                "greater_than",
                "Memory usage is critically high: {current_value:.1f}% (threshold: {threshold:.1f}%)"
            )
            
            # Disk usage alerts
            self.alert_manager.add_alert_rule(
                "high_disk_usage",
                "system.disk_usage",
                90.0,
                AlertLevel.WARNING,
                "greater_than",
                "Disk usage is {current_value:.1f}% (threshold: {threshold:.1f}%)"
            )
            
            self.logger.info("Default alert rules configured")
            
        except Exception as e:
            self.logger.error(f"Failed to setup default alerts: {e}")
    
    def start(self, monitoring_interval: float = 5.0):
        """Start performance monitoring."""
        try:
            # Start system monitoring
            self.system_monitor.start_monitoring(monitoring_interval)
            
            # Start alert checking
            self._start_alert_checking()
            
            self.logger.info("Performance monitoring started")
            
        except Exception as e:
            self.logger.error(f"Failed to start performance monitoring: {e}")
            raise
    
    def stop(self):
        """Stop performance monitoring."""
        try:
            # Stop system monitoring
            self.system_monitor.stop_monitoring()
            
            # Flush any remaining metrics
            self.metrics_collector._flush_buffer()
            
            self.logger.info("Performance monitoring stopped")
            
        except Exception as e:
            self.logger.error(f"Failed to stop performance monitoring: {e}")
    
    def _start_alert_checking(self):
        """Start alert checking in background."""
        def alert_checking_loop():
            while True:
                try:
                    self.alert_manager.check_alerts()
                    time.sleep(10)  # Check alerts every 10 seconds
                except Exception as e:
                    self.logger.error(f"Alert checking error: {e}")
                    time.sleep(10)
        
        alert_thread = threading.Thread(target=alert_checking_loop, daemon=True)
        alert_thread.start()
    
    def record_custom_metric(self, 
                           name: str, 
                           value: float, 
                           metric_type: MetricType = MetricType.GAUGE,
                           tags: Dict[str, str] = None,
                           metadata: Dict[str, Any] = None):
        """Record a custom metric."""
        self.metrics_collector.record_metric(name, value, metric_type, tags, metadata)
    
    def get_metrics_summary(self, 
                          start_time: Optional[datetime] = None,
                          end_time: Optional[datetime] = None) -> Dict[str, Any]:
        """Get metrics summary for a time period."""
        try:
            if not start_time:
                start_time = datetime.now() - timedelta(hours=1)
            if not end_time:
                end_time = datetime.now()
            
            # Get all metrics for the time period
            metrics = self.metrics_collector.get_metrics(
                start_time=start_time,
                end_time=end_time,
                limit=10000
            )
            
            # Group metrics by name
            metrics_by_name = defaultdict(list)
            for metric in metrics:
                metrics_by_name[metric.name].append(metric.value)
            
            # Calculate summary statistics
            summary = {}
            for name, values in metrics_by_name.items():
                if values:
                    summary[name] = {
                        "count": len(values),
                        "min": min(values),
                        "max": max(values),
                        "mean": np.mean(values),
                        "median": np.median(values),
                        "std": np.std(values),
                        "latest": values[0] if values else 0
                    }
            
            return {
                "time_range": {
                    "start": start_time.isoformat(),
                    "end": end_time.isoformat()
                },
                "metrics": summary,
                "total_metrics": len(metrics)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get metrics summary: {e}")
            return {}
    
    def get_active_alerts(self) -> List[Alert]:
        """Get currently active alerts."""
        return list(self.alert_manager.active_alerts.values())
    
    def add_alert_callback(self, callback: Callable[[Alert], None]):
        """Add callback for alert notifications."""
        self.alert_manager.add_alert_callback(callback)

# Export classes
__all__ = [
    "PerformanceMonitor",
    "MetricsCollector",
    "AlertManager",
    "SystemMonitor",
    "Metric",
    "Alert",
    "PerformanceSnapshot",
    "MetricType",
    "AlertLevel"
]


