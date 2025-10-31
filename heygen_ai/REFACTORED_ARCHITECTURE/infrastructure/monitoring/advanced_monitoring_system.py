"""
Advanced Monitoring and Observability System

This module provides comprehensive monitoring and observability capabilities
for the refactored HeyGen AI architecture with real-time metrics, tracing,
and intelligent alerting.
"""

import asyncio
import time
import json
import logging
import threading
import psutil
import gc
from typing import Any, Dict, List, Optional, Callable, Union
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import uuid
import hashlib
import statistics
from pathlib import Path
import sqlite3
import requests
import websockets
from prometheus_client import Counter, Histogram, Gauge, Summary, start_http_server
import redis
from contextvars import ContextVar
import traceback
import sys
import os


# Context variables for tracing
trace_id_var: ContextVar[Optional[str]] = ContextVar('trace_id', default=None)
span_id_var: ContextVar[Optional[str]] = ContextVar('span_id', default=None)
parent_span_id_var: ContextVar[Optional[str]] = ContextVar('parent_span_id', default=None)


class MetricType(str, Enum):
    """Metric types."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class AlertLevel(str, Enum):
    """Alert levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class HealthStatus(str, Enum):
    """Health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class Metric:
    """Metric data structure."""
    name: str
    value: float
    metric_type: MetricType
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    description: Optional[str] = None


@dataclass
class Span:
    """Distributed tracing span."""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    operation_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration: Optional[float] = None
    tags: Dict[str, str] = field(default_factory=dict)
    logs: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "started"


@dataclass
class Alert:
    """Alert data structure."""
    alert_id: str
    name: str
    level: AlertLevel
    message: str
    timestamp: datetime
    source: str
    labels: Dict[str, str] = field(default_factory=dict)
    resolved: bool = False
    resolved_at: Optional[datetime] = None


@dataclass
class HealthCheck:
    """Health check result."""
    name: str
    status: HealthStatus
    message: str
    timestamp: datetime
    duration: float
    details: Dict[str, Any] = field(default_factory=dict)


class MetricsCollector:
    """Advanced metrics collector with multiple backends."""
    
    def __init__(self):
        self.metrics: Dict[str, Any] = {}
        self.counters: Dict[str, Counter] = {}
        self.gauges: Dict[str, Gauge] = {}
        self.histograms: Dict[str, Histogram] = {}
        self.summaries: Dict[str, Summary] = {}
        self.custom_metrics: Dict[str, Metric] = {}
        self.lock = threading.RLock()
    
    def create_counter(self, name: str, description: str = "", labels: List[str] = None):
        """Create a counter metric."""
        with self.lock:
            if name not in self.counters:
                self.counters[name] = Counter(name, description, labels or [])
    
    def create_gauge(self, name: str, description: str = "", labels: List[str] = None):
        """Create a gauge metric."""
        with self.lock:
            if name not in self.gauges:
                self.gauges[name] = Gauge(name, description, labels or [])
    
    def create_histogram(self, name: str, description: str = "", labels: List[str] = None, buckets: List[float] = None):
        """Create a histogram metric."""
        with self.lock:
            if name not in self.histograms:
                self.histograms[name] = Histogram(name, description, labels or [], buckets=buckets)
    
    def create_summary(self, name: str, description: str = "", labels: List[str] = None):
        """Create a summary metric."""
        with self.lock:
            if name not in self.summaries:
                self.summaries[name] = Summary(name, description, labels or [])
    
    def increment_counter(self, name: str, value: float = 1.0, labels: Dict[str, str] = None):
        """Increment a counter metric."""
        with self.lock:
            if name in self.counters:
                if labels:
                    self.counters[name].labels(**labels).inc(value)
                else:
                    self.counters[name].inc(value)
    
    def set_gauge(self, name: str, value: float, labels: Dict[str, str] = None):
        """Set a gauge metric value."""
        with self.lock:
            if name in self.gauges:
                if labels:
                    self.gauges[name].labels(**labels).set(value)
                else:
                    self.gauges[name].set(value)
    
    def observe_histogram(self, name: str, value: float, labels: Dict[str, str] = None):
        """Observe a histogram metric."""
        with self.lock:
            if name in self.histograms:
                if labels:
                    self.histograms[name].labels(**labels).observe(value)
                else:
                    self.histograms[name].observe(value)
    
    def observe_summary(self, name: str, value: float, labels: Dict[str, str] = None):
        """Observe a summary metric."""
        with self.lock:
            if name in self.summaries:
                if labels:
                    self.summaries[name].labels(**labels).observe(value)
                else:
                    self.summaries[name].observe(value)
    
    def add_custom_metric(self, metric: Metric):
        """Add a custom metric."""
        with self.lock:
            self.custom_metrics[metric.name] = metric
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get all metrics."""
        with self.lock:
            return {
                "counters": {name: metric._value.get() for name, metric in self.counters.items()},
                "gauges": {name: metric._value.get() for name, metric in self.gauges.items()},
                "histograms": {name: metric._sum._value.get() for name, metric in self.histograms.items()},
                "summaries": {name: metric._sum._value.get() for name, metric in self.summaries.items()},
                "custom": {name: metric.value for name, metric in self.custom_metrics.items()}
            }


class DistributedTracer:
    """Distributed tracing system."""
    
    def __init__(self):
        self.spans: Dict[str, Span] = {}
        self.traces: Dict[str, List[Span]] = defaultdict(list)
        self.lock = threading.RLock()
    
    def start_span(self, operation_name: str, parent_span_id: Optional[str] = None, tags: Dict[str, str] = None) -> str:
        """Start a new span."""
        trace_id = trace_id_var.get() or str(uuid.uuid4())
        span_id = str(uuid.uuid4())
        
        span = Span(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            operation_name=operation_name,
            start_time=datetime.now(timezone.utc),
            tags=tags or {}
        )
        
        with self.lock:
            self.spans[span_id] = span
            self.traces[trace_id].append(span)
        
        # Set context
        trace_id_var.set(trace_id)
        span_id_var.set(span_id)
        parent_span_id_var.set(parent_span_id)
        
        return span_id
    
    def finish_span(self, span_id: str, status: str = "success", tags: Dict[str, str] = None):
        """Finish a span."""
        with self.lock:
            if span_id in self.spans:
                span = self.spans[span_id]
                span.end_time = datetime.now(timezone.utc)
                span.duration = (span.end_time - span.start_time).total_seconds()
                span.status = status
                
                if tags:
                    span.tags.update(tags)
    
    def add_span_log(self, span_id: str, message: str, level: str = "info", fields: Dict[str, Any] = None):
        """Add a log to a span."""
        with self.lock:
            if span_id in self.spans:
                log_entry = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "level": level,
                    "message": message,
                    "fields": fields or {}
                }
                self.spans[span_id].logs.append(log_entry)
    
    def get_trace(self, trace_id: str) -> List[Span]:
        """Get all spans for a trace."""
        with self.lock:
            return self.traces.get(trace_id, [])
    
    def get_span(self, span_id: str) -> Optional[Span]:
        """Get a specific span."""
        with self.lock:
            return self.spans.get(span_id)


class AlertManager:
    """Advanced alert management system."""
    
    def __init__(self):
        self.alerts: Dict[str, Alert] = {}
        self.rules: Dict[str, Dict[str, Any]] = {}
        self.notification_channels: List[Callable] = []
        self.lock = threading.RLock()
    
    def add_rule(self, name: str, condition: Callable, level: AlertLevel, message: str):
        """Add an alert rule."""
        with self.lock:
            self.rules[name] = {
                "condition": condition,
                "level": level,
                "message": message
            }
    
    def add_notification_channel(self, channel: Callable):
        """Add a notification channel."""
        with self.lock:
            self.notification_channels.append(channel)
    
    def check_alerts(self, metrics: Dict[str, Any]):
        """Check all alert rules."""
        with self.lock:
            for rule_name, rule in self.rules.items():
                try:
                    if rule["condition"](metrics):
                        self._create_alert(rule_name, rule, metrics)
                except Exception as e:
                    logging.error(f"Error checking alert rule {rule_name}: {e}")
    
    def _create_alert(self, rule_name: str, rule: Dict[str, Any], metrics: Dict[str, Any]):
        """Create an alert."""
        alert_id = str(uuid.uuid4())
        alert = Alert(
            alert_id=alert_id,
            name=rule_name,
            level=rule["level"],
            message=rule["message"],
            timestamp=datetime.now(timezone.utc),
            source="monitoring_system",
            labels={"rule": rule_name}
        )
        
        self.alerts[alert_id] = alert
        
        # Send notifications
        for channel in self.notification_channels:
            try:
                channel(alert)
            except Exception as e:
                logging.error(f"Error sending notification: {e}")
    
    def resolve_alert(self, alert_id: str):
        """Resolve an alert."""
        with self.lock:
            if alert_id in self.alerts:
                self.alerts[alert_id].resolved = True
                self.alerts[alert_id].resolved_at = datetime.now(timezone.utc)
    
    def get_active_alerts(self) -> List[Alert]:
        """Get active alerts."""
        with self.lock:
            return [alert for alert in self.alerts.values() if not alert.resolved]


class HealthChecker:
    """System health checking."""
    
    def __init__(self):
        self.checks: Dict[str, Callable] = {}
        self.results: Dict[str, HealthCheck] = {}
        self.lock = threading.RLock()
    
    def add_check(self, name: str, check_func: Callable):
        """Add a health check."""
        with self.lock:
            self.checks[name] = check_func
    
    async def run_checks(self) -> Dict[str, HealthCheck]:
        """Run all health checks."""
        results = {}
        
        with self.lock:
            for name, check_func in self.checks.items():
                start_time = time.time()
                try:
                    result = await check_func() if asyncio.iscoroutinefunction(check_func) else check_func()
                    duration = time.time() - start_time
                    
                    health_check = HealthCheck(
                        name=name,
                        status=HealthStatus.HEALTHY if result else HealthStatus.UNHEALTHY,
                        message="Check passed" if result else "Check failed",
                        timestamp=datetime.now(timezone.utc),
                        duration=duration
                    )
                    
                except Exception as e:
                    duration = time.time() - start_time
                    health_check = HealthCheck(
                        name=name,
                        status=HealthStatus.UNHEALTHY,
                        message=f"Check error: {str(e)}",
                        timestamp=datetime.now(timezone.utc),
                        duration=duration,
                        details={"error": str(e), "traceback": traceback.format_exc()}
                    )
                
                results[name] = health_check
        
        with self.lock:
            self.results.update(results)
        
        return results
    
    def get_overall_health(self) -> HealthStatus:
        """Get overall system health."""
        with self.lock:
            if not self.results:
                return HealthStatus.UNKNOWN
            
            statuses = [check.status for check in self.results.values()]
            
            if HealthStatus.UNHEALTHY in statuses:
                return HealthStatus.UNHEALTHY
            elif HealthStatus.DEGRADED in statuses:
                return HealthStatus.DEGRADED
            else:
                return HealthStatus.HEALTHY


class PerformanceProfiler:
    """Advanced performance profiling."""
    
    def __init__(self):
        self.profiles: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.RLock()
    
    def start_profile(self, name: str) -> str:
        """Start a performance profile."""
        profile_id = str(uuid.uuid4())
        
        with self.lock:
            self.profiles[profile_id] = {
                "name": name,
                "start_time": time.time(),
                "start_memory": psutil.virtual_memory().used,
                "start_cpu": psutil.cpu_percent(),
                "start_gc": sum(stat['collections'] for stat in gc.get_stats()),
                "threads": threading.active_count()
            }
        
        return profile_id
    
    def end_profile(self, profile_id: str) -> Dict[str, Any]:
        """End a performance profile."""
        with self.lock:
            if profile_id not in self.profiles:
                return {}
            
            profile = self.profiles[profile_id]
            end_time = time.time()
            
            profile.update({
                "end_time": end_time,
                "duration": end_time - profile["start_time"],
                "end_memory": psutil.virtual_memory().used,
                "memory_delta": psutil.virtual_memory().used - profile["start_memory"],
                "end_cpu": psutil.cpu_percent(),
                "cpu_delta": psutil.cpu_percent() - profile["start_cpu"],
                "end_gc": sum(stat['collections'] for stat in gc.get_stats()),
                "gc_delta": sum(stat['collections'] for stat in gc.get_stats()) - profile["start_gc"],
                "end_threads": threading.active_count(),
                "thread_delta": threading.active_count() - profile["threads"]
            })
            
            return profile


class AdvancedMonitoringSystem:
    """
    Advanced monitoring and observability system.
    
    Features:
    - Real-time metrics collection
    - Distributed tracing
    - Health checking
    - Alert management
    - Performance profiling
    - Multiple backends (Prometheus, Redis, SQLite)
    - WebSocket real-time updates
    - Intelligent alerting
    """
    
    def __init__(
        self,
        prometheus_port: int = 9090,
        redis_url: str = "redis://localhost:6379/0",
        db_path: str = "monitoring.db",
        enable_websocket: bool = True,
        websocket_port: int = 9091
    ):
        """
        Initialize the advanced monitoring system.
        
        Args:
            prometheus_port: Port for Prometheus metrics server
            redis_url: Redis URL for caching
            db_path: SQLite database path for persistence
            enable_websocket: Enable WebSocket real-time updates
            websocket_port: WebSocket server port
        """
        self.prometheus_port = prometheus_port
        self.redis_url = redis_url
        self.db_path = db_path
        self.enable_websocket = enable_websocket
        self.websocket_port = websocket_port
        
        # Initialize components
        self.metrics_collector = MetricsCollector()
        self.tracer = DistributedTracer()
        self.alert_manager = AlertManager()
        self.health_checker = HealthChecker()
        self.profiler = PerformanceProfiler()
        
        # Initialize backends
        self.redis_client = None
        self.db_connection = None
        self.websocket_server = None
        
        # Setup metrics
        self._setup_metrics()
        
        # Setup health checks
        self._setup_health_checks()
        
        # Setup alert rules
        self._setup_alert_rules()
        
        # Start services
        self._start_services()
    
    def _setup_metrics(self):
        """Setup default metrics."""
        # System metrics
        self.metrics_collector.create_gauge("system_cpu_percent", "CPU usage percentage")
        self.metrics_collector.create_gauge("system_memory_percent", "Memory usage percentage")
        self.metrics_collector.create_gauge("system_memory_used_bytes", "Memory used in bytes")
        self.metrics_collector.create_gauge("system_disk_percent", "Disk usage percentage")
        self.metrics_collector.create_gauge("system_thread_count", "Number of active threads")
        
        # Application metrics
        self.metrics_collector.create_counter("requests_total", "Total number of requests", ["method", "endpoint", "status"])
        self.metrics_collector.create_histogram("request_duration_seconds", "Request duration in seconds", ["method", "endpoint"])
        self.metrics_collector.create_gauge("active_connections", "Number of active connections")
        self.metrics_collector.create_counter("errors_total", "Total number of errors", ["type", "source"])
        
        # AI Model metrics
        self.metrics_collector.create_counter("model_predictions_total", "Total number of model predictions", ["model_name", "model_type"])
        self.metrics_collector.create_histogram("model_prediction_duration_seconds", "Model prediction duration", ["model_name", "model_type"])
        self.metrics_collector.create_gauge("model_memory_usage_bytes", "Model memory usage in bytes", ["model_name"])
        self.metrics_collector.create_counter("model_training_epochs_total", "Total training epochs", ["model_name"])
    
    def _setup_health_checks(self):
        """Setup health checks."""
        self.health_checker.add_check("database", self._check_database)
        self.health_checker.add_check("redis", self._check_redis)
        self.health_checker.add_check("memory", self._check_memory)
        self.health_checker.add_check("disk", self._check_disk)
        self.health_checker.add_check("cpu", self._check_cpu)
    
    def _setup_alert_rules(self):
        """Setup alert rules."""
        # High CPU usage
        self.alert_manager.add_rule(
            "high_cpu_usage",
            lambda metrics: metrics.get("system_cpu_percent", 0) > 80,
            AlertLevel.WARNING,
            "High CPU usage detected"
        )
        
        # High memory usage
        self.alert_manager.add_rule(
            "high_memory_usage",
            lambda metrics: metrics.get("system_memory_percent", 0) > 85,
            AlertLevel.WARNING,
            "High memory usage detected"
        )
        
        # High error rate
        self.alert_manager.add_rule(
            "high_error_rate",
            lambda metrics: metrics.get("errors_total", 0) > 100,
            AlertLevel.ERROR,
            "High error rate detected"
        )
        
        # Low disk space
        self.alert_manager.add_rule(
            "low_disk_space",
            lambda metrics: metrics.get("system_disk_percent", 0) > 90,
            AlertLevel.CRITICAL,
            "Low disk space detected"
        )
    
    def _start_services(self):
        """Start monitoring services."""
        # Start Prometheus server
        start_http_server(self.prometheus_port)
        
        # Initialize Redis
        try:
            self.redis_client = redis.from_url(self.redis_url)
            self.redis_client.ping()
        except redis.ConnectionError:
            logging.warning("Redis not available, using in-memory storage")
        
        # Initialize SQLite database
        self._init_database()
        
        # Start WebSocket server
        if self.enable_websocket:
            asyncio.create_task(self._start_websocket_server())
        
        # Start background tasks
        asyncio.create_task(self._collect_system_metrics())
        asyncio.create_task(self._check_alerts_periodically())
        asyncio.create_task(self._run_health_checks_periodically())
    
    def _init_database(self):
        """Initialize SQLite database."""
        self.db_connection = sqlite3.connect(self.db_path)
        cursor = self.db_connection.cursor()
        
        # Create tables
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                value REAL NOT NULL,
                labels TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                alert_id TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                level TEXT NOT NULL,
                message TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                resolved BOOLEAN DEFAULT FALSE
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS traces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trace_id TEXT NOT NULL,
                span_id TEXT NOT NULL,
                operation_name TEXT NOT NULL,
                start_time DATETIME NOT NULL,
                end_time DATETIME,
                duration REAL,
                tags TEXT,
                logs TEXT
            )
        """)
        
        self.db_connection.commit()
    
    async def _collect_system_metrics(self):
        """Collect system metrics periodically."""
        while True:
            try:
                # Collect system metrics
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                
                # Update metrics
                self.metrics_collector.set_gauge("system_cpu_percent", cpu_percent)
                self.metrics_collector.set_gauge("system_memory_percent", memory.percent)
                self.metrics_collector.set_gauge("system_memory_used_bytes", memory.used)
                self.metrics_collector.set_gauge("system_disk_percent", disk.percent)
                self.metrics_collector.set_gauge("system_thread_count", threading.active_count())
                
                # Store in database
                await self._store_metrics()
                
            except Exception as e:
                logging.error(f"Error collecting system metrics: {e}")
            
            await asyncio.sleep(10)  # Collect every 10 seconds
    
    async def _check_alerts_periodically(self):
        """Check alerts periodically."""
        while True:
            try:
                metrics = self.metrics_collector.get_metrics()
                self.alert_manager.check_alerts(metrics)
            except Exception as e:
                logging.error(f"Error checking alerts: {e}")
            
            await asyncio.sleep(30)  # Check every 30 seconds
    
    async def _run_health_checks_periodically(self):
        """Run health checks periodically."""
        while True:
            try:
                await self.health_checker.run_checks()
            except Exception as e:
                logging.error(f"Error running health checks: {e}")
            
            await asyncio.sleep(60)  # Check every minute
    
    async def _store_metrics(self):
        """Store metrics in database."""
        if not self.db_connection:
            return
        
        try:
            cursor = self.db_connection.cursor()
            metrics = self.metrics_collector.get_metrics()
            
            for metric_type, metric_data in metrics.items():
                for name, value in metric_data.items():
                    cursor.execute("""
                        INSERT INTO metrics (name, value, labels)
                        VALUES (?, ?, ?)
                    """, (name, value, json.dumps({})))
            
            self.db_connection.commit()
        except Exception as e:
            logging.error(f"Error storing metrics: {e}")
    
    async def _start_websocket_server(self):
        """Start WebSocket server for real-time updates."""
        async def handle_client(websocket, path):
            try:
                while True:
                    # Send metrics update
                    metrics = self.metrics_collector.get_metrics()
                    health = self.health_checker.get_overall_health()
                    alerts = self.alert_manager.get_active_alerts()
                    
                    update = {
                        "type": "metrics_update",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "metrics": metrics,
                        "health": health.value,
                        "alerts": [alert.__dict__ for alert in alerts]
                    }
                    
                    await websocket.send(json.dumps(update))
                    await asyncio.sleep(5)  # Send update every 5 seconds
                    
            except websockets.exceptions.ConnectionClosed:
                pass
        
        start_server = websockets.serve(handle_client, "localhost", self.websocket_port)
        await start_server
    
    def _check_database(self) -> bool:
        """Check database health."""
        try:
            if self.db_connection:
                cursor = self.db_connection.cursor()
                cursor.execute("SELECT 1")
                return True
        except Exception:
            pass
        return False
    
    def _check_redis(self) -> bool:
        """Check Redis health."""
        try:
            if self.redis_client:
                self.redis_client.ping()
                return True
        except Exception:
            pass
        return False
    
    def _check_memory(self) -> bool:
        """Check memory health."""
        memory = psutil.virtual_memory()
        return memory.percent < 90
    
    def _check_disk(self) -> bool:
        """Check disk health."""
        disk = psutil.disk_usage('/')
        return disk.percent < 90
    
    def _check_cpu(self) -> bool:
        """Check CPU health."""
        cpu_percent = psutil.cpu_percent(interval=1)
        return cpu_percent < 90
    
    def start_trace(self, operation_name: str, tags: Dict[str, str] = None) -> str:
        """Start a distributed trace."""
        return self.tracer.start_span(operation_name, tags=tags)
    
    def finish_trace(self, span_id: str, status: str = "success", tags: Dict[str, str] = None):
        """Finish a distributed trace."""
        self.tracer.finish_span(span_id, status, tags)
    
    def add_trace_log(self, span_id: str, message: str, level: str = "info", fields: Dict[str, Any] = None):
        """Add a log to a trace."""
        self.tracer.add_span_log(span_id, message, level, fields)
    
    def record_metric(self, name: str, value: float, metric_type: MetricType = MetricType.GAUGE, labels: Dict[str, str] = None):
        """Record a custom metric."""
        if metric_type == MetricType.COUNTER:
            self.metrics_collector.increment_counter(name, value, labels)
        elif metric_type == MetricType.GAUGE:
            self.metrics_collector.set_gauge(name, value, labels)
        elif metric_type == MetricType.HISTOGRAM:
            self.metrics_collector.observe_histogram(name, value, labels)
        elif metric_type == MetricType.SUMMARY:
            self.metrics_collector.observe_summary(name, value, labels)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get all metrics."""
        return self.metrics_collector.get_metrics()
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status."""
        return {
            "overall": self.health_checker.get_overall_health().value,
            "checks": {name: check.__dict__ for name, check in self.health_checker.results.items()}
        }
    
    def get_alerts(self) -> List[Dict[str, Any]]:
        """Get active alerts."""
        return [alert.__dict__ for alert in self.alert_manager.get_active_alerts()]
    
    def get_trace(self, trace_id: str) -> List[Dict[str, Any]]:
        """Get trace data."""
        spans = self.tracer.get_trace(trace_id)
        return [span.__dict__ for span in spans]
    
    def cleanup(self):
        """Cleanup monitoring system."""
        if self.db_connection:
            self.db_connection.close()
        
        if self.redis_client:
            self.redis_client.close()


# Example usage and demonstration
async def main():
    """Demonstrate the advanced monitoring system."""
    print("📊 HeyGen AI - Advanced Monitoring System Demo")
    print("=" * 70)
    
    # Initialize monitoring system
    monitoring = AdvancedMonitoringSystem(
        prometheus_port=9090,
        redis_url="redis://localhost:6379/0",
        db_path="monitoring.db",
        enable_websocket=True,
        websocket_port=9091
    )
    
    try:
        # Start a trace
        print("\n🔍 Starting Distributed Trace...")
        span_id = monitoring.start_trace("demo_operation", {"service": "demo", "version": "1.0.0"})
        
        # Simulate some work
        await asyncio.sleep(1)
        
        # Add trace log
        monitoring.add_trace_log(span_id, "Processing started", "info", {"step": "initialization"})
        
        # Record some metrics
        print("\n📈 Recording Metrics...")
        monitoring.record_metric("demo_requests_total", 1, MetricType.COUNTER, {"endpoint": "/demo"})
        monitoring.record_metric("demo_response_time", 0.5, MetricType.HISTOGRAM, {"endpoint": "/demo"})
        monitoring.record_metric("demo_active_connections", 10, MetricType.GAUGE)
        
        # Simulate more work
        await asyncio.sleep(2)
        
        # Add another trace log
        monitoring.add_trace_log(span_id, "Processing completed", "info", {"step": "completion"})
        
        # Finish trace
        monitoring.finish_trace(span_id, "success", {"result": "completed"})
        
        # Get metrics
        print("\n📊 Current Metrics:")
        metrics = monitoring.get_metrics()
        for metric_type, metric_data in metrics.items():
            print(f"  {metric_type}:")
            for name, value in metric_data.items():
                print(f"    {name}: {value}")
        
        # Get health status
        print("\n🏥 Health Status:")
        health = monitoring.get_health_status()
        print(f"  Overall: {health['overall']}")
        for check_name, check_data in health['checks'].items():
            print(f"  {check_name}: {check_data['status']} - {check_data['message']}")
        
        # Get alerts
        print("\n🚨 Active Alerts:")
        alerts = monitoring.get_alerts()
        if alerts:
            for alert in alerts:
                print(f"  {alert['name']} ({alert['level']}): {alert['message']}")
        else:
            print("  No active alerts")
        
        # Get trace
        print("\n🔍 Trace Data:")
        trace_id = trace_id_var.get()
        if trace_id:
            trace = monitoring.get_trace(trace_id)
            for span in trace:
                print(f"  {span['operation_name']}: {span['duration']:.3f}s ({span['status']})")
        
        print(f"\n🌐 Prometheus metrics available at: http://localhost:{monitoring.prometheus_port}/metrics")
        print(f"🔌 WebSocket server running on: ws://localhost:{monitoring.websocket_port}")
        
    except Exception as e:
        print(f"❌ Demo Error: {e}")
        logging.error(f"Demo failed: {e}")
    
    finally:
        # Cleanup
        monitoring.cleanup()
        print("\n✅ Demo completed")


if __name__ == "__main__":
    asyncio.run(main())

