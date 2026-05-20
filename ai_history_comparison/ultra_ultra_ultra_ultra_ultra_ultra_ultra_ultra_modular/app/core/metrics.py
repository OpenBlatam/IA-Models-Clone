"""
Advanced metrics and monitoring system.
"""

import time
import asyncio
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass, field
from functools import wraps
import threading

from prometheus_client import Counter, Histogram, Gauge, Summary, CollectorRegistry, generate_latest
from ..core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class MetricPoint:
    """Single metric data point."""
    timestamp: datetime
    value: float
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class TimeSeries:
    """Time series data for metrics."""
    name: str
    points: deque = field(default_factory=lambda: deque(maxlen=1000))
    labels: Dict[str, str] = field(default_factory=dict)


class MetricsCollector:
    """Advanced metrics collector with multiple backends."""
    
    def __init__(self):
        self.registry = CollectorRegistry()
        self.time_series: Dict[str, TimeSeries] = {}
        self.custom_metrics: Dict[str, Any] = {}
        self._lock = threading.Lock()
        
        # Prometheus metrics
        self._init_prometheus_metrics()
        
        # Performance tracking
        self.performance_data: Dict[str, List[float]] = defaultdict(list)
        self.error_counts: Dict[str, int] = defaultdict(int)
        self.request_counts: Dict[str, int] = defaultdict(int)
    
    def _init_prometheus_metrics(self) -> None:
        """Initialize Prometheus metrics."""
        # Request metrics
        self.request_total = Counter(
            'http_requests_total',
            'Total HTTP requests',
            ['method', 'endpoint', 'status'],
            registry=self.registry
        )
        
        self.request_duration = Histogram(
            'http_request_duration_seconds',
            'HTTP request duration',
            ['method', 'endpoint'],
            registry=self.registry
        )
        
        # Plugin metrics
        self.plugin_operations = Counter(
            'plugin_operations_total',
            'Total plugin operations',
            ['operation', 'plugin_name', 'status'],
            registry=self.registry
        )
        
        self.plugin_duration = Histogram(
            'plugin_operation_duration_seconds',
            'Plugin operation duration',
            ['operation', 'plugin_name'],
            registry=self.registry
        )
        
        # Analysis metrics
        self.analysis_operations = Counter(
            'analysis_operations_total',
            'Total analysis operations',
            ['analysis_type', 'status'],
            registry=self.registry
        )
        
        self.analysis_duration = Histogram(
            'analysis_duration_seconds',
            'Analysis duration',
            ['analysis_type'],
            registry=self.registry
        )
        
        # System metrics
        self.active_connections = Gauge(
            'active_connections',
            'Number of active connections',
            registry=self.registry
        )
        
        self.cache_hits = Counter(
            'cache_hits_total',
            'Total cache hits',
            ['cache_type'],
            registry=self.registry
        )
        
        self.cache_misses = Counter(
            'cache_misses_total',
            'Total cache misses',
            ['cache_type'],
            registry=self.registry
        )
        
        # Custom metrics
        self.custom_counters: Dict[str, Counter] = {}
        self.custom_gauges: Dict[str, Gauge] = {}
        self.custom_histograms: Dict[str, Histogram] = {}
    
    def record_request(self, method: str, endpoint: str, status: int, duration: float) -> None:
        """Record HTTP request metrics."""
        self.request_total.labels(method=method, endpoint=endpoint, status=str(status)).inc()
        self.request_duration.labels(method=method, endpoint=endpoint).observe(duration)
        
        # Update internal counters
        self.request_counts[f"{method}:{endpoint}"] += 1
    
    def record_plugin_operation(self, operation: str, plugin_name: str, status: str, duration: float) -> None:
        """Record plugin operation metrics."""
        self.plugin_operations.labels(
            operation=operation, 
            plugin_name=plugin_name, 
            status=status
        ).inc()
        self.plugin_duration.labels(
            operation=operation, 
            plugin_name=plugin_name
        ).observe(duration)
    
    def record_analysis(self, analysis_type: str, status: str, duration: float) -> None:
        """Record analysis metrics."""
        self.analysis_operations.labels(
            analysis_type=analysis_type, 
            status=status
        ).inc()
        self.analysis_duration.labels(analysis_type=analysis_type).observe(duration)
    
    def record_cache_operation(self, operation: str, cache_type: str) -> None:
        """Record cache operation metrics."""
        if operation == "hit":
            self.cache_hits.labels(cache_type=cache_type).inc()
        elif operation == "miss":
            self.cache_misses.labels(cache_type=cache_type).inc()
    
    def record_error(self, error_type: str, endpoint: str = None) -> None:
        """Record error metrics."""
        with self._lock:
            self.error_counts[error_type] += 1
            if endpoint:
                self.error_counts[f"{error_type}:{endpoint}"] += 1
    
    def record_performance(self, operation: str, duration: float) -> None:
        """Record performance metrics."""
        with self._lock:
            self.performance_data[operation].append(duration)
            # Keep only last 1000 measurements
            if len(self.performance_data[operation]) > 1000:
                self.performance_data[operation] = self.performance_data[operation][-1000:]
    
    def add_time_series_point(self, name: str, value: float, labels: Dict[str, str] = None) -> None:
        """Add point to time series."""
        with self._lock:
            if name not in self.time_series:
                self.time_series[name] = TimeSeries(name=name, labels=labels or {})
            
            point = MetricPoint(
                timestamp=datetime.utcnow(),
                value=value,
                labels=labels or {}
            )
            self.time_series[name].points.append(point)
    
    def get_time_series(self, name: str, since: Optional[datetime] = None) -> List[MetricPoint]:
        """Get time series data."""
        with self._lock:
            if name not in self.time_series:
                return []
            
            points = list(self.time_series[name].points)
            if since:
                points = [p for p in points if p.timestamp >= since]
            
            return points
    
    def get_performance_stats(self, operation: str) -> Dict[str, float]:
        """Get performance statistics for an operation."""
        with self._lock:
            if operation not in self.performance_data or not self.performance_data[operation]:
                return {}
            
            durations = self.performance_data[operation]
            return {
                "count": len(durations),
                "min": min(durations),
                "max": max(durations),
                "avg": sum(durations) / len(durations),
                "p50": self._percentile(durations, 50),
                "p95": self._percentile(durations, 95),
                "p99": self._percentile(durations, 99)
            }
    
    def get_error_stats(self) -> Dict[str, int]:
        """Get error statistics."""
        with self._lock:
            return dict(self.error_counts)
    
    def get_request_stats(self) -> Dict[str, int]:
        """Get request statistics."""
        with self._lock:
            return dict(self.request_counts)
    
    def get_prometheus_metrics(self) -> str:
        """Get Prometheus metrics in text format."""
        return generate_latest(self.registry).decode('utf-8')
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all metrics in a structured format."""
        with self._lock:
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "performance": {
                    operation: self.get_performance_stats(operation)
                    for operation in self.performance_data.keys()
                },
                "errors": self.get_error_stats(),
                "requests": self.get_request_stats(),
                "time_series": {
                    name: [
                        {
                            "timestamp": point.timestamp.isoformat(),
                            "value": point.value,
                            "labels": point.labels
                        }
                        for point in series.points
                    ]
                    for name, series in self.time_series.items()
                }
            }
    
    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile of data."""
        if not data:
            return 0.0
        
        sorted_data = sorted(data)
        index = int((percentile / 100) * len(sorted_data))
        return sorted_data[min(index, len(sorted_data) - 1)]
    
    def reset_metrics(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self.performance_data.clear()
            self.error_counts.clear()
            self.request_counts.clear()
            self.time_series.clear()
            
            # Reset Prometheus metrics
            for metric in self.registry.collect():
                if hasattr(metric, 'clear'):
                    metric.clear()


# Global metrics collector
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get global metrics collector instance."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


def track_performance(operation: str) -> callable:
    """Decorator to track function performance."""
    def decorator(func: callable) -> callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                get_metrics_collector().record_performance(operation, duration)
                return result
            except Exception as e:
                duration = time.time() - start_time
                get_metrics_collector().record_performance(operation, duration)
                get_metrics_collector().record_error(f"{operation}_error")
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                get_metrics_collector().record_performance(operation, duration)
                return result
            except Exception as e:
                duration = time.time() - start_time
                get_metrics_collector().record_performance(operation, duration)
                get_metrics_collector().record_error(f"{operation}_error")
                raise
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def track_requests(endpoint: str) -> callable:
    """Decorator to track HTTP requests."""
    def decorator(func: callable) -> callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            status = 200
            
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                status = 500
                raise
            finally:
                duration = time.time() - start_time
                get_metrics_collector().record_request("POST", endpoint, status, duration)
        
        return wrapper
    
    return decorator


class MetricsMiddleware:
    """Middleware for automatic metrics collection."""
    
    def __init__(self, app):
        self.app = app
        self.metrics = get_metrics_collector()
    
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        start_time = time.time()
        status = 200
        
        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                nonlocal status
                status = message["status"]
            await send(message)
        
        try:
            await self.app(scope, receive, send_wrapper)
        except Exception as e:
            status = 500
            self.metrics.record_error("http_error", scope["path"])
            raise
        finally:
            duration = time.time() - start_time
            method = scope["method"]
            path = scope["path"]
            self.metrics.record_request(method, path, status, duration)


# Utility functions
async def record_plugin_metrics(operation: str, plugin_name: str, success: bool, duration: float) -> None:
    """Record plugin operation metrics."""
    status = "success" if success else "error"
    get_metrics_collector().record_plugin_operation(operation, plugin_name, status, duration)


async def record_analysis_metrics(analysis_type: str, success: bool, duration: float) -> None:
    """Record analysis operation metrics."""
    status = "success" if success else "error"
    get_metrics_collector().record_analysis(analysis_type, status, duration)


async def record_cache_metrics(operation: str, cache_type: str) -> None:
    """Record cache operation metrics."""
    get_metrics_collector().record_cache_operation(operation, cache_type)


async def get_system_metrics() -> Dict[str, Any]:
    """Get comprehensive system metrics."""
    collector = get_metrics_collector()
    return collector.get_all_metrics()


async def get_prometheus_metrics() -> str:
    """Get Prometheus metrics."""
    collector = get_metrics_collector()
    return collector.get_prometheus_metrics()


