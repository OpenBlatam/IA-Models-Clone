from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import asyncio
import json
import time
import statistics
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
from contextlib import asynccontextmanager
from fastapi import Request, Response
from pydantic import BaseModel, Field
import psutil
from typing import Any, List, Dict, Optional
import logging
"""
API Performance Metrics System

This module provides comprehensive performance monitoring for API endpoints,
tracking response time, latency, throughput, and other key performance indicators
with real-time monitoring, analytics, and alerting capabilities.
"""




class MetricType(Enum):
    """Types of performance metrics."""
    RESPONSE_TIME = "response_time"
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    MEMORY_USAGE = "memory_usage"
    CPU_USAGE = "cpu_usage"
    CONCURRENT_REQUESTS = "concurrent_requests"
    CACHE_HIT_RATE = "cache_hit_rate"
    DATABASE_QUERIES = "database_queries"
    EXTERNAL_API_CALLS = "external_api_calls"


class MetricAggregation(Enum):
    """Aggregation methods for metrics."""
    MEAN = "mean"
    MEDIAN = "median"
    P95 = "p95"
    P99 = "p99"
    MIN = "min"
    MAX = "max"
    COUNT = "count"
    SUM = "sum"


@dataclass
class PerformanceMetric:
    """Individual performance metric."""
    name: str
    value: float
    timestamp: float
    endpoint: str
    method: str
    status_code: int
    request_id: str
    user_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricSummary:
    """Summary statistics for a metric."""
    count: int = 0
    mean: float = 0.0
    median: float = 0.0
    p95: float = 0.0
    p99: float = 0.0
    min: float = 0.0
    max: float = 0.0
    sum: float = 0.0
    std_dev: float = 0.0


class PerformanceThreshold(BaseModel):
    """Performance threshold configuration."""
    metric_type: MetricType
    endpoint: str
    threshold_value: float
    comparison: str = "gt"  # gt, lt, gte, lte, eq
    alert_message: str = ""
    severity: str = "warning"  # warning, error, critical


class PerformanceAlert(BaseModel):
    """Performance alert."""
    threshold: PerformanceThreshold
    current_value: float
    timestamp: float
    message: str
    severity: str


class APIPerformanceMetrics:
    """
    Comprehensive API performance metrics system.
    
    Tracks response time, latency, throughput, and other key performance
    indicators with real-time monitoring and analytics.
    """
    
    def __init__(
        self,
        max_metrics: int = 10000,
        window_size: int = 300,  # 5 minutes
        enable_alerts: bool = True,
        enable_persistence: bool = False
    ):
        """
        Initialize performance metrics system.
        
        Args:
            max_metrics: Maximum number of metrics to keep in memory
            window_size: Time window for metrics in seconds
            enable_alerts: Enable performance alerts
            enable_persistence: Enable metrics persistence
        """
        self.max_metrics = max_metrics
        self.window_size = window_size
        self.enable_alerts = enable_alerts
        self.enable_persistence = enable_persistence
        
        # Metrics storage
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_metrics))
        self.endpoint_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_metrics))
        self.method_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_metrics))
        
        # Performance tracking
        self.request_times: deque = deque(maxlen=max_metrics)
        self.error_counts: Dict[str, int] = defaultdict(int)
        self.success_counts: Dict[str, int] = defaultdict(int)
        
        # System metrics
        self.memory_usage: deque = deque(maxlen=max_metrics)
        self.cpu_usage: deque = deque(maxlen=max_metrics)
        self.concurrent_requests: int = 0
        self.max_concurrent_requests: int = 0
        
        # Throughput tracking
        self.request_count: int = 0
        self.start_time: float = time.time()
        self.last_throughput_calc: float = time.time()
        self.throughput_history: deque = deque(maxlen=100)
        
        # Cache metrics
        self.cache_hits: int = 0
        self.cache_misses: int = 0
        
        # Database metrics
        self.db_query_count: int = 0
        self.db_query_time: float = 0.0
        
        # External API metrics
        self.external_api_calls: int = 0
        self.external_api_time: float = 0.0
        
        # Alerts
        self.thresholds: List[PerformanceThreshold] = []
        self.alerts: List[PerformanceAlert] = []
        
        # Background tasks
        self._cleanup_task: Optional[asyncio.Task] = None
        self._system_metrics_task: Optional[asyncio.Task] = None
        
        if enable_alerts:
            self._setup_default_thresholds()
        
        self._start_background_tasks()
    
    def _setup_default_thresholds(self) -> Any:
        """Setup default performance thresholds."""
        default_thresholds = [
            PerformanceThreshold(
                metric_type=MetricType.RESPONSE_TIME,
                endpoint="*",
                threshold_value=1000.0,  # 1 second
                comparison="gt",
                alert_message="Response time exceeded 1 second",
                severity="warning"
            ),
            PerformanceThreshold(
                metric_type=MetricType.RESPONSE_TIME,
                endpoint="*",
                threshold_value=5000.0,  # 5 seconds
                comparison="gt",
                alert_message="Response time exceeded 5 seconds",
                severity="error"
            ),
            PerformanceThreshold(
                metric_type=MetricType.ERROR_RATE,
                endpoint="*",
                threshold_value=0.05,  # 5% error rate
                comparison="gt",
                alert_message="Error rate exceeded 5%",
                severity="warning"
            ),
            PerformanceThreshold(
                metric_type=MetricType.MEMORY_USAGE,
                endpoint="*",
                threshold_value=0.8,  # 80% memory usage
                comparison="gt",
                alert_message="Memory usage exceeded 80%",
                severity="warning"
            ),
            PerformanceThreshold(
                metric_type=MetricType.CPU_USAGE,
                endpoint="*",
                threshold_value=0.8,  # 80% CPU usage
                comparison="gt",
                alert_message="CPU usage exceeded 80%",
                severity="warning"
            )
        ]
        
        self.thresholds.extend(default_thresholds)
    
    def _start_background_tasks(self) -> Any:
        """Start background tasks for metrics management."""
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._cleanup_old_metrics())
        
        if self._system_metrics_task is None or self._system_metrics_task.done():
            self._system_metrics_task = asyncio.create_task(self._collect_system_metrics())
    
    async def _cleanup_old_metrics(self) -> Any:
        """Cleanup old metrics outside the time window."""
        while True:
            try:
                current_time = time.time()
                cutoff_time = current_time - self.window_size
                
                # Cleanup metrics
                for metric_type in list(self.metrics.keys()):
                    while (self.metrics[metric_type] and 
                           self.metrics[metric_type][0].timestamp < cutoff_time):
                        self.metrics[metric_type].popleft()
                
                # Cleanup endpoint metrics
                for endpoint in list(self.endpoint_metrics.keys()):
                    while (self.endpoint_metrics[endpoint] and 
                           self.endpoint_metrics[endpoint][0].timestamp < cutoff_time):
                        self.endpoint_metrics[endpoint].popleft()
                
                # Cleanup method metrics
                for method in list(self.method_metrics.keys()):
                    while (self.method_metrics[method] and 
                           self.method_metrics[method][0].timestamp < cutoff_time):
                        self.method_metrics[method].popleft()
                
                await asyncio.sleep(60)  # Cleanup every minute
                
            except Exception as e:
                print(f"Error in metrics cleanup: {e}")
                await asyncio.sleep(60)
    
    async def _collect_system_metrics(self) -> Any:
        """Collect system metrics periodically."""
        while True:
            try:
                # Memory usage
                memory_percent = psutil.virtual_memory().percent / 100.0
                self.memory_usage.append(memory_percent)
                
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=1) / 100.0
                self.cpu_usage.append(cpu_percent)
                
                # Check thresholds
                if self.enable_alerts:
                    await self._check_thresholds()
                
                await asyncio.sleep(10)  # Collect every 10 seconds
                
            except Exception as e:
                print(f"Error collecting system metrics: {e}")
                await asyncio.sleep(10)
    
    async def _check_thresholds(self) -> Any:
        """Check performance thresholds and generate alerts."""
        current_time = time.time()
        
        for threshold in self.thresholds:
            current_value = await self._get_current_value(threshold.metric_type, threshold.endpoint)
            
            if current_value is None:
                continue
            
            # Check threshold
            should_alert = False
            if threshold.comparison == "gt" and current_value > threshold.threshold_value:
                should_alert = True
            elif threshold.comparison == "lt" and current_value < threshold.threshold_value:
                should_alert = True
            elif threshold.comparison == "gte" and current_value >= threshold.threshold_value:
                should_alert = True
            elif threshold.comparison == "lte" and current_value <= threshold.threshold_value:
                should_alert = True
            elif threshold.comparison == "eq" and current_value == threshold.threshold_value:
                should_alert = True
            
            if should_alert:
                alert = PerformanceAlert(
                    threshold=threshold,
                    current_value=current_value,
                    timestamp=current_time,
                    message=threshold.alert_message,
                    severity=threshold.severity
                )
                self.alerts.append(alert)
    
    async def _get_current_value(self, metric_type: MetricType, endpoint: str) -> Optional[float]:
        """Get current value for a metric type and endpoint."""
        if metric_type == MetricType.RESPONSE_TIME:
            return self.get_response_time_stats(endpoint).mean
        elif metric_type == MetricType.ERROR_RATE:
            total_requests = self.success_counts.get(endpoint, 0) + self.error_counts.get(endpoint, 0)
            if total_requests == 0:
                return 0.0
            return self.error_counts.get(endpoint, 0) / total_requests
        elmatch metric_type:
    case MetricType.MEMORY_USAGE:
            return statistics.mean(self.memory_usage) if self.memory_usage else 0.0
        elmatch metric_type:
    case MetricType.CPU_USAGE:
            return statistics.mean(self.cpu_usage) if self.cpu_usage else 0.0
        elif metric_type == MetricType.THROUGHPUT:
            return self.get_throughput()
        elif metric_type == MetricType.CACHE_HIT_RATE:
            total_cache_requests = self.cache_hits + self.cache_misses
            if total_cache_requests == 0:
                return 0.0
            return self.cache_hits / total_cache_requests
        
        return None
    
    def record_request(
        self,
        endpoint: str,
        method: str,
        status_code: int,
        response_time: float,
        request_id: str,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Record a request metric.
        
        Args:
            endpoint: API endpoint
            method: HTTP method
            status_code: HTTP status code
            response_time: Response time in milliseconds
            request_id: Unique request ID
            user_id: User ID (optional)
            metadata: Additional metadata (optional)
        """
        current_time = time.time()
        
        # Create metric
        metric = PerformanceMetric(
            name="response_time",
            value=response_time,
            timestamp=current_time,
            endpoint=endpoint,
            method=method,
            status_code=status_code,
            request_id=request_id,
            user_id=user_id,
            metadata=metadata or {}
        )
        
        # Store metrics
        self.metrics["response_time"].append(metric)
        self.endpoint_metrics[endpoint].append(metric)
        self.method_metrics[method].append(metric)
        
        # Update counters
        self.request_count += 1
        self.request_times.append(response_time)
        
        if 200 <= status_code < 400:
            self.success_counts[endpoint] += 1
        else:
            self.error_counts[endpoint] += 1
        
        # Update throughput
        self._update_throughput()
        
        # Update concurrent requests
        self.concurrent_requests += 1
        self.max_concurrent_requests = max(self.max_concurrent_requests, self.concurrent_requests)
    
    async def record_request_end(self) -> Any:
        """Record the end of a request (for concurrent request tracking)."""
        self.concurrent_requests = max(0, self.concurrent_requests - 1)
    
    def record_cache_hit(self) -> Any:
        """Record a cache hit."""
        self.cache_hits += 1
    
    def record_cache_miss(self) -> Any:
        """Record a cache miss."""
        self.cache_misses += 1
    
    def record_database_query(self, query_time: float):
        """Record a database query."""
        self.db_query_count += 1
        self.db_query_time += query_time
    
    def record_external_api_call(self, call_time: float):
        """Record an external API call."""
        self.external_api_calls += 1
        self.external_api_time += call_time
    
    def _update_throughput(self) -> Any:
        """Update throughput calculation."""
        current_time = time.time()
        time_diff = current_time - self.last_throughput_calc
        
        if time_diff >= 1.0:  # Calculate every second
            throughput = self.request_count / (current_time - self.start_time)
            self.throughput_history.append(throughput)
            self.last_throughput_calc = current_time
    
    def get_response_time_stats(self, endpoint: str = "*") -> MetricSummary:
        """Get response time statistics for an endpoint."""
        if endpoint == "*":
            metrics = list(self.metrics["response_time"])
        else:
            metrics = list(self.endpoint_metrics.get(endpoint, []))
        
        if not metrics:
            return MetricSummary()
        
        values = [m.value for m in metrics]
        values.sort()
        
        return MetricSummary(
            count=len(values),
            mean=statistics.mean(values),
            median=statistics.median(values),
            p95=values[int(len(values) * 0.95)] if len(values) > 0 else 0.0,
            p99=values[int(len(values) * 0.99)] if len(values) > 0 else 0.0,
            min=min(values),
            max=max(values),
            sum=sum(values),
            std_dev=statistics.stdev(values) if len(values) > 1 else 0.0
        )
    
    def get_throughput(self) -> float:
        """Get current throughput (requests per second)."""
        if not self.throughput_history:
            return 0.0
        return statistics.mean(self.throughput_history)
    
    def get_error_rate(self, endpoint: str = "*") -> float:
        """Get error rate for an endpoint."""
        if endpoint == "*":
            total_requests = sum(self.success_counts.values()) + sum(self.error_counts.values())
            total_errors = sum(self.error_counts.values())
        else:
            total_requests = self.success_counts.get(endpoint, 0) + self.error_counts.get(endpoint, 0)
            total_errors = self.error_counts.get(endpoint, 0)
        
        if total_requests == 0:
            return 0.0
        
        return total_errors / total_requests
    
    def get_cache_hit_rate(self) -> float:
        """Get cache hit rate."""
        total_requests = self.cache_hits + self.cache_misses
        if total_requests == 0:
            return 0.0
        return self.cache_hits / total_requests
    
    def get_memory_usage(self) -> float:
        """Get current memory usage percentage."""
        if not self.memory_usage:
            return 0.0
        return statistics.mean(self.memory_usage)
    
    def get_cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
        if not self.cpu_usage:
            return 0.0
        return statistics.mean(self.cpu_usage)
    
    async def get_concurrent_requests(self) -> Dict[str, int]:
        """Get concurrent request statistics."""
        return {
            "current": self.concurrent_requests,
            "max": self.max_concurrent_requests
        }
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database performance statistics."""
        return {
            "query_count": self.db_query_count,
            "total_time": self.db_query_time,
            "avg_query_time": self.db_query_time / max(1, self.db_query_count)
        }
    
    async def get_external_api_stats(self) -> Dict[str, Any]:
        """Get external API performance statistics."""
        return {
            "call_count": self.external_api_calls,
            "total_time": self.external_api_time,
            "avg_call_time": self.external_api_time / max(1, self.external_api_calls)
        }
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        return {
            "response_time": {
                "overall": self.get_response_time_stats().__dict__,
                "by_endpoint": {
                    endpoint: self.get_response_time_stats(endpoint).__dict__
                    for endpoint in self.endpoint_metrics.keys()
                }
            },
            "throughput": {
                "current": self.get_throughput(),
                "history": list(self.throughput_history)
            },
            "error_rate": {
                "overall": self.get_error_rate(),
                "by_endpoint": {
                    endpoint: self.get_error_rate(endpoint)
                    for endpoint in set(list(self.success_counts.keys()) + list(self.error_counts.keys()))
                }
            },
            "cache": {
                "hit_rate": self.get_cache_hit_rate(),
                "hits": self.cache_hits,
                "misses": self.cache_misses
            },
            "system": {
                "memory_usage": self.get_memory_usage(),
                "cpu_usage": self.get_cpu_usage(),
                "concurrent_requests": self.get_concurrent_requests()
            },
            "database": self.get_database_stats(),
            "external_api": self.get_external_api_stats(),
            "alerts": [alert.dict() for alert in self.alerts[-10:]],  # Last 10 alerts
            "uptime": time.time() - self.start_time
        }
    
    def add_threshold(self, threshold: PerformanceThreshold):
        """Add a performance threshold."""
        self.thresholds.append(threshold)
    
    def get_alerts(self, severity: Optional[str] = None) -> List[PerformanceAlert]:
        """Get performance alerts."""
        if severity:
            return [alert for alert in self.alerts if alert.severity == severity]
        return self.alerts
    
    def clear_alerts(self) -> Any:
        """Clear all alerts."""
        self.alerts.clear()
    
    async def close(self) -> Any:
        """Close the performance metrics system."""
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        if self._system_metrics_task and not self._system_metrics_task.done():
            self._system_metrics_task.cancel()
            try:
                await self._system_metrics_task
            except asyncio.CancelledError:
                pass


# Global performance metrics instance
_performance_metrics: Optional[APIPerformanceMetrics] = None


def get_performance_metrics() -> APIPerformanceMetrics:
    """Get global performance metrics instance."""
    global _performance_metrics
    if _performance_metrics is None:
        _performance_metrics = APIPerformanceMetrics()
    return _performance_metrics


def set_performance_metrics(metrics: APIPerformanceMetrics):
    """Set global performance metrics instance."""
    global _performance_metrics
    _performance_metrics = metrics


# FastAPI middleware for performance tracking
@asynccontextmanager
async def performance_tracking(request: Request, response: Response):
    """Context manager for tracking request performance."""
    start_time = time.time()
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    
    try:
        yield
    finally:
        end_time = time.time()
        response_time = (end_time - start_time) * 1000  # Convert to milliseconds
        
        # Record metrics
        metrics = get_performance_metrics()
        metrics.record_request(
            endpoint=request.url.path,
            method=request.method,
            status_code=response.status_code,
            response_time=response_time,
            request_id=request_id,
            user_id=getattr(request.state, "user_id", None),
            metadata={
                "query_params": dict(request.query_params),
                "headers": dict(request.headers),
                "client_ip": request.client.host if request.client else None
            }
        )
        metrics.record_request_end()


# Performance decorators
def track_performance(endpoint_name: Optional[str] = None):
    """Decorator to track function performance."""
    def decorator(func) -> Any:
        async def wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                end_time = time.time()
                response_time = (end_time - start_time) * 1000
                
                metrics = get_performance_metrics()
                metrics.record_request(
                    endpoint=endpoint_name or func.__name__,
                    method="FUNCTION",
                    status_code=200,
                    response_time=response_time,
                    request_id=str(uuid.uuid4())
                )
        
        return wrapper
    return decorator


def track_cache_performance(func) -> Any:
    """Decorator to track cache performance."""
    async def wrapper(*args, **kwargs) -> Any:
        metrics = get_performance_metrics()
        
        # Check if result is in cache (simplified)
        cache_key = str(args) + str(kwargs)
        if hasattr(func, '_cache') and cache_key in func._cache:
            metrics.record_cache_hit()
        else:
            metrics.record_cache_miss()
        
        return await func(*args, **kwargs)
    
    return wrapper


def track_database_performance(func) -> Any:
    """Decorator to track database performance."""
    async def wrapper(*args, **kwargs) -> Any:
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            return result
        finally:
            end_time = time.time()
            query_time = (end_time - start_time) * 1000
            
            metrics = get_performance_metrics()
            metrics.record_database_query(query_time)
    
    return wrapper


async def track_external_api_performance(func) -> Any:
    """Decorator to track external API performance."""
    async def wrapper(*args, **kwargs) -> Any:
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            return result
        finally:
            end_time = time.time()
            call_time = (end_time - start_time) * 1000
            
            metrics = get_performance_metrics()
            metrics.record_external_api_call(call_time)
    
    return wrapper


# Performance monitoring utilities
class PerformanceMonitor:
    """Utility class for performance monitoring."""
    
    def __init__(self) -> Any:
        self.metrics = get_performance_metrics()
    
    def get_response_time_percentiles(self, endpoint: str = "*") -> Dict[str, float]:
        """Get response time percentiles."""
        stats = self.metrics.get_response_time_stats(endpoint)
        return {
            "p50": stats.median,
            "p95": stats.p95,
            "p99": stats.p99,
            "p99.9": stats.p99  # Simplified
        }
    
    def get_throughput_trend(self, window_minutes: int = 5) -> List[float]:
        """Get throughput trend over time window."""
        # Simplified implementation
        return list(self.metrics.throughput_history)
    
    def get_error_rate_trend(self, endpoint: str = "*") -> float:
        """Get error rate trend."""
        return self.metrics.get_error_rate(endpoint)
    
    def get_system_health_score(self) -> float:
        """Calculate system health score (0-100)."""
        score = 100.0
        
        # Response time penalty
        response_time = self.metrics.get_response_time_stats().mean
        if response_time > 1000:  # > 1 second
            score -= 20
        elif response_time > 500:  # > 500ms
            score -= 10
        
        # Error rate penalty
        error_rate = self.metrics.get_error_rate()
        if error_rate > 0.05:  # > 5%
            score -= 30
        elif error_rate > 0.01:  # > 1%
            score -= 15
        
        # Memory usage penalty
        memory_usage = self.metrics.get_memory_usage()
        if memory_usage > 0.9:  # > 90%
            score -= 20
        elif memory_usage > 0.8:  # > 80%
            score -= 10
        
        # CPU usage penalty
        cpu_usage = self.metrics.get_cpu_usage()
        if cpu_usage > 0.9:  # > 90%
            score -= 20
        elif cpu_usage > 0.8:  # > 80%
            score -= 10
        
        return max(0.0, score)
    
    def get_performance_recommendations(self) -> List[str]:
        """Get performance improvement recommendations."""
        recommendations = []
        
        # Response time recommendations
        response_time = self.metrics.get_response_time_stats().mean
        if response_time > 1000:
            recommendations.append("Consider implementing caching for slow endpoints")
            recommendations.append("Optimize database queries")
        elif response_time > 500:
            recommendations.append("Monitor response times closely")
        
        # Error rate recommendations
        error_rate = self.metrics.get_error_rate()
        if error_rate > 0.05:
            recommendations.append("Investigate and fix error sources")
            recommendations.append("Implement better error handling")
        elif error_rate > 0.01:
            recommendations.append("Monitor error patterns")
        
        # Memory usage recommendations
        memory_usage = self.metrics.get_memory_usage()
        if memory_usage > 0.9:
            recommendations.append("Consider scaling up memory resources")
            recommendations.append("Implement memory cleanup")
        elif memory_usage > 0.8:
            recommendations.append("Monitor memory usage trends")
        
        # Cache recommendations
        cache_hit_rate = self.metrics.get_cache_hit_rate()
        if cache_hit_rate < 0.5:
            recommendations.append("Consider implementing or improving caching strategy")
        
        return recommendations


# Export main classes and functions
__all__ = [
    "APIPerformanceMetrics",
    "PerformanceMetric",
    "MetricSummary",
    "PerformanceThreshold",
    "PerformanceAlert",
    "PerformanceMonitor",
    "get_performance_metrics",
    "set_performance_metrics",
    "performance_tracking",
    "track_performance",
    "track_cache_performance",
    "track_database_performance",
    "track_external_api_performance"
] 