from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import time
import asyncio
import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import psutil
import threading
from contextlib import contextmanager
from .config import get_config
from typing import Any, List, Dict, Optional
"""
Monitoring and Metrics
=====================

Comprehensive monitoring system with metrics collection, performance tracking,
and alerting capabilities.
"""



# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class MetricPoint:
    """Individual metric data point"""
    timestamp: float
    value: float
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class MetricSeries:
    """Time series of metric points"""
    name: str
    points: deque = field(default_factory=lambda: deque(maxlen=1000))
    labels: Dict[str, str] = field(default_factory=dict)
    
    def add_point(self, value: float, labels: Optional[Dict[str, str]] = None):
        """Add a new metric point"""
        point = MetricPoint(
            timestamp=time.time(),
            value=value,
            labels={**self.labels, **(labels or {})}
        )
        self.points.append(point)
    
    def get_recent_points(self, seconds: int = 300) -> List[MetricPoint]:
        """Get points from the last N seconds"""
        cutoff = time.time() - seconds
        return [p for p in self.points if p.timestamp >= cutoff]
    
    def get_average(self, seconds: int = 300) -> float:
        """Get average value over the last N seconds"""
        points = self.get_recent_points(seconds)
        if not points:
            return 0.0
        return sum(p.value for p in points) / len(points)
    
    def get_max(self, seconds: int = 300) -> float:
        """Get maximum value over the last N seconds"""
        points = self.get_recent_points(seconds)
        if not points:
            return 0.0
        return max(p.value for p in points)
    
    def get_min(self, seconds: int = 300) -> float:
        """Get minimum value over the last N seconds"""
        points = self.get_recent_points(seconds)
        if not points:
            return 0.0
        return min(p.value for p in points)


class MetricsCollector:
    """Centralized metrics collection and monitoring"""
    
    def __init__(self) -> Any:
        self.metrics: Dict[str, MetricSeries] = {}
        self.counters: Dict[str, float] = defaultdict(float)
        self.gauges: Dict[str, float] = {}
        self.histograms: Dict[str, List[float]] = defaultdict(list)
        
        # Performance tracking
        self.request_times: deque = deque(maxlen=1000)
        self.error_counts: Dict[str, int] = defaultdict(int)
        self.success_counts: Dict[str, int] = defaultdict(int)
        
        # System monitoring
        self.start_time = time.time()
        self._monitoring_active = False
        self._monitoring_thread = None
        
        # Alerts
        self.alert_thresholds: Dict[str, Dict[str, Any]] = {}
        self.alert_callbacks: List[Callable] = []
        
        self._setup_default_metrics()
    
    def _setup_default_metrics(self) -> Any:
        """Setup default system metrics"""
        self.metrics["requests_total"] = MetricSeries("requests_total")
        self.metrics["request_duration"] = MetricSeries("request_duration")
        self.metrics["errors_total"] = MetricSeries("errors_total")
        self.metrics["cache_hits"] = MetricSeries("cache_hits")
        self.metrics["cache_misses"] = MetricSeries("cache_misses")
        self.metrics["ai_requests"] = MetricSeries("ai_requests")
        self.metrics["memory_usage"] = MetricSeries("memory_usage")
        self.metrics["cpu_usage"] = MetricSeries("cpu_usage")
    
    def start_monitoring(self) -> Any:
        """Start background monitoring thread"""
        if not self._monitoring_active:
            self._monitoring_active = True
            self._monitoring_thread = threading.Thread(
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                target=self._monitoring_loop,
                daemon=True
            )
            self._monitoring_thread.start()
            logger.info("✓ Monitoring started")
    
    def stop_monitoring(self) -> Any:
        """Stop background monitoring"""
        self._monitoring_active = False
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=5)
        logger.info("✓ Monitoring stopped")
    
    def _monitoring_loop(self) -> Any:
        """Background monitoring loop"""
        while self._monitoring_active:
            try:
                self._collect_system_metrics()
                time.sleep(30)  # Collect every 30 seconds
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                time.sleep(60)  # Wait longer on error
    
    def _collect_system_metrics(self) -> Any:
        """Collect system performance metrics"""
        try:
            # Memory usage
            memory = psutil.virtual_memory()
            self.record_gauge("memory_usage_percent", memory.percent)
            self.record_gauge("memory_usage_bytes", memory.used)
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.record_gauge("cpu_usage_percent", cpu_percent)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            self.record_gauge("disk_usage_percent", disk.percent)
            
            # Process-specific metrics
            process = psutil.Process()
            self.record_gauge("process_memory_rss", process.memory_info().rss)
            self.record_gauge("process_memory_vms", process.memory_info().vms)
            self.record_gauge("process_cpu_percent", process.cpu_percent())
            
            # Check alert thresholds
            self._check_alerts()
            
        except Exception as e:
            logger.warning(f"System metrics collection error: {e}")
    
    def record_counter(self, name: str, value: float = 1.0, labels: Optional[Dict[str, str]] = None):
        """Record a counter metric"""
        self.counters[name] += value
        
        if name not in self.metrics:
            self.metrics[name] = MetricSeries(name)
        
        self.metrics[name].add_point(self.counters[name], labels)
    
    def record_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Record a gauge metric"""
        self.gauges[name] = value
        
        if name not in self.metrics:
            self.metrics[name] = MetricSeries(name)
        
        self.metrics[name].add_point(value, labels)
    
    def record_histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Record a histogram metric"""
        self.histograms[name].append(value)
        
        # Keep only recent values (last 1000)
        if len(self.histograms[name]) > 1000:
            self.histograms[name] = self.histograms[name][-1000:]
        
        if name not in self.metrics:
            self.metrics[name] = MetricSeries(name)
        
        self.metrics[name].add_point(value, labels)
    
    def record_request(self, method: str, endpoint: str, status_code: int, 
                      duration: float, error: Optional[str] = None):
        """Record HTTP request metrics"""
        labels = {
            "method": method,
            "endpoint": endpoint,
            "status_code": str(status_code)
        }
        
        # Request count
        self.record_counter("requests_total", labels=labels)
        
        # Request duration
        self.record_histogram("request_duration", duration, labels=labels)
        self.request_times.append(duration)
        
        # Success/error tracking
        if 200 <= status_code < 400:
            self.record_counter("requests_success", labels=labels)
            self.success_counts[endpoint] += 1
        else:
            self.record_counter("requests_error", labels=labels)
            self.error_counts[endpoint] += 1
            
            if error:
                self.record_counter("errors_by_type", labels={**labels, "error": error})
    
    def record_ai_request(self, provider: str, model: str, tokens: int, 
                         duration: float, success: bool):
        """Record AI service request metrics"""
        labels = {
            "provider": provider,
            "model": model,
            "success": str(success)
        }
        
        self.record_counter("ai_requests_total", labels=labels)
        self.record_histogram("ai_request_duration", duration, labels=labels)
        self.record_histogram("ai_tokens_used", tokens, labels=labels)
        
        if success:
            self.record_counter("ai_requests_success", labels=labels)
        else:
            self.record_counter("ai_requests_error", labels=labels)
    
    def record_cache_operation(self, operation: str, hit: bool, duration: float):
        """Record cache operation metrics"""
        labels = {"operation": operation}
        
        if hit:
            self.record_counter("cache_hits", labels=labels)
        else:
            self.record_counter("cache_misses", labels=labels)
        
        self.record_histogram("cache_operation_duration", duration, labels=labels)
    
    @contextmanager
    def time_operation(self, operation_name: str, labels: Optional[Dict[str, str]] = None):
        """Context manager for timing operations"""
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.record_histogram(f"{operation_name}_duration", duration, labels)
    
    def get_uptime(self) -> float:
        """Get service uptime in seconds"""
        return time.time() - self.start_time
    
    async def get_request_rate(self, seconds: int = 300) -> float:
        """Get request rate (requests per second) over the last N seconds"""
        if "requests_total" not in self.metrics:
            return 0.0
        
        points = self.metrics["requests_total"].get_recent_points(seconds)
        if len(points) < 2:
            return 0.0
        
        # Calculate rate based on first and last points
        first_point = points[0]
        last_point = points[-1]
        time_diff = last_point.timestamp - first_point.timestamp
        value_diff = last_point.value - first_point.value
        
        return value_diff / time_diff if time_diff > 0 else 0.0
    
    def get_error_rate(self, seconds: int = 300) -> float:
        """Get error rate (errors per second) over the last N seconds"""
        if "requests_error" not in self.metrics:
            return 0.0
        
        points = self.metrics["requests_error"].get_recent_points(seconds)
        if len(points) < 2:
            return 0.0
        
        first_point = points[0]
        last_point = points[-1]
        time_diff = last_point.timestamp - first_point.timestamp
        value_diff = last_point.value - first_point.value
        
        return value_diff / time_diff if time_diff > 0 else 0.0
    
    def get_average_response_time(self, seconds: int = 300) -> float:
        """Get average response time over the last N seconds"""
        if "request_duration" not in self.metrics:
            return 0.0
        
        return self.metrics["request_duration"].get_average(seconds)
    
    def get_cache_hit_rate(self, seconds: int = 300) -> float:
        """Get cache hit rate over the last N seconds"""
        hits_metric = self.metrics.get("cache_hits")
        misses_metric = self.metrics.get("cache_misses")
        
        if not hits_metric or not misses_metric:
            return 0.0
        
        hits = hits_metric.get_recent_points(seconds)
        misses = misses_metric.get_recent_points(seconds)
        
        total_hits = sum(p.value for p in hits)
        total_misses = sum(p.value for p in misses)
        total_requests = total_hits + total_misses
        
        return (total_hits / total_requests * 100) if total_requests > 0 else 0.0
    
    def set_alert_threshold(self, metric_name: str, threshold: float, 
                          comparison: str = "greater", callback: Optional[Callable] = None):
        """Set alert threshold for a metric"""
        self.alert_thresholds[metric_name] = {
            "threshold": threshold,
            "comparison": comparison,  # "greater", "less", "equal"
            "callback": callback
        }
    
    def add_alert_callback(self, callback: Callable):
        """Add global alert callback"""
        self.alert_callbacks.append(callback)
    
    def _check_alerts(self) -> Any:
        """Check alert thresholds and trigger callbacks"""
        for metric_name, alert_config in self.alert_thresholds.items():
            if metric_name in self.gauges:
                current_value = self.gauges[metric_name]
                threshold = alert_config["threshold"]
                comparison = alert_config["comparison"]
                
                triggered = False
                if comparison == "greater" and current_value > threshold:
                    triggered = True
                elif comparison == "less" and current_value < threshold:
                    triggered = True
                elif comparison == "equal" and current_value == threshold:
                    triggered = True
                
                if triggered:
                    alert_data = {
                        "metric": metric_name,
                        "value": current_value,
                        "threshold": threshold,
                        "comparison": comparison,
                        "timestamp": time.time()
                    }
                    
                    # Call specific callback
                    if alert_config.get("callback"):
                        try:
                            alert_config["callback"](alert_data)
                        except Exception as e:
                            logger.error(f"Alert callback error: {e}")
                    
                    # Call global callbacks
                    for callback in self.alert_callbacks:
                        try:
                            callback(alert_data)
                        except Exception as e:
                            logger.error(f"Global alert callback error: {e}")
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary"""
        return {
            "uptime": self.get_uptime(),
            "request_rate": self.get_request_rate(),
            "error_rate": self.get_error_rate(),
            "average_response_time": self.get_average_response_time(),
            "cache_hit_rate": self.get_cache_hit_rate(),
            "counters": dict(self.counters),
            "gauges": dict(self.gauges),
            "histogram_summaries": {
                name: {
                    "count": len(values),
                    "avg": sum(values) / len(values) if values else 0,
                    "min": min(values) if values else 0,
                    "max": max(values) if values else 0
                }
                for name, values in self.histograms.items()
            }
        }
    
    def get_prometheus_metrics(self) -> str:
        """Export metrics in Prometheus format"""
        lines = []
        
        # Counters
        for name, value in self.counters.items():
            lines.append(f"# TYPE {name} counter")
            lines.append(f"{name} {value}")
        
        # Gauges
        for name, value in self.gauges.items():
            lines.append(f"# TYPE {name} gauge")
            lines.append(f"{name} {value}")
        
        # Histograms
        for name, values in self.histograms.items():
            if values:
                lines.append(f"# TYPE {name} histogram")
                lines.append(f"{name}_count {len(values)}")
                lines.append(f"{name}_sum {sum(values)}")
                
                # Simple percentiles
                sorted_values = sorted(values)
                p50 = sorted_values[int(len(sorted_values) * 0.5)]
                p95 = sorted_values[int(len(sorted_values) * 0.95)]
                p99 = sorted_values[int(len(sorted_values) * 0.99)]
                
                lines.append(f"{name}_bucket{{le=\"{p50}\"}} {int(len(sorted_values) * 0.5)}")
                lines.append(f"{name}_bucket{{le=\"{p95}\"}} {int(len(sorted_values) * 0.95)}")
                lines.append(f"{name}_bucket{{le=\"{p99}\"}} {int(len(sorted_values) * 0.99)}")
                lines.append(f"{name}_bucket{{le=\"+Inf\"}} {len(sorted_values)}")
        
        return "\n".join(lines)
    
    def reset_metrics(self) -> Any:
        """Reset all metrics"""
        self.metrics.clear()
        self.counters.clear()
        self.gauges.clear()
        self.histograms.clear()
        self.request_times.clear()
        self.error_counts.clear()
        self.success_counts.clear()
        self._setup_default_metrics()
        logger.info("✓ Metrics reset")


# Global metrics collector instance
metrics_collector = MetricsCollector()


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector"""
    return metrics_collector


# Decorator for automatic request monitoring
async def monitor_requests(func) -> Any:
    """Decorator to automatically monitor API requests"""
    def wrapper(*args, **kwargs) -> Any:
        start_time = time.time()
        method = kwargs.get('method', 'UNKNOWN')
        endpoint = kwargs.get('endpoint', func.__name__)
        
        try:
            result = func(*args, **kwargs)
            status_code = getattr(result, 'status_code', 200)
            duration = time.time() - start_time
            
            metrics_collector.record_request(
                method=method,
                endpoint=endpoint,
                status_code=status_code,
                duration=duration
            )
            
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            metrics_collector.record_request(
                method=method,
                endpoint=endpoint,
                status_code=500,
                duration=duration,
                error=str(e)
            )
            raise
    
    return wrapper 