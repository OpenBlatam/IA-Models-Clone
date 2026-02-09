"""
Application metrics and performance monitoring
"""

import time
from typing import Dict, Any, Optional, List
from collections import defaultdict, deque
from datetime import datetime, timedelta
from threading import Lock
from ..core.logging_config import get_logger

logger = get_logger(__name__)


class MetricsCollector:
    """Collector for application metrics"""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self._metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history))
        self._counters: Dict[str, int] = defaultdict(int)
        self._timers: Dict[str, List[float]] = defaultdict(list)
        self._lock = Lock()
    
    def increment(self, metric_name: str, value: int = 1, tags: Optional[Dict[str, str]] = None):
        """Increment a counter metric"""
        with self._lock:
            key = self._build_key(metric_name, tags)
            self._counters[key] += value
            self._record_metric(key, self._counters[key])
    
    def gauge(self, metric_name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Set a gauge metric"""
        with self._lock:
            key = self._build_key(metric_name, tags)
            self._record_metric(key, value)
    
    def timer(self, metric_name: str, duration_ms: float, tags: Optional[Dict[str, str]] = None):
        """Record a timer metric"""
        with self._lock:
            key = self._build_key(metric_name, tags)
            if len(self._timers[key]) >= 100:  # Keep last 100 timings
                self._timers[key].pop(0)
            self._timers[key].append(duration_ms)
            self._record_metric(key, duration_ms)
    
    def _record_metric(self, key: str, value: float):
        """Record a metric value with timestamp"""
        self._metrics[key].append({
            "value": value,
            "timestamp": datetime.now().isoformat()
        })
    
    def _build_key(self, metric_name: str, tags: Optional[Dict[str, str]]) -> str:
        """Build metric key with tags"""
        if tags:
            tag_str = ",".join(f"{k}={v}" for k, v in sorted(tags.items()))
            return f"{metric_name}[{tag_str}]"
        return metric_name
    
    def get_metric(self, metric_name: str, tags: Optional[Dict[str, str]] = None) -> List[Dict[str, Any]]:
        """Get metric history"""
        key = self._build_key(metric_name, tags)
        with self._lock:
            return list(self._metrics[key])
    
    def get_counter(self, metric_name: str, tags: Optional[Dict[str, str]] = None) -> int:
        """Get current counter value"""
        key = self._build_key(metric_name, tags)
        with self._lock:
            return self._counters.get(key, 0)
    
    def get_timer_stats(self, metric_name: str, tags: Optional[Dict[str, str]] = None) -> Dict[str, float]:
        """Get timer statistics"""
        key = self._build_key(metric_name, tags)
        with self._lock:
            timings = self._timers.get(key, [])
            if not timings:
                return {}
            return {
                "count": len(timings),
                "min": min(timings),
                "max": max(timings),
                "avg": sum(timings) / len(timings),
                "p50": self._percentile(timings, 50),
                "p95": self._percentile(timings, 95),
                "p99": self._percentile(timings, 99)
            }
    
    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile"""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile / 100)
        return sorted_data[min(index, len(sorted_data) - 1)]
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all metrics summary"""
        with self._lock:
            return {
                "counters": dict(self._counters),
                "metrics": {k: list(v)[-10:] for k, v in self._metrics.items()},  # Last 10 values
                "timers": {k: self.get_timer_stats(k.split("[")[0]) for k in self._timers.keys()}
            }
    
    def reset(self):
        """Reset all metrics"""
        with self._lock:
            self._metrics.clear()
            self._counters.clear()
            self._timers.clear()


# Global metrics collector instance
_metrics_collector = MetricsCollector()


def get_metrics_collector() -> MetricsCollector:
    """Get global metrics collector"""
    return _metrics_collector


class MetricsContext:
    """Context manager for timing operations"""
    
    def __init__(self, metric_name: str, tags: Optional[Dict[str, str]] = None):
        self.metric_name = metric_name
        self.tags = tags
        self.start_time: Optional[float] = None
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration_ms = (time.perf_counter() - self.start_time) * 1000
            get_metrics_collector().timer(self.metric_name, duration_ms, self.tags)
        return False


def track_metric(metric_name: str, value: float, tags: Optional[Dict[str, str]] = None):
    """Track a metric value"""
    get_metrics_collector().gauge(metric_name, value, tags)


def increment_counter(metric_name: str, value: int = 1, tags: Optional[Dict[str, str]] = None):
    """Increment a counter"""
    get_metrics_collector().increment(metric_name, value, tags)


def time_operation(metric_name: str, tags: Optional[Dict[str, str]] = None):
    """Decorator/context manager for timing operations"""
    return MetricsContext(metric_name, tags)

