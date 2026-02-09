"""
Metrics and monitoring utilities for professional documents module.

Functions for collecting and tracking metrics.
"""

import time
from typing import Dict, Any, Optional
from collections import defaultdict, deque
from datetime import datetime, timedelta


class MetricsCollector:
    """Collector for application metrics."""
    
    def __init__(self):
        """Initialize metrics collector."""
        self._counters: Dict[str, int] = defaultdict(int)
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self._timestamps: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
    
    def increment(self, metric_name: str, value: int = 1) -> None:
        """
        Increment a counter metric.
        
        Args:
            metric_name: Name of the metric
            value: Value to increment by
        """
        self._counters[metric_name] += value
        self._record_timestamp(metric_name)
    
    def decrement(self, metric_name: str, value: int = 1) -> None:
        """
        Decrement a counter metric.
        
        Args:
            metric_name: Name of the metric
            value: Value to decrement by
        """
        self._counters[metric_name] -= value
        self._record_timestamp(metric_name)
    
    def set_gauge(self, metric_name: str, value: float) -> None:
        """
        Set a gauge metric value.
        
        Args:
            metric_name: Name of the metric
            value: Gauge value
        """
        self._gauges[metric_name] = value
        self._record_timestamp(metric_name)
    
    def record_duration(self, metric_name: str, duration: float) -> None:
        """
        Record a duration in a histogram.
        
        Args:
            metric_name: Name of the metric
            duration: Duration in seconds
        """
        self._histograms[metric_name].append(duration)
        self._record_timestamp(metric_name)
    
    def get_counter(self, metric_name: str) -> int:
        """Get counter value."""
        return self._counters.get(metric_name, 0)
    
    def get_gauge(self, metric_name: str) -> Optional[float]:
        """Get gauge value."""
        return self._gauges.get(metric_name)
    
    def get_histogram_stats(self, metric_name: str) -> Dict[str, float]:
        """
        Get statistics for a histogram metric.
        
        Args:
            metric_name: Name of the metric
            
        Returns:
            Dictionary with min, max, avg, count
        """
        values = list(self._histograms.get(metric_name, deque()))
        if not values:
            return {"min": 0.0, "max": 0.0, "avg": 0.0, "count": 0}
        
        return {
            "min": min(values),
            "max": max(values),
            "avg": sum(values) / len(values),
            "count": len(values)
        }
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """
        Get all metrics as a dictionary.
        
        Returns:
            Dictionary with all metrics
        """
        metrics = {
            "counters": dict(self._counters),
            "gauges": dict(self._gauges),
            "histograms": {
                name: self.get_histogram_stats(name)
                for name in self._histograms.keys()
            }
        }
        return metrics
    
    def reset(self, metric_name: Optional[str] = None) -> None:
        """
        Reset metrics for a specific metric or all metrics.
        
        Args:
            metric_name: Metric to reset, or None for all
        """
        if metric_name:
            self._counters.pop(metric_name, None)
            self._gauges.pop(metric_name, None)
            self._histograms.pop(metric_name, None)
            self._timestamps.pop(metric_name, None)
        else:
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()
            self._timestamps.clear()
    
    def _record_timestamp(self, metric_name: str) -> None:
        """Record timestamp for a metric update."""
        self._timestamps[metric_name].append(time.time())
    
    def get_rate(self, metric_name: str, window_seconds: float = 60.0) -> float:
        """
        Get rate of metric updates per second.
        
        Args:
            metric_name: Name of the metric
            window_seconds: Time window in seconds
            
        Returns:
            Rate per second
        """
        timestamps = self._timestamps.get(metric_name, deque())
        if not timestamps:
            return 0.0
        
        now = time.time()
        cutoff = now - window_seconds
        
        recent = [ts for ts in timestamps if ts >= cutoff]
        return len(recent) / window_seconds if recent else 0.0


# Global metrics collector instance
metrics = MetricsCollector()


class PerformanceMetrics:
    """Context manager for tracking performance metrics."""
    
    def __init__(self, operation_name: str, metrics_collector: Optional[MetricsCollector] = None):
        """
        Initialize performance metrics tracker.
        
        Args:
            operation_name: Name of the operation
            metrics_collector: Optional metrics collector (uses global if None)
        """
        self.operation_name = operation_name
        self.metrics = metrics_collector or metrics
        self.start_time: Optional[float] = None
    
    def __enter__(self) -> "PerformanceMetrics":
        """Start tracking."""
        self.start_time = time.time()
        self.metrics.increment(f"{self.operation_name}.count")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop tracking and record duration."""
        if self.start_time:
            duration = time.time() - self.start_time
            self.metrics.record_duration(f"{self.operation_name}.duration", duration)
            
            if exc_type:
                self.metrics.increment(f"{self.operation_name}.errors")
            else:
                self.metrics.increment(f"{self.operation_name}.success")
        
        return False






