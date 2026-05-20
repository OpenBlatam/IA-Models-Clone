"""Metrics and Monitoring for Markdown to Professional Documents AI"""
from typing import Dict, Any
from datetime import datetime, timedelta
from collections import defaultdict
import threading
import time


class MetricsCollector:
    """Collect and track metrics for the service"""
    
    def __init__(self):
        self._lock = threading.Lock()
        self._counters = defaultdict(int)
        self._histograms = defaultdict(list)
        self._timers = defaultdict(list)
        self._start_time = datetime.now()
    
    def increment(self, metric_name: str, value: int = 1) -> None:
        """Increment a counter metric"""
        with self._lock:
            self._counters[metric_name] += value
    
    def record_value(self, metric_name: str, value: float) -> None:
        """Record a value for histogram"""
        with self._lock:
            self._histograms[metric_name].append(value)
            # Keep only last 1000 values
            if len(self._histograms[metric_name]) > 1000:
                self._histograms[metric_name] = self._histograms[metric_name][-1000:]
    
    def record_timing(self, metric_name: str, duration_ms: float) -> None:
        """Record a timing metric"""
        with self._lock:
            self._timers[metric_name].append(duration_ms)
            # Keep only last 1000 timings
            if len(self._timers[metric_name]) > 1000:
                self._timers[metric_name] = self._timers[metric_name][-1000:]
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get all metrics"""
        with self._lock:
            uptime = (datetime.now() - self._start_time).total_seconds()
            
            # Calculate histogram statistics
            histogram_stats = {}
            for name, values in self._histograms.items():
                if values:
                    histogram_stats[name] = {
                        "count": len(values),
                        "min": min(values),
                        "max": max(values),
                        "avg": sum(values) / len(values),
                        "p95": self._percentile(values, 95),
                        "p99": self._percentile(values, 99)
                    }
            
            # Calculate timer statistics
            timer_stats = {}
            for name, timings in self._timers.items():
                if timings:
                    timer_stats[name] = {
                        "count": len(timings),
                        "min_ms": min(timings),
                        "max_ms": max(timings),
                        "avg_ms": sum(timings) / len(timings),
                        "p95_ms": self._percentile(timings, 95),
                        "p99_ms": self._percentile(timings, 99)
                    }
            
            return {
                "uptime_seconds": uptime,
                "counters": dict(self._counters),
                "histograms": histogram_stats,
                "timers": timer_stats
            }
    
    def _percentile(self, values: list, percentile: int) -> float:
        """Calculate percentile"""
        if not values:
            return 0.0
        sorted_values = sorted(values)
        index = int(len(sorted_values) * percentile / 100)
        return sorted_values[min(index, len(sorted_values) - 1)]
    
    def reset(self) -> None:
        """Reset all metrics"""
        with self._lock:
            self._counters.clear()
            self._histograms.clear()
            self._timers.clear()
            self._start_time = datetime.now()


# Global metrics instance
_metrics_instance: MetricsCollector = None


def get_metrics() -> MetricsCollector:
    """Get global metrics instance"""
    global _metrics_instance
    if _metrics_instance is None:
        _metrics_instance = MetricsCollector()
    return _metrics_instance


class TimingContext:
    """Context manager for timing operations"""
    
    def __init__(self, metric_name: str):
        self.metric_name = metric_name
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration_ms = (time.time() - self.start_time) * 1000
        get_metrics().record_timing(self.metric_name, duration_ms)

