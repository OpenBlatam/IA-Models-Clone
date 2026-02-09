"""
Metrics Collector

Collects and aggregates system metrics.
"""

import logging
import time
from typing import Dict, Any, List
from datetime import datetime, timedelta
from collections import defaultdict
from threading import Lock

logger = logging.getLogger(__name__)


class MetricsCollector:
    """
    Collects and aggregates system metrics.
    
    Tracks inspection counts, performance metrics, and error rates.
    """
    
    def __init__(self):
        """Initialize metrics collector."""
        self._lock = Lock()
        self._counters = defaultdict(int)
        self._timers = defaultdict(list)
        self._gauges = {}
        self._errors = []
        self._start_time = datetime.utcnow()
    
    def increment(self, metric_name: str, value: int = 1):
        """
        Increment a counter metric.
        
        Args:
            metric_name: Name of the metric
            value: Value to increment by
        """
        with self._lock:
            self._counters[metric_name] += value
    
    def record_timing(self, metric_name: str, duration_ms: float):
        """
        Record a timing metric.
        
        Args:
            metric_name: Name of the metric
            duration_ms: Duration in milliseconds
        """
        with self._lock:
            self._timers[metric_name].append(duration_ms)
            # Keep only last 1000 timings
            if len(self._timers[metric_name]) > 1000:
                self._timers[metric_name] = self._timers[metric_name][-1000:]
    
    def set_gauge(self, metric_name: str, value: float):
        """
        Set a gauge metric.
        
        Args:
            metric_name: Name of the metric
            value: Gauge value
        """
        with self._lock:
            self._gauges[metric_name] = value
    
    def record_error(self, error_type: str, error_message: str):
        """
        Record an error.
        
        Args:
            error_type: Type of error
            error_message: Error message
        """
        with self._lock:
            self._errors.append({
                "type": error_type,
                "message": error_message,
                "timestamp": datetime.utcnow().isoformat(),
            })
            # Keep only last 100 errors
            if len(self._errors) > 100:
                self._errors = self._errors[-100:]
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get all metrics.
        
        Returns:
            Dictionary with all metrics
        """
        with self._lock:
            # Calculate timing statistics
            timing_stats = {}
            for metric_name, timings in self._timers.items():
                if timings:
                    timing_stats[metric_name] = {
                        "count": len(timings),
                        "min": min(timings),
                        "max": max(timings),
                        "avg": sum(timings) / len(timings),
                        "p50": self._percentile(timings, 50),
                        "p95": self._percentile(timings, 95),
                        "p99": self._percentile(timings, 99),
                    }
            
            # Calculate error rate
            total_requests = self._counters.get("inspections.total", 0)
            total_errors = len(self._errors)
            error_rate = (total_errors / total_requests * 100) if total_requests > 0 else 0.0
            
            # Uptime
            uptime_seconds = (datetime.utcnow() - self._start_time).total_seconds()
            
            return {
                "counters": dict(self._counters),
                "timings": timing_stats,
                "gauges": dict(self._gauges),
                "errors": {
                    "total": total_errors,
                    "recent": self._errors[-10:] if self._errors else [],
                    "error_rate": error_rate,
                },
                "uptime_seconds": uptime_seconds,
                "timestamp": datetime.utcnow().isoformat(),
            }
    
    def _percentile(self, data: List[float], percentile: int) -> float:
        """
        Calculate percentile.
        
        Args:
            data: List of values
            percentile: Percentile (0-100)
        
        Returns:
            Percentile value
        """
        if not data:
            return 0.0
        
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile / 100)
        return sorted_data[min(index, len(sorted_data) - 1)]
    
    def reset(self):
        """Reset all metrics."""
        with self._lock:
            self._counters.clear()
            self._timers.clear()
            self._gauges.clear()
            self._errors.clear()
            self._start_time = datetime.utcnow()


# Global metrics collector instance
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """
    Get global metrics collector instance.
    
    Returns:
        MetricsCollector instance
    """
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector



