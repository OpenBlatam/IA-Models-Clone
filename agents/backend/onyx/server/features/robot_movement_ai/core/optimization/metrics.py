"""
Metrics for Trajectory Optimization
===================================
Metrics collection and tracking for trajectory optimization.
"""

from typing import Dict, Any, Optional
import time
from collections import defaultdict
import threading

# Thread-safe metrics storage
_metrics_lock = threading.Lock()
_metrics_data: Dict[str, Any] = defaultdict(list)
_counters: Dict[str, int] = defaultdict(int)
_timings: Dict[str, list] = defaultdict(list)


def record_timing(metric_name: str, duration: float):
    """Record a timing metric"""
    with _metrics_lock:
        _timings[metric_name].append(duration)


def increment_counter(metric_name: str, value: int = 1):
    """Increment a counter metric"""
    with _metrics_lock:
        _counters[metric_name] += value


def record_value(metric_name: str, value: float):
    """Record a value metric"""
    with _metrics_lock:
        _metrics_data[metric_name].append(value)


def get_metrics_collector():
    """Get a metrics collector instance"""
    return {
        'timings': dict(_timings),
        'counters': dict(_counters),
        'values': {k: v[-1] if v else None for k, v in _metrics_data.items()}
    }


def reset_metrics():
    """Reset all metrics"""
    with _metrics_lock:
        _metrics_data.clear()
        _counters.clear()
        _timings.clear()



