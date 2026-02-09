"""
Monitoring Module
================

Sistema de monitoring y métricas.
"""

from .metrics import (
    MetricsCollector,
    Counter,
    Gauge,
    Histogram,
    Timer,
    get_metrics_collector
)

try:
    from .performance_monitor import (
        PerformanceMonitor,
        monitor_function,
        monitor_class_methods
    )
except ImportError:
    # Fallback if performance_monitor not available
    PerformanceMonitor = None
    def monitor_function(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    def monitor_class_methods(*args, **kwargs):
        def decorator(cls):
            return cls
        return decorator

def get_monitoring_system():
    """Get monitoring system instance."""
    return get_metrics_collector()

__all__ = [
    "MetricsCollector",
    "Counter",
    "Gauge",
    "Histogram",
    "Timer",
    "get_metrics_collector",
    "get_monitoring_system",
    "PerformanceMonitor",
    "monitor_function",
    "monitor_class_methods"
]

