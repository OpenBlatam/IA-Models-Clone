"""
Performance Utilities
=====================

Utilities for performance monitoring and optimization.
"""

import time
import functools
from typing import Callable, Dict, Any, Optional
from contextlib import contextmanager
from dataclasses import dataclass, field
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetric:
    """Performance metric data"""
    name: str
    duration: float
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class PerformanceMonitor:
    """
    Performance monitor for tracking function execution times.
    
    Features:
    - Track execution times
    - Aggregate statistics
    - Performance reports
    """
    
    def __init__(self):
        """Initialize performance monitor"""
        self.metrics: list[PerformanceMetric] = []
        self.max_metrics = 1000
    
    def record(self, name: str, duration: float, metadata: Dict[str, Any] = None):
        """
        Record a performance metric.
        
        Args:
            name: Metric name
            duration: Duration in seconds
            metadata: Optional metadata
        """
        metric = PerformanceMetric(
            name=name,
            duration=duration,
            timestamp=time.time(),
            metadata=metadata or {}
        )
        
        self.metrics.append(metric)
        
        # Keep only last max_metrics
        if len(self.metrics) > self.max_metrics:
            self.metrics = self.metrics[-self.max_metrics:]
    
    def get_stats(self, name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get performance statistics.
        
        Args:
            name: Optional metric name to filter by
            
        Returns:
            Dictionary with statistics
        """
        metrics = self.metrics
        if name:
            metrics = [m for m in metrics if m.name == name]
        
        if not metrics:
            return {
                "count": 0,
                "name": name or "all"
            }
        
        durations = [m.duration for m in metrics]
        
        return {
            "name": name or "all",
            "count": len(metrics),
            "min": min(durations),
            "max": max(durations),
            "avg": sum(durations) / len(durations),
            "total": sum(durations)
        }
    
    def clear(self):
        """Clear all metrics"""
        self.metrics.clear()


# Global performance monitor
_performance_monitor: Optional[PerformanceMonitor] = None


def get_performance_monitor() -> PerformanceMonitor:
    """Get or create performance monitor instance"""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()
    return _performance_monitor


@contextmanager
def measure_performance(name: str, metadata: Dict[str, Any] = None):
    """
    Context manager for measuring performance.
    
    Usage:
        with measure_performance("my_operation"):
            # code to measure
            pass
    """
    start_time = time.time()
    try:
        yield
    finally:
        duration = time.time() - start_time
        monitor = get_performance_monitor()
        monitor.record(name, duration, metadata)


def track_performance(name: Optional[str] = None) -> Callable:
    """
    Decorator to track function performance.
    
    Usage:
        @track_performance("my_function")
        async def my_function():
            ...
    """
    def decorator(func: Callable) -> Callable:
        metric_name = name or func.__name__
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            with measure_performance(metric_name):
                return await func(*args, **kwargs)
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            with measure_performance(metric_name):
                return func(*args, **kwargs)
        
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator

