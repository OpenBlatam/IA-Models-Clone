"""
Performance utilities

Utilities for performance monitoring, profiling, and optimization.
"""

import time
import logging
from functools import wraps
from typing import Callable, Any, Optional
from contextlib import contextmanager

logger = logging.getLogger(__name__)


@contextmanager
def timer(operation_name: str = "Operation"):
    """
    Context manager for timing operations.
    
    Usage:
        with timer("Database query"):
            result = db.query(...)
    """
    start_time = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start_time
        logger.debug(f"{operation_name} took {elapsed:.4f}s")


def measure_time(func: Callable) -> Callable:
    """
    Decorator to measure function execution time.
    
    Usage:
        @measure_time
        def my_function():
            ...
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            elapsed = time.perf_counter() - start_time
            logger.info(f"{func.__name__} took {elapsed:.4f}s")
    
    return wrapper


class PerformanceMonitor:
    """
    Performance monitor for tracking operation metrics.
    """
    
    def __init__(self):
        self.metrics: dict[str, list[float]] = {}
    
    def record(self, operation: str, duration: float) -> None:
        """
        Record operation duration.
        
        Args:
            operation: Operation name
            duration: Duration in seconds
        """
        if operation not in self.metrics:
            self.metrics[operation] = []
        self.metrics[operation].append(duration)
    
    def get_stats(self, operation: str) -> Optional[dict[str, float]]:
        """
        Get statistics for an operation.
        
        Args:
            operation: Operation name
            
        Returns:
            Dictionary with stats or None if no data
        """
        if operation not in self.metrics or not self.metrics[operation]:
            return None
        
        durations = self.metrics[operation]
        return {
            "count": len(durations),
            "min": min(durations),
            "max": max(durations),
            "avg": sum(durations) / len(durations),
            "total": sum(durations)
        }
    
    def get_all_stats(self) -> dict[str, dict[str, float]]:
        """
        Get statistics for all operations.
        
        Returns:
            Dictionary mapping operation names to stats
        """
        return {
            operation: self.get_stats(operation)
            for operation in self.metrics.keys()
        }
    
    def reset(self) -> None:
        """Reset all metrics."""
        self.metrics.clear()


# Global performance monitor
_performance_monitor: Optional[PerformanceMonitor] = None


def get_performance_monitor() -> PerformanceMonitor:
    """Get global performance monitor instance."""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()
    return _performance_monitor
