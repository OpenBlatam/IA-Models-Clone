"""
Performance Profiling Tools
For identifying bottlenecks and optimizing code
"""

import time
import functools
import cProfile
import pstats
from typing import Callable, Any, Optional
from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)


@contextmanager
def profile_context(output_file: Optional[str] = None):
    """
    Context manager for profiling code blocks
    
    Usage:
        with profile_context("profile.stats"):
            # Code to profile
            pass
    """
    profiler = cProfile.Profile()
    profiler.enable()
    
    try:
        yield profiler
    finally:
        profiler.disable()
        
        if output_file:
            profiler.dump_stats(output_file)
            logger.info(f"Profile saved to {output_file}")
        else:
            # Print stats
            stats = pstats.Stats(profiler)
            stats.sort_stats('cumulative')
            stats.print_stats(20)


def profile_function(func: Callable) -> Callable:
    """
    Decorator to profile function execution
    
    Usage:
        @profile_function
        async def my_function():
            pass
    """
    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            return result
        finally:
            duration = time.time() - start_time
            logger.info(f"{func.__name__} took {duration:.4f}s")
    
    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            duration = time.time() - start_time
            logger.info(f"{func.__name__} took {duration:.4f}s")
    
    import asyncio
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    return sync_wrapper


class PerformanceMonitor:
    """
    Performance monitoring for tracking metrics
    """
    
    def __init__(self):
        self.metrics: dict = {}
    
    def record(self, operation: str, duration: float, metadata: Optional[dict] = None):
        """Record performance metric"""
        if operation not in self.metrics:
            self.metrics[operation] = {
                "count": 0,
                "total_duration": 0.0,
                "min_duration": float('inf'),
                "max_duration": 0.0,
                "metadata": []
            }
        
        metric = self.metrics[operation]
        metric["count"] += 1
        metric["total_duration"] += duration
        metric["min_duration"] = min(metric["min_duration"], duration)
        metric["max_duration"] = max(metric["max_duration"], duration)
        
        if metadata:
            metric["metadata"].append(metadata)
    
    def get_stats(self, operation: str) -> Optional[dict]:
        """Get statistics for operation"""
        if operation not in self.metrics:
            return None
        
        metric = self.metrics[operation]
        avg_duration = metric["total_duration"] / metric["count"]
        
        return {
            "operation": operation,
            "count": metric["count"],
            "avg_duration": avg_duration,
            "min_duration": metric["min_duration"],
            "max_duration": metric["max_duration"],
            "total_duration": metric["total_duration"]
        }
    
    def get_all_stats(self) -> dict:
        """Get statistics for all operations"""
        return {
            op: self.get_stats(op)
            for op in self.metrics.keys()
        }
    
    def reset(self):
        """Reset all metrics"""
        self.metrics.clear()


# Global performance monitor
_performance_monitor: Optional[PerformanceMonitor] = None


def get_performance_monitor() -> PerformanceMonitor:
    """Get or create global performance monitor"""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()
    return _performance_monitor















