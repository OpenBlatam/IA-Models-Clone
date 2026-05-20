"""
Performance utilities for API endpoints
"""

import time
import functools
from typing import Callable, Any
import logging

logger = logging.getLogger(__name__)


def measure_time(func: Callable) -> Callable:
    """Decorator to measure function execution time"""
    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        start = time.time()
        try:
            result = await func(*args, **kwargs)
            elapsed = time.time() - start
            logger.debug(f"{func.__name__} took {elapsed:.3f}s")
            return result
        except Exception as e:
            elapsed = time.time() - start
            logger.error(f"{func.__name__} failed after {elapsed:.3f}s: {e}")
            raise
    
    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        start = time.time()
        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start
            logger.debug(f"{func.__name__} took {elapsed:.3f}s")
            return result
        except Exception as e:
            elapsed = time.time() - start
            logger.error(f"{func.__name__} failed after {elapsed:.3f}s: {e}")
            raise
    
    import asyncio
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    return sync_wrapper


class PerformanceMonitor:
    """Monitor performance metrics"""
    
    def __init__(self):
        self.metrics = {}
    
    def record(self, operation: str, duration: float):
        """Record operation duration"""
        if operation not in self.metrics:
            self.metrics[operation] = {
                "count": 0,
                "total": 0.0,
                "min": float('inf'),
                "max": 0.0
            }
        
        metric = self.metrics[operation]
        metric["count"] += 1
        metric["total"] += duration
        metric["min"] = min(metric["min"], duration)
        metric["max"] = max(metric["max"], duration)
    
    def get_stats(self, operation: str) -> dict:
        """Get statistics for an operation"""
        if operation not in self.metrics:
            return None
        
        metric = self.metrics[operation]
        return {
            "count": metric["count"],
            "avg": metric["total"] / metric["count"],
            "min": metric["min"],
            "max": metric["max"],
            "total": metric["total"]
        }
    
    def get_all_stats(self) -> dict:
        """Get all statistics"""
        return {
            op: self.get_stats(op)
            for op in self.metrics.keys()
        }


performance_monitor = PerformanceMonitor()

