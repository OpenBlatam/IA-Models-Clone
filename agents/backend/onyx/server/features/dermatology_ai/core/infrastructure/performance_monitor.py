"""
Performance Monitoring
Decorators and utilities for tracking performance metrics
"""

import time
import functools
from typing import Callable, Any, Dict, Optional
import logging
from collections import defaultdict
import asyncio

logger = logging.getLogger(__name__)


class PerformanceMetrics:
    """Track performance metrics"""
    
    def __init__(self):
        self.metrics: Dict[str, list] = defaultdict(list)
        self.counts: Dict[str, int] = defaultdict(int)
        self.errors: Dict[str, int] = defaultdict(int)
    
    def record(self, operation: str, duration: float, success: bool = True):
        """Record a performance metric"""
        self.metrics[operation].append(duration)
        self.counts[operation] += 1
        if not success:
            self.errors[operation] += 1
    
    def get_stats(self, operation: str) -> Dict[str, Any]:
        """Get statistics for an operation"""
        if operation not in self.metrics or not self.metrics[operation]:
            return {}
        
        durations = self.metrics[operation]
        return {
            "count": self.counts[operation],
            "errors": self.errors[operation],
            "min": min(durations),
            "max": max(durations),
            "avg": sum(durations) / len(durations),
            "total": sum(durations),
        }
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all operations"""
        return {
            operation: self.get_stats(operation)
            for operation in self.metrics.keys()
        }
    
    def reset(self):
        """Reset all metrics"""
        self.metrics.clear()
        self.counts.clear()
        self.errors.clear()


# Global metrics instance
_metrics = PerformanceMetrics()


def get_metrics() -> PerformanceMetrics:
    """Get global metrics instance"""
    return _metrics


def monitor_performance(
    operation_name: Optional[str] = None,
    log_threshold: Optional[float] = None,
    track_metrics: bool = True
):
    """
    Decorator to monitor function performance
    
    Args:
        operation_name: Name for the operation (defaults to function name)
        log_threshold: Log warning if duration exceeds this (seconds)
        track_metrics: Whether to track metrics
    """
    def decorator(func: Callable) -> Callable:
        op_name = operation_name or f"{func.__module__}.{func.__name__}"
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            success = True
            
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                raise
            finally:
                duration = time.time() - start_time
                
                if track_metrics:
                    _metrics.record(op_name, duration, success)
                
                if log_threshold and duration > log_threshold:
                    logger.warning(
                        f"Slow operation detected: {op_name} took {duration:.3f}s "
                        f"(threshold: {log_threshold}s)"
                    )
                else:
                    logger.debug(f"{op_name} completed in {duration:.3f}s")
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            success = True
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                raise
            finally:
                duration = time.time() - start_time
                
                if track_metrics:
                    _metrics.record(op_name, duration, success)
                
                if log_threshold and duration > log_threshold:
                    logger.warning(
                        f"Slow operation detected: {op_name} took {duration:.3f}s "
                        f"(threshold: {log_threshold}s)"
                    )
                else:
                    logger.debug(f"{op_name} completed in {duration:.3f}s")
        
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


@monitor_performance(operation_name="database.query", log_threshold=1.0)
async def monitored_query(query_func: Callable, *args, **kwargs):
    """Wrapper for database queries with performance monitoring"""
    return await query_func(*args, **kwargs)















