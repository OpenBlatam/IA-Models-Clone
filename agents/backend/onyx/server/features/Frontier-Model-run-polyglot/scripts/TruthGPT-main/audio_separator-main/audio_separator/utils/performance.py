"""
Performance monitoring and optimization utilities.
"""

import time
from functools import wraps
from typing import Callable, Any, Dict
from contextlib import contextmanager

from ..logger import logger


def timeit(func: Callable) -> Callable:
    """
    Decorator to measure function execution time.
    
    Usage:
        @timeit
        def my_function():
            ...
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time
        logger.info(f"{func.__name__} took {elapsed:.2f} seconds")
        return result
    return wrapper


@contextmanager
def timer(operation_name: str = "Operation"):
    """
    Context manager to measure execution time.
    
    Usage:
        with timer("Processing audio"):
            process_audio()
    """
    start_time = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - start_time
        logger.info(f"{operation_name} took {elapsed:.2f} seconds")


class PerformanceMonitor:
    """Monitor performance metrics."""
    
    def __init__(self):
        """Initialize performance monitor."""
        self.metrics: Dict[str, list] = {}
        self.current_operation: Dict[str, float] = {}
    
    def start(self, operation: str):
        """Start timing an operation."""
        self.current_operation[operation] = time.time()
    
    def stop(self, operation: str) -> float:
        """Stop timing an operation and return elapsed time."""
        if operation not in self.current_operation:
            logger.warning(f"Operation '{operation}' was not started")
            return 0.0
        
        elapsed = time.time() - self.current_operation[operation]
        del self.current_operation[operation]
        
        if operation not in self.metrics:
            self.metrics[operation] = []
        self.metrics[operation].append(elapsed)
        
        return elapsed
    
    def get_stats(self, operation: str) -> Dict[str, float]:
        """
        Get statistics for an operation.
        
        Args:
            operation: Operation name
            
        Returns:
            Dictionary with statistics (mean, min, max, count)
        """
        if operation not in self.metrics or not self.metrics[operation]:
            return {}
        
        times = self.metrics[operation]
        return {
            "mean": sum(times) / len(times),
            "min": min(times),
            "max": max(times),
            "count": len(times),
            "total": sum(times)
        }
    
    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all operations."""
        return {op: self.get_stats(op) for op in self.metrics.keys()}
    
    def reset(self):
        """Reset all metrics."""
        self.metrics.clear()
        self.current_operation.clear()
    
    def print_summary(self):
        """Print summary of all metrics."""
        logger.info("Performance Summary:")
        for operation, stats in self.get_all_stats().items():
            logger.info(
                f"  {operation}: "
                f"mean={stats['mean']:.2f}s, "
                f"min={stats['min']:.2f}s, "
                f"max={stats['max']:.2f}s, "
                f"count={stats['count']}"
            )


# Global performance monitor instance
performance_monitor = PerformanceMonitor()


def profile_memory(func: Callable) -> Callable:
    """
    Decorator to profile memory usage.
    
    Requires psutil to be installed.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            mem_before = process.memory_info().rss / 1024 / 1024  # MB
            
            result = func(*args, **kwargs)
            
            mem_after = process.memory_info().rss / 1024 / 1024  # MB
            mem_used = mem_after - mem_before
            
            logger.info(
                f"{func.__name__} memory: "
                f"{mem_before:.2f} MB -> {mem_after:.2f} MB "
                f"(delta: {mem_used:+.2f} MB)"
            )
            
            return result
        except ImportError:
            logger.warning("psutil not available, skipping memory profiling")
            return func(*args, **kwargs)
    
    return wrapper

