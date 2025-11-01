"""
Profiling utilities for KV Cache.

Provides performance profiling and analysis.
"""
import logging
import time
from typing import Dict, Any, Optional, Callable
from contextlib import contextmanager
from collections import defaultdict
import torch

logger = logging.getLogger(__name__)


class CacheProfiler:
    """
    Profiles cache operations for performance analysis.
    
    Tracks timing, memory usage, and operation counts.
    """
    
    def __init__(self, enabled: bool = True):
        """
        Initialize profiler.
        
        Args:
            enabled: Whether profiling is enabled
        """
        self.enabled = enabled
        self._timings: Dict[str, list] = defaultdict(list)
        self._memory_usage: Dict[str, list] = defaultdict(list)
        self._operation_counts: Dict[str, int] = defaultdict(int)
        self._start_times: Dict[str, float] = {}
    
    @contextmanager
    def profile_operation(self, operation_name: str):
        """
        Context manager for profiling an operation.
        
        Args:
            operation_name: Name of operation to profile
            
        Example:
            with profiler.profile_operation("put"):
                cache.put(position, key, value)
        """
        if not self.enabled:
            yield
            return
        
        start_time = time.perf_counter()
        start_memory = self._get_memory_usage()
        
        try:
            yield
        finally:
            end_time = time.perf_counter()
            end_memory = self._get_memory_usage()
            
            duration = end_time - start_time
            memory_delta = end_memory - start_memory
            
            self._timings[operation_name].append(duration)
            self._memory_usage[operation_name].append(memory_delta)
            self._operation_counts[operation_name] += 1
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024**2)
        return 0.0
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get profiling statistics.
        
        Returns:
            Dictionary with profiling stats
        """
        stats = {}
        
        for op_name in self._timings:
            timings = self._timings[op_name]
            memory = self._memory_usage[op_name]
            
            stats[op_name] = {
                "count": self._operation_counts[op_name],
                "total_time": sum(timings),
                "avg_time": sum(timings) / len(timings) if timings else 0.0,
                "min_time": min(timings) if timings else 0.0,
                "max_time": max(timings) if timings else 0.0,
                "avg_memory_delta_mb": sum(memory) / len(memory) if memory else 0.0,
            }
        
        return stats
    
    def print_stats(self) -> None:
        """Print profiling statistics."""
        if not self.enabled:
            logger.info("Profiling is disabled")
            return
        
        stats = self.get_stats()
        
        logger.info("=" * 60)
        logger.info("Cache Profiling Statistics")
        logger.info("=" * 60)
        
        for op_name, op_stats in stats.items():
            logger.info(f"\n{op_name}:")
            logger.info(f"  Count: {op_stats['count']}")
            logger.info(f"  Total time: {op_stats['total_time']:.4f}s")
            logger.info(f"  Avg time: {op_stats['avg_time']:.4f}s")
            logger.info(f"  Min/Max: {op_stats['min_time']:.4f}s / {op_stats['max_time']:.4f}s")
            logger.info(f"  Avg memory delta: {op_stats['avg_memory_delta_mb']:.2f} MB")
        
        logger.info("=" * 60)
    
    def reset(self) -> None:
        """Reset profiling data."""
        self._timings.clear()
        self._memory_usage.clear()
        self._operation_counts.clear()
        self._start_times.clear()
    
    def enable(self) -> None:
        """Enable profiling."""
        self.enabled = True
    
    def disable(self) -> None:
        """Disable profiling."""
        self.enabled = False


def profile_function(func: Callable) -> Callable:
    """
    Decorator to profile a function.
    
    Args:
        func: Function to profile
        
    Returns:
        Wrapped function with profiling
    """
    def wrapper(*args, **kwargs):
        profiler = getattr(args[0], '_profiler', None) if args else None
        if profiler is None:
            return func(*args, **kwargs)
        
        func_name = func.__name__
        with profiler.profile_operation(func_name):
            return func(*args, **kwargs)
    
    return wrapper

