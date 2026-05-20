"""
Performance Profiler
===================

Utilities for performance profiling and analysis.
"""

import time
import cProfile
import pstats
import io
from typing import Dict, Any, Optional, Callable
from contextlib import contextmanager
from functools import wraps
import logging

logger = logging.getLogger(__name__)


class PerformanceProfiler:
    """Performance profiler for code analysis."""
    
    def __init__(self):
        """Initialize profiler."""
        self.profiles: Dict[str, cProfile.Profile] = {}
    
    @contextmanager
    def profile(self, name: str):
        """
        Profile a code block.
        
        Args:
            name: Profile name
            
        Usage:
            with profiler.profile("my_function"):
                # code to profile
        """
        profile = cProfile.Profile()
        profile.enable()
        
        try:
            yield
        finally:
            profile.disable()
            self.profiles[name] = profile
    
    def get_stats(self, name: str, sort_by: str = "cumulative") -> str:
        """
        Get profiling statistics.
        
        Args:
            name: Profile name
            sort_by: Sort key (cumulative, time, calls, etc.)
            
        Returns:
            Formatted statistics string
        """
        if name not in self.profiles:
            return f"Profile '{name}' not found"
        
        profile = self.profiles[name]
        stream = io.StringIO()
        stats = pstats.Stats(profile, stream=stream)
        stats.sort_stats(sort_by)
        stats.print_stats()
        
        return stream.getvalue()
    
    def get_top_functions(self, name: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get top functions by time.
        
        Args:
            name: Profile name
            limit: Number of top functions
            
        Returns:
            List of function statistics
        """
        if name not in self.profiles:
            return []
        
        profile = self.profiles[name]
        stats = pstats.Stats(profile)
        stats.sort_stats('cumulative')
        
        top_functions = []
        for func_info in stats.stats.items():
            func_name, (call_count, total_time, cumulative_time, callers) = func_info
            top_functions.append({
                "function": f"{func_name[0]}:{func_name[1]}({func_name[2]})",
                "call_count": call_count,
                "total_time": total_time,
                "cumulative_time": cumulative_time
            })
        
        return sorted(top_functions, key=lambda x: x["cumulative_time"], reverse=True)[:limit]
    
    def clear(self, name: Optional[str] = None):
        """
        Clear profile(s).
        
        Args:
            name: Profile name (None to clear all)
        """
        if name:
            self.profiles.pop(name, None)
        else:
            self.profiles.clear()


def profile_function(func: Callable) -> Callable:
    """
    Decorator to profile a function.
    
    Usage:
        @profile_function
        def my_function():
            ...
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        profiler = PerformanceProfiler()
        with profiler.profile(func.__name__):
            result = func(*args, **kwargs)
        
        # Log top functions
        top = profiler.get_top_functions(func.__name__, limit=5)
        if top:
            logger.debug(f"Top functions in {func.__name__}:")
            for func_info in top:
                logger.debug(f"  {func_info['function']}: {func_info['cumulative_time']:.4f}s")
        
        return result
    
    return wrapper


def profile_async_function(func: Callable) -> Callable:
    """
    Decorator to profile an async function.
    
    Usage:
        @profile_async_function
        async def my_function():
            ...
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        profiler = PerformanceProfiler()
        with profiler.profile(func.__name__):
            result = await func(*args, **kwargs)
        
        # Log top functions
        top = profiler.get_top_functions(func.__name__, limit=5)
        if top:
            logger.debug(f"Top functions in {func.__name__}:")
            for func_info in top:
                logger.debug(f"  {func_info['function']}: {func_info['cumulative_time']:.4f}s")
        
        return result
    
    return wrapper


class TimingContext:
    """Context manager for timing code blocks."""
    
    def __init__(self, name: str = "operation"):
        """
        Initialize timing context.
        
        Args:
            name: Operation name
        """
        self.name = name
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        """Start timing."""
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """End timing."""
        self.end_time = time.perf_counter()
        duration = self.end_time - self.start_time
        logger.info(f"{self.name} took {duration:.4f} seconds")
    
    @property
    def duration(self) -> Optional[float]:
        """Get duration in seconds."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None




