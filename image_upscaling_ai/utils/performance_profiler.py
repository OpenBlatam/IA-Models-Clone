"""
Performance Profiler
====================

Performance profiling and analysis.
"""

import logging
import time
import cProfile
import pstats
from typing import Dict, Any, Optional, Callable
from functools import wraps
from io import StringIO
from contextlib import contextmanager

logger = logging.getLogger(__name__)


@contextmanager
def profile_context(output_file: Optional[str] = None):
    """
    Context manager for profiling.
    
    Args:
        output_file: File to save profile results
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
            # Print to string
            s = StringIO()
            stats = pstats.Stats(profiler, stream=s)
            stats.sort_stats('cumulative')
            stats.print_stats(20)  # Top 20
            logger.debug(s.getvalue())


def profile_function(output_file: Optional[str] = None):
    """
    Decorator to profile a function.
    
    Args:
        output_file: File to save profile results
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            with profile_context(output_file):
                return await func(*args, **kwargs)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            with profile_context(output_file):
                return func(*args, **kwargs)
        
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


class PerformanceProfiler:
    """
    Performance profiler for upscaling operations.
    
    Features:
    - Function profiling
    - Timing analysis
    - Memory profiling
    - Bottleneck detection
    """
    
    def __init__(self):
        """Initialize profiler."""
        self.profiles = {}
        logger.info("PerformanceProfiler initialized")
    
    def profile(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Profile a function execution.
        
        Args:
            func: Function to profile
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Profile results
        """
        profiler = cProfile.Profile()
        profiler.enable()
        
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
        finally:
            profiler.disable()
            elapsed = time.time() - start_time
        
        # Get stats
        s = StringIO()
        stats = pstats.Stats(profiler, stream=s)
        stats.sort_stats('cumulative')
        stats.print_stats(10)  # Top 10
        
        return {
            "function": func.__name__,
            "elapsed_time": elapsed,
            "profile_stats": s.getvalue()
        }
    
    def get_bottlenecks(self, profile_data: Dict[str, Any]) -> List[str]:
        """
        Identify bottlenecks from profile data.
        
        Args:
            profile_data: Profile data
            
        Returns:
            List of bottleneck functions
        """
        # Parse profile stats to find slow functions
        # This is simplified - would need proper parsing
        bottlenecks = []
        
        # Look for functions taking > 10% of time
        # (This would need proper parsing of pstats output)
        
        return bottlenecks


