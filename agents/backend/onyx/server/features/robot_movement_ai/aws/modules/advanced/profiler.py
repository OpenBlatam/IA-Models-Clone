"""
Advanced Profiler
=================

Advanced profiling and performance analysis.
"""

import logging
import time
import cProfile
import pstats
import io
from typing import Dict, Any, Optional, Callable
from functools import wraps
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class AdvancedProfiler:
    """Advanced profiler with detailed analysis."""
    
    def __init__(self):
        self._profiler: Optional[cProfile.Profile] = None
        self._profiles: Dict[str, cProfile.Profile] = {}
        self._function_stats: Dict[str, Dict[str, Any]] = {}
    
    def start_profiling(self, name: str = "default"):
        """Start profiling."""
        if name in self._profiles:
            logger.warning(f"Profile {name} already exists, stopping previous")
            self.stop_profiling(name)
        
        profiler = cProfile.Profile()
        profiler.enable()
        self._profiles[name] = profiler
        logger.info(f"Started profiling: {name}")
    
    def stop_profiling(self, name: str = "default") -> Optional[Dict[str, Any]]:
        """Stop profiling and get results."""
        if name not in self._profiles:
            return None
        
        profiler = self._profiles[name]
        profiler.disable()
        
        # Get stats
        stats_stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stats_stream)
        stats.sort_stats('cumulative')
        stats.print_stats(20)  # Top 20
        
        stats_output = stats_stream.getvalue()
        
        # Parse stats
        function_stats = self._parse_stats(stats)
        
        result = {
            "name": name,
            "stats_output": stats_output,
            "function_stats": function_stats,
            "total_calls": stats.total_calls,
            "total_time": stats.total_tt
        }
        
        self._function_stats[name] = function_stats
        
        logger.info(f"Stopped profiling: {name}")
        return result
    
    def _parse_stats(self, stats: pstats.Stats) -> Dict[str, Dict[str, Any]]:
        """Parse profiler stats."""
        function_stats = {}
        
        for func, (cc, nc, tt, ct, callers) in stats.stats.items():
            func_name = f"{func[0]}:{func[1]}:{func[2]}"
            function_stats[func_name] = {
                "call_count": cc,
                "primitive_calls": nc,
                "total_time": tt,
                "cumulative_time": ct,
                "time_per_call": tt / cc if cc > 0 else 0
            }
        
        return function_stats
    
    @contextmanager
    def profile_context(self, name: str = "default"):
        """Context manager for profiling."""
        self.start_profiling(name)
        try:
            yield
        finally:
            self.stop_profiling(name)
    
    def profile_function(self, func_name: Optional[str] = None):
        """Decorator to profile function."""
        def decorator(func: Callable):
            name = func_name or func.__name__
            
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                start = time.perf_counter()
                try:
                    result = await func(*args, **kwargs)
                    return result
                finally:
                    elapsed = time.perf_counter() - start
                    self._record_function_time(name, elapsed)
            
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                start = time.perf_counter()
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    elapsed = time.perf_counter() - start
                    self._record_function_time(name, elapsed)
            
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            return sync_wrapper
        
        return decorator
    
    def _record_function_time(self, func_name: str, elapsed: float):
        """Record function execution time."""
        if func_name not in self._function_stats:
            self._function_stats[func_name] = {
                "calls": 0,
                "total_time": 0.0,
                "min_time": float('inf'),
                "max_time": 0.0
            }
        
        stats = self._function_stats[func_name]
        stats["calls"] += 1
        stats["total_time"] += elapsed
        stats["min_time"] = min(stats["min_time"], elapsed)
        stats["max_time"] = max(stats["max_time"], elapsed)
        stats["avg_time"] = stats["total_time"] / stats["calls"]
    
    def get_function_stats(self, func_name: Optional[str] = None) -> Dict[str, Any]:
        """Get function statistics."""
        if func_name:
            return self._function_stats.get(func_name, {})
        return self._function_stats.copy()
    
    def get_slowest_functions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get slowest functions."""
        functions = []
        for name, stats in self._function_stats.items():
            functions.append({
                "name": name,
                "avg_time": stats.get("avg_time", 0),
                "total_time": stats.get("total_time", 0),
                "calls": stats.get("calls", 0)
            })
        
        functions.sort(key=lambda x: x["avg_time"], reverse=True)
        return functions[:limit]


# Import asyncio
import asyncio















