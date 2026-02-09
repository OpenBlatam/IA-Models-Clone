"""
Profiler for Piel Mejorador AI SAM3
===================================

Performance profiling and analysis.
"""

import asyncio
import time
import cProfile
import pstats
import io
import logging
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, field
from contextlib import contextmanager
from functools import wraps

logger = logging.getLogger(__name__)


@dataclass
class ProfileResult:
    """Profile result."""
    function_name: str
    total_time: float
    call_count: int
    cumulative_time: float
    per_call_time: float


class Profiler:
    """
    Performance profiler.
    
    Features:
    - Function profiling
    - Context manager support
    - Decorator support
    - Statistics aggregation
    """
    
    def __init__(self):
        """Initialize profiler."""
        self._profiler: Optional[cProfile.Profile] = None
        self._results: List[ProfileResult] = []
        self._active_profiles: Dict[str, float] = {}
    
    @contextmanager
    def profile(self, name: Optional[str] = None):
        """Context manager for profiling."""
        if self._profiler is None:
            self._profiler = cProfile.Profile()
        
        self._profiler.enable()
        start_time = time.time()
        
        try:
            yield
        finally:
            self._profiler.disable()
            elapsed = time.time() - start_time
            
            if name:
                self._active_profiles[name] = elapsed
    
    def profile_function(self, func: Callable):
        """Decorator for profiling functions."""
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            with self.profile(func.__name__):
                return await func(*args, **kwargs)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            with self.profile(func.__name__):
                return func(*args, **kwargs)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    def get_stats(self, sort_by: str = "cumulative") -> List[ProfileResult]:
        """
        Get profiling statistics.
        
        Args:
            sort_by: Sort key (cumulative, time, calls)
            
        Returns:
            List of profile results
        """
        if not self._profiler:
            return []
        
        stream = io.StringIO()
        stats = pstats.Stats(self._profiler, stream=stream)
        stats.sort_stats(sort_by)
        stats.print_stats()
        
        # Parse results (simplified)
        results = []
        for func_name, (call_count, total_time, cumulative_time, _) in stats.stats.items():
            per_call = cumulative_time / call_count if call_count > 0 else 0
            results.append(ProfileResult(
                function_name=f"{func_name[0]}:{func_name[1]}({func_name[2]})",
                total_time=total_time,
                call_count=call_count,
                cumulative_time=cumulative_time,
                per_call_time=per_call
            ))
        
        return sorted(results, key=lambda x: getattr(x, sort_by), reverse=True)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get profiling summary."""
        stats = self.get_stats()
        
        if not stats:
            return {
                "total_functions": 0,
                "total_time": 0,
                "total_calls": 0,
            }
        
        return {
            "total_functions": len(stats),
            "total_time": sum(s.cumulative_time for s in stats),
            "total_calls": sum(s.call_count for s in stats),
            "top_functions": [
                {
                    "name": s.function_name,
                    "time": s.cumulative_time,
                    "calls": s.call_count,
                }
                for s in stats[:10]
            ],
            "active_profiles": self._active_profiles,
        }
    
    def reset(self):
        """Reset profiler."""
        self._profiler = None
        self._results.clear()
        self._active_profiles.clear()


# Global profiler instance
_global_profiler = Profiler()


def profile_function(func: Callable):
    """Global decorator for profiling."""
    return _global_profiler.profile_function(func)


def get_profiler() -> Profiler:
    """Get global profiler instance."""
    return _global_profiler

