"""
Advanced Performance Profiler
==============================

Advanced performance profiling system with detailed analysis.
"""

import asyncio
import logging
import time
import cProfile
import pstats
import io
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime
from contextlib import contextmanager
from functools import wraps

logger = logging.getLogger(__name__)


@dataclass
class ProfileResult:
    """Profile result."""
    function_name: str
    total_time: float
    cumulative_time: float
    call_count: int
    per_call_time: float
    file_name: str
    line_number: int


@dataclass
class PerformanceProfile:
    """Performance profile."""
    name: str
    start_time: datetime
    end_time: datetime
    duration: float
    results: List[ProfileResult]
    total_calls: int
    total_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class AdvancedPerformanceProfiler:
    """Advanced performance profiler."""
    
    def __init__(self):
        """Initialize advanced performance profiler."""
        self.profiles: Dict[str, PerformanceProfile] = {}
        self.active_profiles: Dict[str, cProfile.Profile] = {}
    
    @contextmanager
    def profile(self, name: str):
        """
        Context manager for profiling.
        
        Args:
            name: Profile name
        """
        profiler = cProfile.Profile()
        profiler.enable()
        start_time = datetime.now()
        
        try:
            yield
        finally:
            profiler.disable()
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # Analyze profile
            s = io.StringIO()
            stats = pstats.Stats(profiler, stream=s)
            stats.sort_stats('cumulative')
            stats.print_stats()
            
            # Parse results
            results = self._parse_stats(stats)
            
            profile = PerformanceProfile(
                name=name,
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                results=results,
                total_calls=stats.total_calls,
                total_time=stats.total_tt
            )
            
            self.profiles[name] = profile
            logger.info(f"Profile completed: {name} ({duration:.3f}s)")
    
    def _parse_stats(self, stats: pstats.Stats) -> List[ProfileResult]:
        """Parse stats into ProfileResult list."""
        results = []
        
        for func_info, (cc, nc, tt, ct, callers) in stats.stats.items():
            file_name, line_number, function_name = func_info
            
            result = ProfileResult(
                function_name=function_name,
                total_time=tt,
                cumulative_time=ct,
                call_count=cc,
                per_call_time=tt / cc if cc > 0 else 0,
                file_name=file_name,
                line_number=line_number
            )
            results.append(result)
        
        return sorted(results, key=lambda r: r.cumulative_time, reverse=True)
    
    def profile_function(self, name: Optional[str] = None):
        """
        Decorator for profiling functions.
        
        Args:
            name: Optional profile name
        """
        def decorator(func: Callable):
            profile_name = name or f"{func.__module__}.{func.__name__}"
            
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                with self.profile(profile_name):
                    return await func(*args, **kwargs)
            
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                with self.profile(profile_name):
                    return func(*args, **kwargs)
            
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper
        
        return decorator
    
    def get_profile(self, name: str) -> Optional[PerformanceProfile]:
        """
        Get profile by name.
        
        Args:
            name: Profile name
            
        Returns:
            Performance profile or None
        """
        return self.profiles.get(name)
    
    def get_all_profiles(self) -> Dict[str, PerformanceProfile]:
        """
        Get all profiles.
        
        Returns:
            Dictionary of name -> profile
        """
        return self.profiles.copy()
    
    def get_top_functions(self, name: str, limit: int = 10) -> List[ProfileResult]:
        """
        Get top functions by cumulative time.
        
        Args:
            name: Profile name
            limit: Number of top functions
            
        Returns:
            List of profile results
        """
        profile = self.profiles.get(name)
        if not profile:
            return []
        
        return profile.results[:limit]
    
    def clear(self):
        """Clear all profiles."""
        self.profiles.clear()
        self.active_profiles.clear()



