"""
Profiler Service
Performance profiling and analysis
"""

from typing import Dict, Any, Optional, Callable
from functools import wraps
import time
import logging
import asyncio
from collections import defaultdict

logger = logging.getLogger(__name__)


class ProfilerService:
    """Performance profiling service"""
    
    def __init__(self):
        self.profiles: Dict[str, list] = defaultdict(list)
        self.enabled = True
    
    def profile(self, name: Optional[str] = None):
        """
        Decorator to profile function execution
        
        Args:
            name: Custom profile name (defaults to function name)
        """
        def decorator(func: Callable) -> Callable:
            profile_name = name or f"{func.__module__}.{func.__name__}"
            
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                if not self.enabled:
                    return await func(*args, **kwargs)
                
                start_time = time.perf_counter()
                try:
                    result = await func(*args, **kwargs)
                    return result
                finally:
                    elapsed = time.perf_counter() - start_time
                    self.record_profile(profile_name, elapsed)
            
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                if not self.enabled:
                    return func(*args, **kwargs)
                
                start_time = time.perf_counter()
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    elapsed = time.perf_counter() - start_time
                    self.record_profile(profile_name, elapsed)
            
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper
        
        return decorator
    
    def record_profile(self, name: str, elapsed: float):
        """Record profile data"""
        self.profiles[name].append(elapsed)
        
        # Keep only last 1000 measurements
        if len(self.profiles[name]) > 1000:
            self.profiles[name] = self.profiles[name][-1000:]
    
    def get_profile_stats(self, name: str) -> Optional[Dict[str, Any]]:
        """Get statistics for a profile"""
        if name not in self.profiles or not self.profiles[name]:
            return None
        
        times = self.profiles[name]
        
        return {
            "name": name,
            "count": len(times),
            "total": sum(times),
            "average": sum(times) / len(times),
            "min": min(times),
            "max": max(times),
            "p50": self._percentile(times, 50),
            "p95": self._percentile(times, 95),
            "p99": self._percentile(times, 99),
        }
    
    def get_all_profiles(self) -> Dict[str, Dict[str, Any]]:
        """Get all profile statistics"""
        return {
            name: self.get_profile_stats(name)
            for name in self.profiles.keys()
        }
    
    def _percentile(self, data: list, percentile: int) -> float:
        """Calculate percentile"""
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile / 100)
        return sorted_data[min(index, len(sorted_data) - 1)]
    
    def clear_profiles(self, name: Optional[str] = None):
        """Clear profile data"""
        if name:
            if name in self.profiles:
                del self.profiles[name]
        else:
            self.profiles.clear()


_profiler_service: Optional[ProfilerService] = None


def get_profiler_service() -> ProfilerService:
    """Get profiler service instance (singleton)"""
    global _profiler_service
    if _profiler_service is None:
        _profiler_service = ProfilerService()
    return _profiler_service

