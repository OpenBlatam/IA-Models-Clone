"""
Performance Profiler for Color Grading AI
==========================================

Advanced performance profiling and analysis.
"""

import logging
import time
import cProfile
import pstats
import io
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime
from contextlib import contextmanager
from enum import Enum

logger = logging.getLogger(__name__)


class ProfilerMode(Enum):
    """Profiler modes."""
    SIMPLE = "simple"  # Simple timing
    DETAILED = "detailed"  # Detailed profiling
    MEMORY = "memory"  # Memory profiling
    FULL = "full"  # Full profiling


@dataclass
class ProfileResult:
    """Profile result."""
    function_name: str
    total_time: float
    call_count: int
    cumulative_time: float
    per_call_time: float
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class PerformanceProfiler:
    """
    Performance profiler.
    
    Features:
    - Simple and detailed profiling
    - Memory profiling
    - Call statistics
    - Performance reports
    - Context managers
    - Decorators
    """
    
    def __init__(self, mode: ProfilerMode = ProfilerMode.SIMPLE):
        """
        Initialize profiler.
        
        Args:
            mode: Profiler mode
        """
        self.mode = mode
        self._profiler: Optional[cProfile.Profile] = None
        self._results: Dict[str, ProfileResult] = {}
        self._active_profiles: Dict[str, float] = {}
    
    @contextmanager
    def profile(self, name: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Context manager for profiling.
        
        Args:
            name: Profile name
            metadata: Optional metadata
        """
        start_time = time.time()
        
        if self.mode in [ProfilerMode.DETAILED, ProfilerMode.FULL]:
            profiler = cProfile.Profile()
            profiler.enable()
        else:
            profiler = None
        
        try:
            yield
        finally:
            elapsed = time.time() - start_time
            
            if profiler:
                profiler.disable()
                stats = self._extract_stats(profiler, name)
            else:
                stats = ProfileResult(
                    function_name=name,
                    total_time=elapsed,
                    call_count=1,
                    cumulative_time=elapsed,
                    per_call_time=elapsed,
                    metadata=metadata or {}
                )
            
            self._results[name] = stats
            logger.debug(f"Profiled {name}: {elapsed:.4f}s")
    
    def _extract_stats(self, profiler: cProfile.Profile, name: str) -> ProfileResult:
        """Extract statistics from profiler."""
        stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stream)
        stats.sort_stats('cumulative')
        
        # Get top function stats
        total_time = stats.total_tt
        call_count = stats.total_calls
        
        return ProfileResult(
            function_name=name,
            total_time=total_time,
            call_count=call_count,
            cumulative_time=total_time,
            per_call_time=total_time / call_count if call_count > 0 else 0.0,
            metadata={"stats_output": stream.getvalue()}
        )
    
    def profile_function(
        self,
        func: Callable,
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Decorator for profiling functions.
        
        Args:
            func: Function to profile
            name: Optional profile name
            metadata: Optional metadata
        """
        profile_name = name or func.__name__
        
        if asyncio.iscoroutinefunction(func):
            async def async_wrapper(*args, **kwargs):
                with self.profile(profile_name, metadata):
                    return await func(*args, **kwargs)
            return async_wrapper
        else:
            def sync_wrapper(*args, **kwargs):
                with self.profile(profile_name, metadata):
                    return func(*args, **kwargs)
            return sync_wrapper
    
    def get_results(self, name: Optional[str] = None) -> Union[ProfileResult, Dict[str, ProfileResult]]:
        """
        Get profiling results.
        
        Args:
            name: Optional profile name
            
        Returns:
            Profile result or all results
        """
        if name:
            return self._results.get(name)
        return self._results.copy()
    
    def get_slowest_functions(self, limit: int = 10) -> List[ProfileResult]:
        """
        Get slowest functions.
        
        Args:
            limit: Number of functions to return
            
        Returns:
            List of slowest functions
        """
        results = sorted(
            self._results.values(),
            key=lambda x: x.total_time,
            reverse=True
        )
        return results[:limit]
    
    def generate_report(self) -> str:
        """Generate profiling report."""
        if not self._results:
            return "No profiling data available."
        
        report = ["Performance Profiling Report", "=" * 50, ""]
        
        # Summary
        total_time = sum(r.total_time for r in self._results.values())
        report.append(f"Total Functions Profiled: {len(self._results)}")
        report.append(f"Total Time: {total_time:.4f}s")
        report.append("")
        
        # Slowest functions
        report.append("Slowest Functions:")
        report.append("-" * 50)
        for result in self.get_slowest_functions(10):
            report.append(
                f"{result.function_name}: {result.total_time:.4f}s "
                f"({result.call_count} calls, {result.per_call_time:.4f}s/call)"
            )
        
        return "\n".join(report)
    
    def clear(self):
        """Clear profiling results."""
        self._results.clear()
        logger.info("Profiling results cleared")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get profiler statistics."""
        if not self._results:
            return {
                "profiles_count": 0,
                "total_time": 0.0,
            }
        
        return {
            "profiles_count": len(self._results),
            "total_time": sum(r.total_time for r in self._results.values()),
            "avg_time": sum(r.total_time for r in self._results.values()) / len(self._results),
            "slowest_function": max(self._results.values(), key=lambda x: x.total_time).function_name,
        }


# Import asyncio for async support
import asyncio




