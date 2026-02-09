"""
Performance Module - Enhanced performance optimization utilities.

Provides:
- Advanced performance profiling with context managers
- Memory optimization and tracking
- CPU optimization and monitoring
- Caching strategies
- Performance recommendations
- Resource usage tracking
"""

import logging
import time
import psutil
import gc
import sys
from typing import Dict, Any, List, Optional, Callable, ContextManager
from dataclasses import dataclass, field
from functools import wraps
from datetime import datetime
from contextlib import contextmanager
from enum import Enum

logger = logging.getLogger(__name__)


class ProfilingLevel(str, Enum):
    """Profiling detail levels."""
    MINIMAL = "minimal"  # Time only
    BASIC = "basic"  # Time + memory
    DETAILED = "detailed"  # Time + memory + CPU
    FULL = "full"  # All metrics + system resources


@dataclass
class PerformanceProfile:
    """Enhanced performance profile data."""
    function_name: str
    call_count: int = 0
    total_time: float = 0.0
    avg_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0
    std_dev_time: float = 0.0
    memory_delta: float = 0.0
    memory_peak: float = 0.0
    cpu_percent: float = 0.0
    cpu_peak: float = 0.0
    last_call_time: Optional[float] = None
    time_history: List[float] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "function_name": self.function_name,
            "call_count": self.call_count,
            "total_time": self.total_time,
            "avg_time": self.avg_time,
            "min_time": self.min_time if self.min_time != float('inf') else 0.0,
            "max_time": self.max_time,
            "std_dev_time": self.std_dev_time,
            "memory_delta": self.memory_delta,
            "memory_peak": self.memory_peak,
            "cpu_percent": self.cpu_percent,
            "cpu_peak": self.cpu_peak,
            "last_call_time": self.last_call_time,
        }


class PerformanceProfiler:
    """
    Enhanced performance profiler with multiple profiling levels.
    
    Features:
    - Multiple profiling levels
    - Context manager support
    - Statistical analysis
    - Memory and CPU tracking
    """
    
    def __init__(self, level: ProfilingLevel = ProfilingLevel.BASIC):
        """
        Initialize profiler.
        
        Args:
            level: Profiling detail level
        """
        self.profiles: Dict[str, PerformanceProfile] = {}
        self.enabled = True
        self.level = level
        self.process = psutil.Process()
    
    def profile(
        self,
        func: Optional[Callable] = None,
        level: Optional[ProfilingLevel] = None,
    ) -> Callable:
        """
        Decorator to profile function.
        
        Args:
            func: Function to profile (if used as decorator)
            level: Optional profiling level override
        
        Returns:
            Wrapped function
        """
        profiling_level = level or self.level
        
        def decorator(f: Callable) -> Callable:
            @wraps(f)
            def wrapper(*args, **kwargs):
                if not self.enabled:
                    return f(*args, **kwargs)
                
                return self._profile_execution(f, args, kwargs, profiling_level)
            
            return wrapper
        
        if func is None:
            return decorator
        else:
            return decorator(func)
    
    @contextmanager
    def profile_context(
        self,
        name: str,
        level: Optional[ProfilingLevel] = None,
    ) -> ContextManager[None]:
        """
        Context manager for profiling code blocks.
        
        Args:
            name: Context name
            level: Optional profiling level override
        
        Yields:
            None
        """
        if not self.enabled:
            yield
            return
        
        profiling_level = level or self.level
        start_time = time.time()
        
        # Get initial metrics
        mem_before = 0.0
        cpu_before = 0.0
        
        if profiling_level in [ProfilingLevel.BASIC, ProfilingLevel.DETAILED, ProfilingLevel.FULL]:
            mem_before = self.process.memory_info().rss / 1024 / 1024  # MB
        
        if profiling_level in [ProfilingLevel.DETAILED, ProfilingLevel.FULL]:
            cpu_before = self.process.cpu_percent()
        
        try:
            yield
        finally:
            end_time = time.time()
            duration = end_time - start_time
            
            # Get final metrics
            mem_after = 0.0
            cpu_after = 0.0
            
            if profiling_level in [ProfilingLevel.BASIC, ProfilingLevel.DETAILED, ProfilingLevel.FULL]:
                mem_after = self.process.memory_info().rss / 1024 / 1024  # MB
            
            if profiling_level in [ProfilingLevel.DETAILED, ProfilingLevel.FULL]:
                cpu_after = self.process.cpu_percent()
            
            self._update_profile(
                name,
                duration,
                mem_after - mem_before,
                (cpu_before + cpu_after) / 2 if profiling_level in [ProfilingLevel.DETAILED, ProfilingLevel.FULL] else 0.0,
            )
    
    def _profile_execution(
        self,
        func: Callable,
        args: tuple,
        kwargs: dict,
        level: ProfilingLevel,
    ) -> Any:
        """Execute function with profiling."""
        start_time = time.time()
        
        # Get initial metrics
        mem_before = 0.0
        cpu_before = 0.0
        
        if level in [ProfilingLevel.BASIC, ProfilingLevel.DETAILED, ProfilingLevel.FULL]:
            mem_before = self.process.memory_info().rss / 1024 / 1024  # MB
        
        if level in [ProfilingLevel.DETAILED, ProfilingLevel.FULL]:
            cpu_before = self.process.cpu_percent()
        
        try:
            result = func(*args, **kwargs)
        finally:
            end_time = time.time()
            duration = end_time - start_time
            
            # Get final metrics
            mem_after = 0.0
            cpu_after = 0.0
            
            if level in [ProfilingLevel.BASIC, ProfilingLevel.DETAILED, ProfilingLevel.FULL]:
                mem_after = self.process.memory_info().rss / 1024 / 1024  # MB
            
            if level in [ProfilingLevel.DETAILED, ProfilingLevel.FULL]:
                cpu_after = self.process.cpu_percent()
            
            self._update_profile(
                func.__name__,
                duration,
                mem_after - mem_before,
                (cpu_before + cpu_after) / 2 if level in [ProfilingLevel.DETAILED, ProfilingLevel.FULL] else 0.0,
            )
        
        return result
    
    def _update_profile(
        self,
        function_name: str,
        duration: float,
        memory_delta: float,
        cpu_percent: float,
    ) -> None:
        """Update performance profile with statistics."""
        if function_name not in self.profiles:
            self.profiles[function_name] = PerformanceProfile(function_name=function_name)
        
        profile = self.profiles[function_name]
        profile.call_count += 1
        profile.total_time += duration
        profile.avg_time = profile.total_time / profile.call_count
        profile.min_time = min(profile.min_time, duration)
        profile.max_time = max(profile.max_time, duration)
        profile.memory_delta += memory_delta
        profile.memory_peak = max(profile.memory_peak, abs(memory_delta))
        profile.cpu_percent = (profile.cpu_percent * (profile.call_count - 1) + cpu_percent) / profile.call_count
        profile.cpu_peak = max(profile.cpu_peak, cpu_percent)
        profile.last_call_time = time.time()
        profile.time_history.append(duration)
        
        # Calculate standard deviation
        if len(profile.time_history) > 1:
            mean = profile.avg_time
            variance = sum((t - mean) ** 2 for t in profile.time_history) / len(profile.time_history)
            profile.std_dev_time = variance ** 0.5
    
    def get_profile(self, function_name: str) -> Optional[PerformanceProfile]:
        """Get profile for function."""
        return self.profiles.get(function_name)
    
    def get_all_profiles(self) -> List[PerformanceProfile]:
        """Get all profiles."""
        return list(self.profiles.values())
    
    def get_slowest_functions(self, top_n: int = 10) -> List[PerformanceProfile]:
        """Get slowest functions by average time."""
        profiles = sorted(
            self.profiles.values(),
            key=lambda p: p.avg_time,
            reverse=True
        )
        return profiles[:top_n]
    
    def get_most_called_functions(self, top_n: int = 10) -> List[PerformanceProfile]:
        """Get most called functions."""
        profiles = sorted(
            self.profiles.values(),
            key=lambda p: p.call_count,
            reverse=True
        )
        return profiles[:top_n]
    
    def get_memory_intensive_functions(self, top_n: int = 10) -> List[PerformanceProfile]:
        """Get most memory-intensive functions."""
        profiles = sorted(
            self.profiles.values(),
            key=lambda p: p.memory_peak,
            reverse=True
        )
        return profiles[:top_n]
    
    def reset(self) -> None:
        """Reset all profiles."""
        self.profiles.clear()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get profiling summary."""
        if not self.profiles:
            return {"enabled": self.enabled, "profiles": 0}
        
        total_calls = sum(p.call_count for p in self.profiles.values())
        total_time = sum(p.total_time for p in self.profiles.values())
        
        return {
            "enabled": self.enabled,
            "level": self.level.value,
            "total_functions": len(self.profiles),
            "total_calls": total_calls,
            "total_time": total_time,
            "avg_time_per_call": total_time / total_calls if total_calls > 0 else 0.0,
            "slowest_functions": [
                p.to_dict() for p in self.get_slowest_functions(5)
            ],
        }


class MemoryOptimizer:
    """Enhanced memory optimization utilities."""
    
    @staticmethod
    def optimize_memory(aggressive: bool = False) -> Dict[str, Any]:
        """
        Optimize memory usage.
        
        Args:
            aggressive: If True, perform aggressive cleanup
        
        Returns:
            Optimization results
        """
        process = psutil.Process()
        before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Force garbage collection
        collected = gc.collect()
        
        if aggressive:
            # Multiple passes
            for _ in range(3):
                gc.collect()
        
        after = process.memory_info().rss / 1024 / 1024  # MB
        freed = before - after
        
        return {
            "before_mb": before,
            "after_mb": after,
            "freed_mb": freed,
            "freed_percent": (freed / before * 100) if before > 0 else 0.0,
            "collected_objects": collected,
        }
    
    @staticmethod
    def get_memory_usage() -> Dict[str, float]:
        """Get current memory usage."""
        process = psutil.Process()
        mem_info = process.memory_info()
        
        return {
            "rss_mb": mem_info.rss / 1024 / 1024,
            "vms_mb": mem_info.vms / 1024 / 1024,
            "percent": process.memory_percent(),
            "available_mb": psutil.virtual_memory().available / 1024 / 1024,
        }
    
    @staticmethod
    def get_system_memory() -> Dict[str, Any]:
        """Get system memory information."""
        mem = psutil.virtual_memory()
        
        return {
            "total_gb": mem.total / 1024 / 1024 / 1024,
            "available_gb": mem.available / 1024 / 1024 / 1024,
            "used_gb": mem.used / 1024 / 1024 / 1024,
            "free_gb": mem.free / 1024 / 1024 / 1024,
            "percent": mem.percent,
            "cached_gb": getattr(mem, 'cached', 0) / 1024 / 1024 / 1024,
        }
    
    @staticmethod
    def get_memory_info() -> Dict[str, Any]:
        """Get detailed memory information."""
        process = psutil.Process()
        mem_info = process.memory_info()
        system_mem = psutil.virtual_memory()
        
        return {
            "process": {
                "rss_mb": mem_info.rss / 1024 / 1024,
                "vms_mb": mem_info.vms / 1024 / 1024,
                "percent": process.memory_percent(),
            },
            "system": {
                "total_gb": system_mem.total / 1024 / 1024 / 1024,
                "available_gb": system_mem.available / 1024 / 1024 / 1024,
                "used_gb": system_mem.used / 1024 / 1024 / 1024,
                "percent": system_mem.percent,
            },
        }


class CPUOptimizer:
    """Enhanced CPU optimization utilities."""
    
    @staticmethod
    def get_cpu_usage(interval: float = 1.0) -> Dict[str, Any]:
        """
        Get CPU usage.
        
        Args:
            interval: Sampling interval in seconds
        
        Returns:
            CPU usage information
        """
        cpu_percent = psutil.cpu_percent(interval=interval)
        cpu_per_core = psutil.cpu_percent(interval=interval, percpu=True)
        
        return {
            "percent": cpu_percent,
            "per_core": cpu_per_core,
            "count": psutil.cpu_count(),
            "count_logical": psutil.cpu_count(logical=True),
        }
    
    @staticmethod
    def get_cpu_frequency() -> Dict[str, Any]:
        """Get CPU frequency information."""
        try:
            freq = psutil.cpu_freq()
            return {
                "current_mhz": freq.current,
                "min_mhz": freq.min,
                "max_mhz": freq.max,
            }
        except (AttributeError, RuntimeError):
            return {}
    
    @staticmethod
    def get_cpu_stats() -> Dict[str, Any]:
        """Get CPU statistics."""
        try:
            stats = psutil.cpu_stats()
            return {
                "ctx_switches": stats.ctx_switches,
                "interrupts": stats.interrupts,
                "soft_interrupts": stats.soft_interrupts,
                "syscalls": stats.syscalls,
            }
        except (AttributeError, RuntimeError):
            return {}
    
    @staticmethod
    def get_cpu_times() -> Dict[str, float]:
        """Get CPU times."""
        try:
            times = psutil.cpu_times()
            return {
                "user": times.user,
                "system": times.system,
                "idle": times.idle,
                "nice": getattr(times, 'nice', 0.0),
                "iowait": getattr(times, 'iowait', 0.0),
            }
        except (AttributeError, RuntimeError):
            return {}


class PerformanceOptimizer:
    """Enhanced performance optimizer with recommendations."""
    
    def __init__(self, profiler: Optional[PerformanceProfiler] = None):
        """
        Initialize optimizer.
        
        Args:
            profiler: Optional profiler instance
        """
        self.profiler = profiler or PerformanceProfiler()
        self.memory_optimizer = MemoryOptimizer()
        self.cpu_optimizer = CPUOptimizer()
    
    def get_recommendations(self) -> List[Dict[str, Any]]:
        """
        Get performance recommendations with priorities.
        
        Returns:
            List of recommendation dictionaries
        """
        recommendations = []
        
        # Memory recommendations
        mem_usage = self.memory_optimizer.get_memory_usage()
        if mem_usage["percent"] > 90:
            recommendations.append({
                "priority": "high",
                "category": "memory",
                "message": (
                    f"Critical memory usage ({mem_usage['percent']:.1f}%). "
                    "Consider optimizing memory or increasing available memory."
                ),
                "action": "optimize_memory",
            })
        elif mem_usage["percent"] > 80:
            recommendations.append({
                "priority": "medium",
                "category": "memory",
                "message": (
                    f"High memory usage ({mem_usage['percent']:.1f}%). "
                    "Monitor and consider optimization."
                ),
                "action": "monitor_memory",
            })
        
        # CPU recommendations
        cpu_usage = self.cpu_optimizer.get_cpu_usage()
        if cpu_usage["percent"] > 90:
            recommendations.append({
                "priority": "high",
                "category": "cpu",
                "message": (
                    f"Critical CPU usage ({cpu_usage['percent']:.1f}%). "
                    "Consider optimizing CPU-intensive operations or scaling."
                ),
                "action": "optimize_cpu",
            })
        elif cpu_usage["percent"] > 80:
            recommendations.append({
                "priority": "medium",
                "category": "cpu",
                "message": (
                    f"High CPU usage ({cpu_usage['percent']:.1f}%). "
                    "Monitor and consider optimization."
                ),
                "action": "monitor_cpu",
            })
        
        # Slow function recommendations
        slowest = self.profiler.get_slowest_functions(top_n=5)
        for profile in slowest:
            if profile.avg_time > 5.0:
                recommendations.append({
                    "priority": "high",
                    "category": "performance",
                    "message": (
                        f"Very slow function: {profile.function_name} "
                        f"(avg: {profile.avg_time:.2f}s). "
                        "Consider optimization or caching."
                    ),
                    "action": f"optimize_function:{profile.function_name}",
                })
            elif profile.avg_time > 1.0:
                recommendations.append({
                    "priority": "medium",
                    "category": "performance",
                    "message": (
                        f"Slow function: {profile.function_name} "
                        f"(avg: {profile.avg_time:.2f}s). "
                        "Consider optimization."
                    ),
                    "action": f"review_function:{profile.function_name}",
                })
        
        # Memory-intensive function recommendations
        memory_intensive = self.profiler.get_memory_intensive_functions(top_n=3)
        for profile in memory_intensive:
            if profile.memory_peak > 100:  # MB
                recommendations.append({
                    "priority": "medium",
                    "category": "memory",
                    "message": (
                        f"Memory-intensive function: {profile.function_name} "
                        f"(peak: {profile.memory_peak:.1f}MB). "
                        "Consider memory optimization."
                    ),
                    "action": f"optimize_memory:{profile.function_name}",
                })
        
        return recommendations
    
    def get_performance_report(self) -> Dict[str, Any]:
        """
        Get comprehensive performance report.
        
        Returns:
            Performance report dictionary
        """
        return {
            "timestamp": datetime.now().isoformat(),
            "memory": self.memory_optimizer.get_memory_info(),
            "cpu": {
                "usage": self.cpu_optimizer.get_cpu_usage(),
                "frequency": self.cpu_optimizer.get_cpu_frequency(),
                "stats": self.cpu_optimizer.get_cpu_stats(),
                "times": self.cpu_optimizer.get_cpu_times(),
            },
            "profiling": self.profiler.get_summary(),
            "recommendations": self.get_recommendations(),
        }


# Global performance optimizer
performance_optimizer = PerformanceOptimizer()


__all__ = [
    "ProfilingLevel",
    "PerformanceProfile",
    "PerformanceProfiler",
    "MemoryOptimizer",
    "CPUOptimizer",
    "PerformanceOptimizer",
    "performance_optimizer",
]
