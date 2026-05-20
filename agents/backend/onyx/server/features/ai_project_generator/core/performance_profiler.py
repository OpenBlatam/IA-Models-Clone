"""
Performance Profiler - Profiler de Performance
=============================================

Profiling avanzado de performance:
- Function profiling
- Memory profiling
- Database query profiling
- Request profiling
- Performance recommendations
"""

import logging
import time
import tracemalloc
from typing import Optional, Dict, Any, List, Callable
from functools import wraps
from contextlib import contextmanager
from datetime import datetime

logger = logging.getLogger(__name__)


class PerformanceProfiler:
    """
    Profiler de performance avanzado.
    """
    
    def __init__(self) -> None:
        self.profiles: Dict[str, List[Dict[str, Any]]] = {}
        self.tracemalloc_enabled = False
    
    def start_tracemalloc(self) -> None:
        """Inicia tracemalloc"""
        if not self.tracemalloc_enabled:
            tracemalloc.start()
            self.tracemalloc_enabled = True
    
    def stop_tracemalloc(self) -> None:
        """Detiene tracemalloc"""
        if self.tracemalloc_enabled:
            tracemalloc.stop()
            self.tracemalloc_enabled = False
    
    @contextmanager
    def profile_function(
        self,
        function_name: str,
        collect_memory: bool = True
    ):
        """Context manager para profiling de funciones"""
        start_time = time.perf_counter()
        start_memory = None
        
        if collect_memory and self.tracemalloc_enabled:
            start_memory = tracemalloc.take_snapshot()
        
        try:
            yield
        finally:
            end_time = time.perf_counter()
            duration = end_time - start_time
            
            memory_info = {}
            if collect_memory and self.tracemalloc_enabled and start_memory:
                end_memory = tracemalloc.take_snapshot()
                top_stats = end_memory.compare_to(start_memory, "lineno")
                memory_info = {
                    "peak_memory": sum(stat.size_diff for stat in top_stats[:10]),
                    "top_allocations": [
                        {
                            "file": stat.traceback[0].filename,
                            "line": stat.traceback[0].lineno,
                            "size": stat.size_diff
                        }
                        for stat in top_stats[:5]
                    ]
                }
            
            profile_data = {
                "function": function_name,
                "duration": duration,
                "timestamp": datetime.now().isoformat(),
                **memory_info
            }
            
            if function_name not in self.profiles:
                self.profiles[function_name] = []
            self.profiles[function_name].append(profile_data)
    
    def profile_decorator(
        self,
        function_name: Optional[str] = None,
        collect_memory: bool = True
    ):
        """Decorator para profiling"""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                name = function_name or f"{func.__module__}.{func.__name__}"
                with self.profile_function(name, collect_memory):
                    return await func(*args, **kwargs)
            
            @wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                name = function_name or f"{func.__module__}.{func.__name__}"
                with self.profile_function(name, collect_memory):
                    return func(*args, **kwargs)
            
            import asyncio
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            return sync_wrapper
        
        return decorator
    
    def get_function_stats(self, function_name: str) -> Optional[Dict[str, Any]]:
        """Obtiene estadísticas de una función"""
        if function_name not in self.profiles:
            return None
        
        profiles = self.profiles[function_name]
        if not profiles:
            return None
        
        durations = [p["duration"] for p in profiles]
        
        return {
            "function": function_name,
            "call_count": len(profiles),
            "avg_duration": sum(durations) / len(durations),
            "min_duration": min(durations),
            "max_duration": max(durations),
            "total_duration": sum(durations)
        }
    
    def get_all_stats(self) -> Dict[str, Any]:
        """Obtiene todas las estadísticas"""
        stats = {}
        for function_name in self.profiles:
            stats[function_name] = self.get_function_stats(function_name)
        
        return stats
    
    def get_slow_functions(self, threshold: float = 1.0) -> List[Dict[str, Any]]:
        """Obtiene funciones lentas"""
        slow_functions = []
        
        for function_name in self.profiles:
            stats = self.get_function_stats(function_name)
            if stats and stats["avg_duration"] > threshold:
                slow_functions.append(stats)
        
        return sorted(slow_functions, key=lambda x: x["avg_duration"], reverse=True)
    
    def get_recommendations(self) -> List[Dict[str, Any]]:
        """Obtiene recomendaciones de optimización"""
        recommendations = []
        
        slow_functions = self.get_slow_functions(threshold=0.5)
        if slow_functions:
            recommendations.append({
                "type": "slow_functions",
                "priority": "high",
                "message": f"{len(slow_functions)} functions are slow (>0.5s)",
                "details": slow_functions[:5]
            })
        
        # Analizar uso de memoria
        for function_name, profiles in self.profiles.items():
            memory_profiles = [p for p in profiles if "peak_memory" in p]
            if memory_profiles:
                avg_memory = sum(p["peak_memory"] for p in memory_profiles) / len(memory_profiles)
                if avg_memory > 10 * 1024 * 1024:  # > 10MB
                    recommendations.append({
                        "type": "high_memory",
                        "priority": "medium",
                        "message": f"{function_name} uses high memory ({avg_memory / 1024 / 1024:.2f}MB)",
                        "function": function_name
                    })
        
        return recommendations


def get_performance_profiler() -> PerformanceProfiler:
    """Obtiene profiler de performance"""
    return PerformanceProfiler()















