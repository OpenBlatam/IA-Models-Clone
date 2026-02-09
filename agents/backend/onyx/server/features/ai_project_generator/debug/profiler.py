"""
Profiler - Profiler de performance
==================================

Profiler para analizar performance de código.
"""

import time
import cProfile
import pstats
import io
import logging
from typing import Optional, Dict, Any, Callable
from functools import wraps
from contextlib import contextmanager
from collections import defaultdict

logger = logging.getLogger(__name__)


class Profiler:
    """
    Profiler de performance con:
    - Profiling de funciones
    - Análisis de tiempo de ejecución
    - Estadísticas de llamadas
    - Exportación de resultados
    """
    
    def __init__(self):
        self.profiler = cProfile.Profile()
        self.function_stats: Dict[str, list] = defaultdict(list)
        self.enabled = False
    
    def enable(self):
        """Habilita profiling"""
        self.enabled = True
        self.profiler.enable()
    
    def disable(self):
        """Deshabilita profiling"""
        self.enabled = False
        self.profiler.disable()
    
    @contextmanager
    def profile(self, name: str = "operation"):
        """
        Context manager para profiling.
        
        Example:
            with profiler.profile("my_function"):
                # código a profilear
                pass
        """
        start_time = time.time()
        if self.enabled:
            self.profiler.enable()
        
        try:
            yield
        finally:
            if self.enabled:
                self.profiler.disable()
            
            duration = time.time() - start_time
            self.function_stats[name].append(duration)
    
    def profile_function(self, func: Callable):
        """
        Decorator para profilear función.
        
        Example:
            @profiler.profile_function
            async def my_function():
                pass
        """
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            with self.profile(func.__name__):
                return await func(*args, **kwargs)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            with self.profile(func.__name__):
                return func(*args, **kwargs)
        
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas de profiling"""
        stats = {}
        
        # Estadísticas de funciones
        for func_name, durations in self.function_stats.items():
            if durations:
                stats[func_name] = {
                    "call_count": len(durations),
                    "total_time": sum(durations),
                    "avg_time": sum(durations) / len(durations),
                    "min_time": min(durations),
                    "max_time": max(durations)
                }
        
        # Estadísticas de cProfile
        if self.enabled:
            stream = io.StringIO()
            pstats.Stats(self.profiler, stream=stream).sort_stats('cumulative').print_stats(20)
            stats["cprofile"] = stream.getvalue()
        
        return stats
    
    def get_slowest_functions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Obtiene funciones más lentas"""
        function_avg_times = []
        for func_name, durations in self.function_stats.items():
            if durations:
                avg_time = sum(durations) / len(durations)
                function_avg_times.append({
                    "function": func_name,
                    "avg_time": avg_time,
                    "call_count": len(durations)
                })
        
        return sorted(
            function_avg_times,
            key=lambda x: x["avg_time"],
            reverse=True
        )[:limit]
    
    def reset(self):
        """Resetea estadísticas"""
        self.profiler = cProfile.Profile()
        self.function_stats.clear()


# Instancia global
_profiler: Optional[Profiler] = None


def get_profiler() -> Profiler:
    """Obtiene instancia de profiler"""
    global _profiler
    if _profiler is None:
        _profiler = Profiler()
    return _profiler















