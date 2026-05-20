"""
MCP Profiling - Profiling de performance
=========================================
"""

import time
import cProfile
import pstats
import io
import logging
from typing import Dict, Any, Optional, Callable
from functools import wraps
from contextlib import contextmanager
from datetime import datetime

logger = logging.getLogger(__name__)


class PerformanceProfiler:
    """
    Profiler de performance
    
    Permite medir tiempo de ejecución y generar reportes.
    """
    
    def __init__(self):
        self._profiles: Dict[str, Dict[str, Any]] = {}
        self._enabled = True
    
    @contextmanager
    def profile(self, name: str):
        """
        Context manager para profiling
        
        Args:
            name: Nombre de la operación a perfilar
        """
        if not self._enabled:
            yield
            return
        
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        profiler = cProfile.Profile()
        profiler.enable()
        
        try:
            yield
        finally:
            profiler.disable()
            
            end_time = time.time()
            end_memory = self._get_memory_usage()
            
            duration = end_time - start_time
            memory_delta = end_memory - start_memory
            
            # Generar stats
            stats_stream = io.StringIO()
            stats = pstats.Stats(profiler, stream=stats_stream)
            stats.sort_stats('cumulative')
            stats.print_stats(20)  # Top 20 funciones
            
            self._profiles[name] = {
                "duration": duration,
                "memory_delta": memory_delta,
                "start_memory": start_memory,
                "end_memory": end_memory,
                "stats": stats_stream.getvalue(),
                "timestamp": datetime.utcnow().isoformat(),
            }
            
            logger.info(
                f"Profiled {name}: {duration:.3f}s, "
                f"memory: {memory_delta / 1024 / 1024:.2f}MB"
            )
    
    def _get_memory_usage(self) -> int:
        """Obtiene uso de memoria en bytes"""
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            return process.memory_info().rss
        except ImportError:
            return 0
    
    def get_profile(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Obtiene perfil de una operación
        
        Args:
            name: Nombre de la operación
            
        Returns:
            Diccionario con perfil o None
        """
        return self._profiles.get(name)
    
    def get_all_profiles(self) -> Dict[str, Dict[str, Any]]:
        """
        Obtiene todos los perfiles
        
        Returns:
            Diccionario con todos los perfiles
        """
        return self._profiles.copy()
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Obtiene resumen de perfiles
        
        Returns:
            Diccionario con resumen
        """
        if not self._profiles:
            return {"total_profiles": 0}
        
        durations = [p["duration"] for p in self._profiles.values()]
        memory_deltas = [p["memory_delta"] for p in self._profiles.values()]
        
        return {
            "total_profiles": len(self._profiles),
            "total_duration": sum(durations),
            "avg_duration": sum(durations) / len(durations),
            "min_duration": min(durations),
            "max_duration": max(durations),
            "total_memory_delta": sum(memory_deltas),
            "avg_memory_delta": sum(memory_deltas) / len(memory_deltas),
        }
    
    def clear(self):
        """Limpia todos los perfiles"""
        self._profiles.clear()
    
    def enable(self):
        """Habilita profiling"""
        self._enabled = True
    
    def disable(self):
        """Deshabilita profiling"""
        self._enabled = False


def profile_function(name: Optional[str] = None):
    """
    Decorador para perfilar funciones
    
    Args:
        name: Nombre de la operación (opcional, usa nombre de función si no se especifica)
    """
    def decorator(func: Callable) -> Callable:
        func_name = name or func.__name__
        profiler = PerformanceProfiler()
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            with profiler.profile(func_name):
                return await func(*args, **kwargs)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            with profiler.profile(func_name):
                return func(*args, **kwargs)
        
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

