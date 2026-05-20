"""
Performance Profiler - Sistema de performance profiling
=======================================================
"""

import logging
import time
import cProfile
import pstats
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
from functools import wraps
from io import StringIO

logger = logging.getLogger(__name__)


class PerformanceProfiler:
    """Sistema de performance profiling"""
    
    def __init__(self):
        self.profiles: Dict[str, cProfile.Profile] = {}
        self.function_timings: Dict[str, List[float]] = {}
        self.slow_queries: List[Dict[str, Any]] = []
    
    def profile_function(self, func_name: Optional[str] = None):
        """Decorador para perfilar una función"""
        def decorator(func: Callable):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                start_time = time.time()
                profiler = cProfile.Profile()
                profiler.enable()
                
                try:
                    result = await func(*args, **kwargs)
                finally:
                    profiler.disable()
                    end_time = time.time()
                    duration = end_time - start_time
                    
                    name = func_name or func.__name__
                    self._record_timing(name, duration)
                    self._save_profile(name, profiler)
                    
                    if duration > 1.0:  # Funciones lentas
                        self._record_slow_query(name, duration, args, kwargs)
                
                return result
            
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                start_time = time.time()
                profiler = cProfile.Profile()
                profiler.enable()
                
                try:
                    result = func(*args, **kwargs)
                finally:
                    profiler.disable()
                    end_time = time.time()
                    duration = end_time - start_time
                    
                    name = func_name or func.__name__
                    self._record_timing(name, duration)
                    self._save_profile(name, profiler)
                    
                    if duration > 1.0:
                        self._record_slow_query(name, duration, args, kwargs)
                
                return result
            
            if hasattr(func, '__await__'):
                return async_wrapper
            return sync_wrapper
        
        return decorator
    
    def _record_timing(self, func_name: str, duration: float):
        """Registra tiempo de ejecución"""
        if func_name not in self.function_timings:
            self.function_timings[func_name] = []
        
        self.function_timings[func_name].append(duration)
        
        # Mantener solo últimas 1000
        if len(self.function_timings[func_name]) > 1000:
            self.function_timings[func_name] = self.function_timings[func_name][-1000:]
    
    def _save_profile(self, name: str, profiler: cProfile.Profile):
        """Guarda perfil"""
        self.profiles[name] = profiler
    
    def _record_slow_query(self, func_name: str, duration: float, args: tuple, kwargs: dict):
        """Registra consulta lenta"""
        self.slow_queries.append({
            "function": func_name,
            "duration": duration,
            "timestamp": datetime.now().isoformat(),
            "args_count": len(args),
            "kwargs_keys": list(kwargs.keys())
        })
        
        # Mantener solo últimas 100
        if len(self.slow_queries) > 100:
            self.slow_queries = self.slow_queries[-100:]
    
    def get_function_stats(self, func_name: str) -> Optional[Dict[str, Any]]:
        """Obtiene estadísticas de una función"""
        timings = self.function_timings.get(func_name)
        if not timings:
            return None
        
        return {
            "function": func_name,
            "call_count": len(timings),
            "avg_duration": sum(timings) / len(timings),
            "min_duration": min(timings),
            "max_duration": max(timings),
            "total_duration": sum(timings)
        }
    
    def get_profile_report(self, func_name: str, limit: int = 20) -> Optional[str]:
        """Obtiene reporte de perfil"""
        profiler = self.profiles.get(func_name)
        if not profiler:
            return None
        
        stream = StringIO()
        stats = pstats.Stats(profiler, stream=stream)
        stats.sort_stats('cumulative')
        stats.print_stats(limit)
        
        return stream.getvalue()
    
    def get_slow_queries(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Obtiene consultas lentas"""
        return sorted(self.slow_queries, key=lambda x: x["duration"], reverse=True)[:limit]
    
    def get_all_stats(self) -> Dict[str, Any]:
        """Obtiene todas las estadísticas"""
        all_stats = {}
        for func_name in self.function_timings.keys():
            stats = self.get_function_stats(func_name)
            if stats:
                all_stats[func_name] = stats
        
        return {
            "functions": all_stats,
            "slow_queries": self.get_slow_queries(),
            "total_functions_profiled": len(self.function_timings)
        }




