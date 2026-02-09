"""
Performance Service - Optimizaciones de performance
===================================================

Servicio para optimizaciones y mejoras de performance.
"""

import logging
import time
from functools import wraps
from typing import Callable, Any
from collections import defaultdict

logger = logging.getLogger(__name__)


class PerformanceService:
    """Servicio de performance"""
    
    def __init__(self):
        """Inicializar servicio"""
        self.query_times: defaultdict = defaultdict(list)
        self.slow_queries: list = []
        logger.info("PerformanceService initialized")
    
    def track_performance(self, threshold_seconds: float = 1.0):
        """Decorator para trackear performance"""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = await func(*args, **kwargs)
                    duration = time.time() - start_time
                    
                    self.query_times[func.__name__].append(duration)
                    
                    if duration > threshold_seconds:
                        self.slow_queries.append({
                            "function": func.__name__,
                            "duration": duration,
                            "args": str(args)[:100],
                        })
                        logger.warning(
                            f"Slow query detected: {func.__name__} took {duration:.2f}s"
                        )
                    
                    return result
                except Exception as e:
                    duration = time.time() - start_time
                    logger.error(f"Error in {func.__name__} after {duration:.2f}s: {e}")
                    raise
            
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    duration = time.time() - start_time
                    
                    self.query_times[func.__name__].append(duration)
                    
                    if duration > threshold_seconds:
                        self.slow_queries.append({
                            "function": func.__name__,
                            "duration": duration,
                        })
                    
                    return result
                except Exception as e:
                    duration = time.time() - start_time
                    logger.error(f"Error in {func.__name__} after {duration:.2f}s: {e}")
                    raise
            
            # Retornar wrapper apropiado
            import asyncio
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            return sync_wrapper
        
        return decorator
    
    def get_performance_stats(self) -> dict:
        """Obtener estadísticas de performance"""
        stats = {}
        
        for func_name, times in self.query_times.items():
            if times:
                stats[func_name] = {
                    "count": len(times),
                    "avg": sum(times) / len(times),
                    "min": min(times),
                    "max": max(times),
                    "total": sum(times),
                }
        
        return {
            "functions": stats,
            "slow_queries_count": len(self.slow_queries),
            "slow_queries": self.slow_queries[-10:],  # Últimos 10
        }
    
    def optimize_query(self, query: str) -> str:
        """Optimizar query (placeholder para optimizaciones futuras)"""
        # En producción, aquí se harían optimizaciones reales
        return query


# Instancia global
performance_service = PerformanceService()




