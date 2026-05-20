"""
MCP Performance Optimization - Optimizaciones de rendimiento
============================================================
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional, Callable
from functools import lru_cache, wraps
from datetime import datetime
import time

logger = logging.getLogger(__name__)


class PerformanceOptimizer:
    """
    Optimizador de rendimiento
    
    Proporciona herramientas para optimizar performance.
    """
    
    def __init__(self):
        self._cache_stats: Dict[str, int] = {"hits": 0, "misses": 0}
    
    @staticmethod
    def memoize(ttl: int = 300):
        """
        Decorador de memoización con TTL
        
        Args:
            ttl: Time to live en segundos
        """
        def decorator(func):
            cache = {}
            cache_times = {}
            
            @wraps(func)
            async def wrapper(*args, **kwargs):
                cache_key = str(args) + str(sorted(kwargs.items()))
                now = time.time()
                
                # Limpiar entradas expiradas
                expired_keys = [
                    k for k, t in cache_times.items()
                    if now - t > ttl
                ]
                for k in expired_keys:
                    cache.pop(k, None)
                    cache_times.pop(k, None)
                
                # Verificar cache
                if cache_key in cache:
                    return cache[cache_key]
                
                # Ejecutar función
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                # Guardar en cache
                cache[cache_key] = result
                cache_times[cache_key] = now
                
                return result
            
            return wrapper
        return decorator
    
    @staticmethod
    def batch_async(
        items: List[Any],
        batch_size: int = 10,
        max_concurrent: int = 5,
    ) -> Callable:
        """
        Procesa items en batches asíncronos.
        
        Args:
            items: Lista de items a procesar
            batch_size: Tamaño del batch (default: 10)
            max_concurrent: Máximo de batches concurrentes (default: 5)
            
        Returns:
            Función async que procesa todos los items en batches
            
        Raises:
            ValueError: Si batch_size o max_concurrent son inválidos
        """
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if not isinstance(max_concurrent, int) or max_concurrent <= 0:
            raise ValueError("max_concurrent must be a positive integer")
        async def process_batch(batch: List[Any], processor: Callable):
            tasks = [processor(item) for item in batch]
            return await asyncio.gather(*tasks)
        
        async def process_all(processor: Callable):
            batches = [
                items[i:i + batch_size]
                for i in range(0, len(items), batch_size)
            ]
            
            semaphore = asyncio.Semaphore(max_concurrent)
            
            async def process_with_semaphore(batch):
                async with semaphore:
                    return await process_batch(batch, processor)
            
            tasks = [process_with_semaphore(batch) for batch in batches]
            results = await asyncio.gather(*tasks)
            
            # Aplanar resultados
            return [item for batch_results in results for item in batch_results]
        
        return process_all
    
    @staticmethod
    def debounce(wait_seconds: float) -> Callable:
        """
        Decorador de debounce para funciones.
        
        Args:
            wait_seconds: Tiempo de espera en segundos antes de ejecutar
            
        Returns:
            Decorador de función
            
        Raises:
            ValueError: Si wait_seconds es inválido
        """
        if not isinstance(wait_seconds, (int, float)) or wait_seconds < 0:
            raise ValueError("wait_seconds must be a non-negative number")
        
        def decorator(func):
            last_call = [None]
            
            @wraps(func)
            async def wrapper(*args, **kwargs):
                now = time.time()
                
                if last_call[0] is None or now - last_call[0] >= wait_seconds:
                    last_call[0] = now
                    if asyncio.iscoroutinefunction(func):
                        return await func(*args, **kwargs)
                    return func(*args, **kwargs)
                
                return None
            
            return wrapper
        return decorator
    
    @staticmethod
    def throttle(max_calls: int, period_seconds: float):
        """
        Decorador de throttle
        
        Args:
            max_calls: Máximo de llamadas
            period_seconds: Período en segundos
        """
        def decorator(func):
            calls = []
            
            @wraps(func)
            async def wrapper(*args, **kwargs):
                now = time.time()
                
                # Limpiar llamadas antiguas
                calls[:] = [t for t in calls if now - t < period_seconds]
                
                if len(calls) >= max_calls:
                    return None
                
                calls.append(now)
                
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                return func(*args, **kwargs)
            
            return wrapper
        return decorator
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Obtiene estadísticas de cache"""
        return self._cache_stats.copy()

