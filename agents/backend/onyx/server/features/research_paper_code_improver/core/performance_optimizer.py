"""
Performance Optimizer - Optimizaciones avanzadas de performance
================================================================
"""

import logging
import time
import functools
from typing import Dict, Any, Optional, Callable, List
from collections import OrderedDict
import asyncio

logger = logging.getLogger(__name__)


class LRUCache:
    """Cache LRU (Least Recently Used) para optimización"""
    
    def __init__(self, max_size: int = 128):
        self.cache = OrderedDict()
        self.max_size = max_size
    
    def get(self, key: str) -> Optional[Any]:
        """Obtiene valor del cache"""
        if key in self.cache:
            # Mover al final (más reciente)
            self.cache.move_to_end(key)
            return self.cache[key]
        return None
    
    def set(self, key: str, value: Any):
        """Establece valor en cache"""
        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            if len(self.cache) >= self.max_size:
                # Eliminar el más antiguo
                self.cache.popitem(last=False)
        self.cache[key] = value
    
    def clear(self):
        """Limpia el cache"""
        self.cache.clear()


class PerformanceOptimizer:
    """
    Optimizador de performance con caching avanzado y profiling.
    """
    
    def __init__(self):
        """Inicializar optimizador"""
        self.lru_cache = LRUCache(max_size=256)
        self.profiling_data: Dict[str, List[float]] = {}
        self.enabled = True
    
    def cached(self, ttl_seconds: int = 3600):
        """
        Decorador para cachear resultados de funciones.
        
        Args:
            ttl_seconds: Tiempo de vida del cache en segundos
        """
        def decorator(func: Callable) -> Callable:
            cache_data = {}
            
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                if not self.enabled:
                    return await func(*args, **kwargs)
                
                # Generar clave de cache
                cache_key = self._generate_cache_key(func.__name__, args, kwargs)
                
                # Verificar cache
                cached = self.lru_cache.get(cache_key)
                if cached and time.time() - cached["timestamp"] < ttl_seconds:
                    logger.debug(f"Cache hit: {func.__name__}")
                    return cached["value"]
                
                # Ejecutar función
                result = await func(*args, **kwargs)
                
                # Guardar en cache
                self.lru_cache.set(cache_key, {
                    "value": result,
                    "timestamp": time.time()
                })
                
                return result
            
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                if not self.enabled:
                    return func(*args, **kwargs)
                
                cache_key = self._generate_cache_key(func.__name__, args, kwargs)
                
                cached = self.lru_cache.get(cache_key)
                if cached and time.time() - cached["timestamp"] < ttl_seconds:
                    logger.debug(f"Cache hit: {func.__name__}")
                    return cached["value"]
                
                result = func(*args, **kwargs)
                
                self.lru_cache.set(cache_key, {
                    "value": result,
                    "timestamp": time.time()
                })
                
                return result
            
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper
        
        return decorator
    
    def profile(self, func_name: Optional[str] = None):
        """
        Decorador para profiling de funciones.
        
        Args:
            func_name: Nombre de la función (opcional)
        """
        def decorator(func: Callable) -> Callable:
            name = func_name or func.__name__
            
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                start = time.time()
                try:
                    result = await func(*args, **kwargs)
                    return result
                finally:
                    duration = time.time() - start
                    if name not in self.profiling_data:
                        self.profiling_data[name] = []
                    self.profiling_data[name].append(duration)
                    logger.debug(f"Profiling {name}: {duration:.4f}s")
            
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                start = time.time()
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    duration = time.time() - start
                    if name not in self.profiling_data:
                        self.profiling_data[name] = []
                    self.profiling_data[name].append(duration)
                    logger.debug(f"Profiling {name}: {duration:.4f}s")
            
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper
        
        return decorator
    
    def _generate_cache_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """Genera clave de cache"""
        import hashlib
        import json
        
        key_data = {
            "func": func_name,
            "args": str(args),
            "kwargs": json.dumps(kwargs, sort_keys=True)
        }
        
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def get_profiling_stats(self) -> Dict[str, Dict[str, float]]:
        """Obtiene estadísticas de profiling"""
        stats = {}
        
        for func_name, durations in self.profiling_data.items():
            if durations:
                stats[func_name] = {
                    "count": len(durations),
                    "total": sum(durations),
                    "avg": sum(durations) / len(durations),
                    "min": min(durations),
                    "max": max(durations)
                }
        
        return stats
    
    def clear_profiling(self):
        """Limpia datos de profiling"""
        self.profiling_data.clear()
    
    def optimize_embeddings_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Optimiza generación de embeddings procesando en lotes.
        
        Args:
            texts: Lista de textos
            batch_size: Tamaño del lote
            
        Returns:
            Lista de embeddings
        """
        # En producción, esto procesaría en lotes para mejor performance
        # Por ahora, retorna estructura básica
        return [[] for _ in texts]




