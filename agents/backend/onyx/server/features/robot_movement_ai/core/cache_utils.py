"""
Cache Utilities
===============

Utilidades para sistemas de cache avanzados.
"""

from typing import Any, Optional, Callable, Dict, Tuple
from functools import wraps, lru_cache
import time
import hashlib
import json


class TTLCache:
    """Cache con Time To Live."""
    
    def __init__(self, ttl: float = 3600.0, maxsize: int = 128):
        """
        Inicializar cache TTL.
        
        Args:
            ttl: Time to live en segundos
            maxsize: Tamaño máximo del cache
        """
        self.ttl = ttl
        self.maxsize = maxsize
        self._cache: Dict[str, Tuple[Any, float]] = {}
    
    def get(self, key: str) -> Optional[Any]:
        """
        Obtener valor del cache.
        
        Args:
            key: Clave del cache
        
        Returns:
            Valor o None si expiró/no existe
        """
        if key not in self._cache:
            return None
        
        value, timestamp = self._cache[key]
        
        if time.time() - timestamp > self.ttl:
            del self._cache[key]
            return None
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """
        Guardar valor en cache.
        
        Args:
            key: Clave del cache
            value: Valor a guardar
        """
        if len(self._cache) >= self.maxsize:
            # Eliminar el más antiguo
            oldest_key = min(
                self._cache.keys(),
                key=lambda k: self._cache[k][1]
            )
            del self._cache[oldest_key]
        
        self._cache[key] = (value, time.time())
    
    def clear(self) -> None:
        """Limpiar cache."""
        self._cache.clear()
    
    def size(self) -> int:
        """Obtener tamaño del cache."""
        return len(self._cache)


def cache_key(*args, **kwargs) -> str:
    """
    Generar clave de cache desde argumentos.
    
    Args:
        *args: Argumentos posicionales
        **kwargs: Argumentos nombrados
    
    Returns:
        Clave hash
    """
    key_data = {
        "args": args,
        "kwargs": sorted(kwargs.items())
    }
    key_str = json.dumps(key_data, sort_keys=True, default=str)
    return hashlib.md5(key_str.encode()).hexdigest()


def cached_ttl(ttl: float = 3600.0, maxsize: int = 128):
    """
    Decorador para cachear resultados con TTL.
    
    Args:
        ttl: Time to live en segundos
        maxsize: Tamaño máximo del cache
    
    Returns:
        Decorador
    """
    cache = TTLCache(ttl=ttl, maxsize=maxsize)
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            key = cache_key(*args, **kwargs)
            cached_value = cache.get(key)
            
            if cached_value is not None:
                return cached_value
            
            result = func(*args, **kwargs)
            cache.set(key, result)
            return result
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            key = cache_key(*args, **kwargs)
            cached_value = cache.get(key)
            
            if cached_value is not None:
                return cached_value
            
            result = await func(*args, **kwargs)
            cache.set(key, result)
            return result
        
        if hasattr(func, '__code__') and func.__code__.co_flags & 0x80:  # CO_COROUTINE
            return async_wrapper
        return wrapper
    
    return decorator


def memoize(maxsize: int = 128):
    """
    Decorador memoize simple.
    
    Args:
        maxsize: Tamaño máximo del cache
    
    Returns:
        Decorador
    """
    return lru_cache(maxsize=maxsize)

