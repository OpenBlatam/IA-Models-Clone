"""
Caching Utilities
=================
"""

from typing import Optional, Any
from functools import wraps
import hashlib
import json
from cachetools import TTLCache

# Cache con TTL de 1 hora
cache = TTLCache(maxsize=1000, ttl=3600)


def cache_key(*args, **kwargs) -> str:
    """Generar clave de cache."""
    key_data = {
        "args": args,
        "kwargs": sorted(kwargs.items())
    }
    key_str = json.dumps(key_data, sort_keys=True, default=str)
    return hashlib.md5(key_str.encode()).hexdigest()


def cached(ttl: int = 3600):
    """Decorator para cachear resultados."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            key = f"{func.__name__}:{cache_key(*args, **kwargs)}"
            
            # Verificar cache
            if key in cache:
                return cache[key]
            
            # Ejecutar función
            result = await func(*args, **kwargs)
            
            # Guardar en cache
            cache[key] = result
            
            return result
        return wrapper
    return decorator


def get_cached_response(key: str) -> Optional[Any]:
    """
    Obtener valor del cache.
    
    Args:
        key: Clave de cache
        
    Returns:
        Valor cacheado o None
    """
    return cache.get(key)


def set_cached_response(key: str, value: Any, ttl: Optional[int] = None) -> None:
    """
    Guardar valor en cache.
    
    Args:
        key: Clave de cache
        value: Valor a guardar
        ttl: Time to live en segundos (opcional, usa el TTL del cache)
    """
    cache[key] = value


def clear_cache():
    """Limpiar cache."""
    cache.clear()
