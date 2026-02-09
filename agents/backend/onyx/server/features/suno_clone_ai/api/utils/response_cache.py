"""
Utilidades para caching de respuestas HTTP

Implementa caching en memoria para respuestas frecuentemente accedidas.
Optimizado con orjson para mejor rendimiento.
"""

from functools import wraps
from typing import Callable, Any, Optional
import hashlib
import time

try:
    import orjson
    _use_orjson = True
except ImportError:
    import json
    _use_orjson = False

# Cache en memoria para respuestas
_response_cache: dict = {}


def cache_response(ttl: int = 60, max_size: int = 1000):
    """
    Decorador para cachear respuestas de endpoints.
    
    Args:
        ttl: Tiempo de vida del caché en segundos
        max_size: Tamaño máximo del caché
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Generar clave del caché basada en args y kwargs
            cache_key = _generate_cache_key(func.__name__, args, kwargs)
            
            # Verificar caché
            if cache_key in _response_cache:
                cached_data, timestamp = _response_cache[cache_key]
                if time.time() - timestamp < ttl:
                    return cached_data
            
            # Ejecutar función y cachear resultado
            result = await func(*args, **kwargs)
            
            # Limpiar caché si excede tamaño máximo
            if len(_response_cache) >= max_size:
                _clean_old_cache(ttl)
            
            _response_cache[cache_key] = (result, time.time())
            return result
        
        return wrapper
    return decorator


def _generate_cache_key(func_name: str, args: tuple, kwargs: dict) -> str:
    """
    Genera una clave única para el caché.
    
    Usa orjson si está disponible para mejor rendimiento.
    """
    key_data = {
        "func": func_name,
        "args": str(args),
        "kwargs": kwargs
    }
    
    if _use_orjson:
        key_str = orjson.dumps(key_data, option=orjson.OPT_SORT_KEYS).decode()
    else:
        key_str = json.dumps(key_data, sort_keys=True, default=str)
    
    return hashlib.sha256(key_str.encode()).hexdigest()


def _clean_old_cache(ttl: int) -> None:
    """Limpia entradas antiguas del caché"""
    current_time = time.time()
    keys_to_remove = [
        key for key, (_, timestamp) in _response_cache.items()
        if current_time - timestamp >= ttl
    ]
    for key in keys_to_remove:
        _response_cache.pop(key, None)


def clear_response_cache() -> None:
    """Limpia todo el caché de respuestas"""
    _response_cache.clear()


def get_cache_stats() -> dict:
    """Obtiene estadísticas del caché"""
    return {
        "size": len(_response_cache),
        "keys": list(_response_cache.keys())[:10]  # Primeras 10 claves
    }

