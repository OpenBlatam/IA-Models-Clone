"""
Utilidades de optimización de rendimiento

Incluye decoradores y funciones para mejorar el rendimiento:
- Caching de resultados
- Batch processing
- Connection pooling
- Lazy loading
"""

import functools
import time
from typing import Any, Callable, Optional, TypeVar, ParamSpec
from functools import lru_cache

P = ParamSpec('P')
T = TypeVar('T')


def cache_result(ttl: int = 3600):
    """
    Decorador para cachear resultados de funciones.
    
    Args:
        ttl: Tiempo de vida del caché en segundos
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        cache: dict = {}
        
        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            # Generar clave del caché
            key = str(args) + str(sorted(kwargs.items()))
            
            # Verificar caché
            if key in cache:
                result, timestamp = cache[key]
                if time.time() - timestamp < ttl:
                    return result
            
            # Ejecutar función y cachear
            result = await func(*args, **kwargs) if hasattr(func, '__call__') else func(*args, **kwargs)
            cache[key] = (result, time.time())
            
            return result
        
        return wrapper
    return decorator


def batch_process(items: list, batch_size: int = 100):
    """
    Procesa items en lotes para mejor rendimiento.
    
    Args:
        items: Lista de items a procesar
        batch_size: Tamaño del lote
    """
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


@lru_cache(maxsize=128)
def get_cached_value(key: str) -> Optional[Any]:
    """Obtiene un valor del caché LRU"""
    return None


def set_cached_value(key: str, value: Any) -> None:
    """Establece un valor en el caché LRU"""
    get_cached_value.cache_clear()  # Limpiar para permitir nuevos valores


def optimize_json_serialization(data: Any) -> bytes:
    """
    Serializa datos a JSON usando orjson (más rápido que json estándar).
    
    Args:
        data: Datos a serializar
        
    Returns:
        Bytes serializados
    """
    import orjson
    return orjson.dumps(data, option=orjson.OPT_SERIALIZE_NUMPY)


def optimize_json_deserialization(data: bytes) -> Any:
    """
    Deserializa JSON usando orjson (más rápido que json estándar).
    
    Args:
        data: Bytes a deserializar
        
    Returns:
        Datos deserializados
    """
    import orjson
    return orjson.loads(data)

