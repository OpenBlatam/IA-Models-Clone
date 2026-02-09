"""
Advanced cache manager for API responses and service results.
Implements LRU cache with size limits and automatic eviction.
"""

from typing import Any, Optional, Callable, TypeVar, OrderedDict
from collections import OrderedDict
import hashlib
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

T = TypeVar('T')


class ResponseCache:
    """
    Gestor de caché para respuestas de API con tamaño limitado.
    Implementa política LRU (Least Recently Used).
    """
    
    def __init__(self, max_size: int = 1000):
        """
        Inicializa el caché.
        
        Args:
            max_size: Tamaño máximo del caché
        """
        self._cache: OrderedDict[str, Any] = OrderedDict()
        self.max_size = max_size
        logger.info(f"ResponseCache initialized with max_size={max_size}")
    
    def get(self, key: str) -> Optional[Any]:
        """
        Obtiene valor del caché.
        
        Args:
            key: Clave de caché
            
        Returns:
            Valor en caché o None si no existe
        """
        if key in self._cache:
            # Mover al final (más reciente)
            self._cache.move_to_end(key)
            logger.debug(f"Cache hit: {key}")
            return self._cache[key]
        logger.debug(f"Cache miss: {key}")
        return None
    
    def set(self, key: str, value: Any) -> None:
        """
        Guarda valor en caché.
        
        Args:
            key: Clave de caché
            value: Valor a guardar
        """
        # Si existe, mover al final
        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            # Si está lleno, eliminar el más antiguo
            if len(self._cache) >= self.max_size:
                oldest_key, _ = self._cache.popitem(last=False)
                logger.debug(f"Cache evicted: {oldest_key}")
        
        self._cache[key] = value
        logger.debug(f"Cache set: {key}")
    
    def delete(self, key: str) -> bool:
        """
        Elimina una entrada del caché.
        
        Args:
            key: Clave a eliminar
            
        Returns:
            True si se eliminó, False si no existía
        """
        if key in self._cache:
            del self._cache[key]
            logger.debug(f"Cache deleted: {key}")
            return True
        return False
    
    def clear(self) -> None:
        """Limpia todo el caché."""
        count = len(self._cache)
        self._cache.clear()
        logger.info(f"Cache cleared: {count} entries removed")
    
    def size(self) -> int:
        """Retorna el tamaño actual del caché."""
        return len(self._cache)
    
    def keys(self) -> list:
        """Retorna lista de claves en el caché."""
        return list(self._cache.keys())


# Instancia global
_global_cache = ResponseCache(max_size=1000)


def get_cache() -> ResponseCache:
    """Obtiene la instancia global del caché."""
    return _global_cache


def cached(
    cache_key_func: Optional[Callable[..., str]] = None,
    use_cache: bool = True
):
    """
    Decorador para cachear resultados de funciones.
    
    Args:
        cache_key_func: Función para generar clave de caché desde argumentos
        use_cache: Si usar caché (default: True)
        
    Usage:
        @cached(lambda identity_id: f"identity_{identity_id}")
        async def get_identity(identity_id: str):
            # ...
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        async def async_wrapper(*args, **kwargs):
            if not use_cache:
                return await func(*args, **kwargs)
            
            # Generar clave
            if cache_key_func:
                cache_key = cache_key_func(*args, **kwargs)
            else:
                # Default: usar nombre de función + argumentos
                key_parts = [func.__name__] + [str(a) for a in args] + [
                    f"{k}:{v}" for k, v in sorted(kwargs.items())
                ]
                cache_key = hashlib.md5("_".join(key_parts).encode()).hexdigest()
            
            # Verificar caché
            cache = get_cache()
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                logger.debug(f"Cache hit for {func.__name__}")
                return cached_result
            
            # Ejecutar función
            result = await func(*args, **kwargs)
            
            # Guardar en caché
            cache.set(cache_key, result)
            logger.debug(f"Cached result for {func.__name__}")
            
            return result
        
        def sync_wrapper(*args, **kwargs):
            if not use_cache:
                return func(*args, **kwargs)
            
            # Generar clave
            if cache_key_func:
                cache_key = cache_key_func(*args, **kwargs)
            else:
                key_parts = [func.__name__] + [str(a) for a in args] + [
                    f"{k}:{v}" for k, v in sorted(kwargs.items())
                ]
                cache_key = hashlib.md5("_".join(key_parts).encode()).hexdigest()
            
            # Verificar caché
            cache = get_cache()
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                logger.debug(f"Cache hit for {func.__name__}")
                return cached_result
            
            # Ejecutar función
            result = func(*args, **kwargs)
            
            # Guardar en caché
            cache.set(cache_key, result)
            logger.debug(f"Cached result for {func.__name__}")
            
            return result
        
        # Retornar wrapper apropiado
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator








