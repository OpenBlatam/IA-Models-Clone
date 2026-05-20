"""
Cache Manager - Gestor de Caché
================================

Sistema de caché para mejorar performance.
"""

import logging
import hashlib
import json
from typing import Any, Optional, Dict, Callable
from datetime import datetime, timedelta
from functools import wraps
import time

logger = logging.getLogger(__name__)


class CacheManager:
    """Gestor de caché simple y eficiente"""
    
    def __init__(self, default_ttl: int = 300):
        """
        Inicializar gestor de caché
        
        Args:
            default_ttl: TTL por defecto en segundos (default: 5 minutos)
        """
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.default_ttl = default_ttl
        logger.info(f"Cache Manager inicializado (default_ttl={default_ttl}s)")
    
    def _generate_key(self, prefix: str, *args, **kwargs) -> str:
        """Generar clave de caché"""
        key_data = {
            "prefix": prefix,
            "args": args,
            "kwargs": sorted(kwargs.items())
        }
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """
        Obtener valor del caché
        
        Args:
            key: Clave del caché
            
        Returns:
            Valor o None si no existe o expiró
        """
        if key not in self.cache:
            return None
        
        entry = self.cache[key]
        expires_at = entry.get("expires_at")
        
        if expires_at and datetime.now() > expires_at:
            del self.cache[key]
            return None
        
        return entry.get("value")
    
    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> None:
        """
        Guardar valor en caché
        
        Args:
            key: Clave del caché
            value: Valor a guardar
            ttl: TTL en segundos (None = usar default)
        """
        ttl = ttl or self.default_ttl
        expires_at = datetime.now() + timedelta(seconds=ttl)
        
        self.cache[key] = {
            "value": value,
            "expires_at": expires_at,
            "created_at": datetime.now()
        }
    
    def delete(self, key: str) -> bool:
        """
        Eliminar valor del caché
        
        Args:
            key: Clave del caché
            
        Returns:
            True si se eliminó, False si no existía
        """
        if key in self.cache:
            del self.cache[key]
            return True
        return False
    
    def clear(self) -> None:
        """Limpiar todo el caché"""
        self.cache.clear()
        logger.info("Caché limpiado")
    
    def cleanup_expired(self) -> int:
        """
        Limpiar entradas expiradas
        
        Returns:
            Número de entradas eliminadas
        """
        now = datetime.now()
        expired_keys = [
            key for key, entry in self.cache.items()
            if entry.get("expires_at") and entry["expires_at"] < now
        ]
        
        for key in expired_keys:
            del self.cache[key]
        
        if expired_keys:
            logger.debug(f"Limpiadas {len(expired_keys)} entradas expiradas del caché")
        
        return len(expired_keys)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Obtener estadísticas del caché
        
        Returns:
            Dict con estadísticas
        """
        now = datetime.now()
        total = len(self.cache)
        expired = sum(
            1 for entry in self.cache.values()
            if entry.get("expires_at") and entry["expires_at"] < now
        )
        
        return {
            "total_entries": total,
            "active_entries": total - expired,
            "expired_entries": expired,
            "default_ttl": self.default_ttl
        }


def cached(
    ttl: Optional[int] = None,
    key_prefix: Optional[str] = None
):
    """
    Decorador para cachear resultados de funciones
    
    Args:
        ttl: TTL en segundos (None = usar default del cache manager)
        key_prefix: Prefijo para la clave (None = usar nombre de función)
    
    Example:
        @cached(ttl=300)
        def expensive_function(arg1, arg2):
            return expensive_computation(arg1, arg2)
    """
    def decorator(func: Callable) -> Callable:
        cache = CacheManager(default_ttl=ttl or 300)
        prefix = key_prefix or func.__name__
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache_key = cache._generate_key(prefix, *args, **kwargs)
            
            cached_value = cache.get(cache_key)
            if cached_value is not None:
                logger.debug(f"Cache hit para {func.__name__}")
                return cached_value
            
            result = func(*args, **kwargs)
            cache.set(cache_key, result, ttl=ttl)
            logger.debug(f"Cache miss para {func.__name__}, resultado guardado")
            
            return result
        
        wrapper.cache = cache
        wrapper.clear_cache = lambda: cache.clear()
        return wrapper
    
    return decorator


def timed_cache(ttl: Optional[int] = None):
    """
    Decorador para cachear con tiempo de ejecución
    
    Args:
        ttl: TTL en segundos
    
    Example:
        @timed_cache(ttl=300)
        def slow_function():
            return slow_computation()
    """
    def decorator(func: Callable) -> Callable:
        cache = CacheManager(default_ttl=ttl or 300)
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache_key = cache._generate_key(func.__name__, *args, **kwargs)
            
            cached_value = cache.get(cache_key)
            if cached_value is not None:
                return cached_value
            
            start_time = time.time()
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            cache.set(cache_key, result, ttl=ttl)
            logger.debug(f"{func.__name__} ejecutado en {execution_time:.3f}s")
            
            return result
        
        wrapper.cache = cache
        return wrapper
    
    return decorator


