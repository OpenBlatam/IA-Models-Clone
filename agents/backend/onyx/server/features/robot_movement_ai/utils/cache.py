"""
Cache Utilities - Utilidades de caché
======================================

Sistema de caché con TTL y invalidación.
"""

import time
import hashlib
import json
from typing import Any, Optional, Callable, Dict
from functools import wraps
from threading import Lock
import logging

logger = logging.getLogger(__name__)


class TTLCache:
    """
    Cache con TTL (Time To Live).
    
    Thread-safe cache con expiración automática.
    """
    
    def __init__(self, default_ttl: float = 300.0, max_size: Optional[int] = None):
        """
        Inicializar cache.
        
        Args:
            default_ttl: TTL por defecto en segundos
            max_size: Tamaño máximo del cache (None = ilimitado)
        """
        self.default_ttl = default_ttl
        self.max_size = max_size
        self._cache: Dict[str, tuple] = {}
        self._lock = Lock()
        self._access_times: Dict[str, float] = {}
    
    def _generate_key(self, *args, **kwargs) -> str:
        """Generar clave de cache desde argumentos."""
        key_data = {
            'args': args,
            'kwargs': sorted(kwargs.items())
        }
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.sha256(key_str.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """
        Obtener valor del cache.
        
        Args:
            key: Clave del cache
        
        Returns:
            Valor o None si no existe o expiró
        """
        with self._lock:
            if key not in self._cache:
                return None
            
            value, expiry_time = self._cache[key]
            
            if time.time() > expiry_time:
                del self._cache[key]
                if key in self._access_times:
                    del self._access_times[key]
                return None
            
            self._access_times[key] = time.time()
            return value
    
    def set(self, key: str, value: Any, ttl: Optional[float] = None):
        """
        Establecer valor en cache.
        
        Args:
            key: Clave del cache
            value: Valor a cachear
            ttl: TTL en segundos (None = usar default)
        """
        ttl = ttl or self.default_ttl
        expiry_time = time.time() + ttl
        
        with self._lock:
            if self.max_size and len(self._cache) >= self.max_size:
                self._evict_lru()
            
            self._cache[key] = (value, expiry_time)
            self._access_times[key] = time.time()
    
    def _evict_lru(self):
        """Eliminar entrada menos recientemente usada."""
        if not self._access_times:
            self._cache.clear()
            return
        
        lru_key = min(self._access_times.items(), key=lambda x: x[1])[0]
        del self._cache[lru_key]
        del self._access_times[lru_key]
    
    def delete(self, key: str):
        """Eliminar entrada del cache."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
            if key in self._access_times:
                del self._access_times[key]
    
    def clear(self):
        """Limpiar todo el cache."""
        with self._lock:
            self._cache.clear()
            self._access_times.clear()
    
    def cleanup_expired(self):
        """Limpiar entradas expiradas."""
        current_time = time.time()
        with self._lock:
            expired_keys = [
                key for key, (_, expiry_time) in self._cache.items()
                if current_time > expiry_time
            ]
            for key in expired_keys:
                del self._cache[key]
                if key in self._access_times:
                    del self._access_times[key]
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del cache."""
        with self._lock:
            self.cleanup_expired()
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "default_ttl": self.default_ttl,
                "usage_percent": (len(self._cache) / self.max_size * 100) if self.max_size else None
            }


def cached(ttl: float = 300.0, key_func: Optional[Callable] = None):
    """
    Decorador para cachear resultados de funciones.
    
    Args:
        ttl: TTL en segundos
        key_func: Función personalizada para generar clave (opcional)
    
    Example:
        @cached(ttl=60.0)
        def expensive_operation(x, y):
            return x + y
    """
    cache = TTLCache(default_ttl=ttl)
    
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = cache._generate_key(*args, **kwargs)
            
            result = cache.get(cache_key)
            if result is not None:
                logger.debug(f"Cache hit for {func.__name__}")
                return result
            
            logger.debug(f"Cache miss for {func.__name__}")
            result = func(*args, **kwargs)
            cache.set(cache_key, result, ttl)
            return result
        
        wrapper.cache = cache
        return wrapper
    
    return decorator


def async_cached(ttl: float = 300.0, key_func: Optional[Callable] = None):
    """
    Decorador para cachear resultados de funciones async.
    
    Args:
        ttl: TTL en segundos
        key_func: Función personalizada para generar clave (opcional)
    """
    cache = TTLCache(default_ttl=ttl)
    
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = cache._generate_key(*args, **kwargs)
            
            result = cache.get(cache_key)
            if result is not None:
                logger.debug(f"Cache hit for {func.__name__}")
                return result
            
            logger.debug(f"Cache miss for {func.__name__}")
            result = await func(*args, **kwargs)
            cache.set(cache_key, result, ttl)
            return result
        
        wrapper.cache = cache
        return wrapper
    
    return decorator

