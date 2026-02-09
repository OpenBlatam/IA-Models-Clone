"""
Advanced Caching System
=======================

Sistema de caché avanzado con múltiples estrategias.
"""

import time
import hashlib
import pickle
from typing import Any, Optional, Dict, Callable
from collections import OrderedDict
from threading import Lock
import logging

logger = logging.getLogger(__name__)


class LRUCache:
    """
    Least Recently Used (LRU) Cache.
    
    Implementación thread-safe de caché LRU.
    """
    
    def __init__(self, maxsize: int = 128):
        """
        Inicializar caché LRU.
        
        Args:
            maxsize: Tamaño máximo del caché
        """
        self.maxsize = maxsize
        self.cache: OrderedDict = OrderedDict()
        self.lock = Lock()
        self.hits = 0
        self.misses = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Obtener valor del caché."""
        with self.lock:
            if key in self.cache:
                # Mover al final (más reciente)
                value = self.cache.pop(key)
                self.cache[key] = value
                self.hits += 1
                return value
            else:
                self.misses += 1
                return None
    
    def set(self, key: str, value: Any) -> None:
        """Guardar valor en caché."""
        with self.lock:
            if key in self.cache:
                # Actualizar existente
                self.cache.pop(key)
            elif len(self.cache) >= self.maxsize:
                # Eliminar menos reciente
                self.cache.popitem(last=False)
            
            self.cache[key] = value
    
    def clear(self) -> None:
        """Limpiar caché."""
        with self.lock:
            self.cache.clear()
            self.hits = 0
            self.misses = 0
    
    def get_statistics(self) -> Dict[str, Any]:
        """Obtener estadísticas del caché."""
        with self.lock:
            total = self.hits + self.misses
            hit_rate = (self.hits / total * 100) if total > 0 else 0.0
            
            return {
                "size": len(self.cache),
                "maxsize": self.maxsize,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": hit_rate
            }


class TTLCache:
    """
    Time-To-Live (TTL) Cache.
    
    Caché con expiración automática de entradas.
    """
    
    def __init__(self, maxsize: int = 128, ttl_seconds: float = 3600.0):
        """
        Inicializar caché TTL.
        
        Args:
            maxsize: Tamaño máximo
            ttl_seconds: Tiempo de vida en segundos
        """
        self.maxsize = maxsize
        self.ttl_seconds = ttl_seconds
        self.cache: Dict[str, tuple] = {}  # {key: (value, timestamp)}
        self.lock = Lock()
        self.hits = 0
        self.misses = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Obtener valor del caché."""
        with self.lock:
            if key not in self.cache:
                self.misses += 1
                return None
            
            value, timestamp = self.cache[key]
            
            # Verificar TTL
            if time.time() - timestamp > self.ttl_seconds:
                del self.cache[key]
                self.misses += 1
                return None
            
            self.hits += 1
            return value
    
    def set(self, key: str, value: Any) -> None:
        """Guardar valor en caché."""
        with self.lock:
            # Limpiar expirados
            self._cleanup_expired()
            
            # Agregar nuevo
            if len(self.cache) >= self.maxsize:
                # Eliminar más antiguo
                oldest_key = min(
                    self.cache.keys(),
                    key=lambda k: self.cache[k][1]
                )
                del self.cache[oldest_key]
            
            self.cache[key] = (value, time.time())
    
    def _cleanup_expired(self) -> None:
        """Limpiar entradas expiradas."""
        current_time = time.time()
        expired_keys = [
            key for key, (_, timestamp) in self.cache.items()
            if current_time - timestamp > self.ttl_seconds
        ]
        for key in expired_keys:
            del self.cache[key]
    
    def clear(self) -> None:
        """Limpiar caché."""
        with self.lock:
            self.cache.clear()
            self.hits = 0
            self.misses = 0


class CacheManager:
    """
    Gestor de múltiples cachés.
    
    Permite usar diferentes estrategias de caché.
    """
    
    def __init__(self):
        """Inicializar gestor de cachés."""
        self.caches: Dict[str, Any] = {}
        self.default_cache_type = "lru"
    
    def create_cache(
        self,
        name: str,
        cache_type: str = "lru",
        **kwargs
    ) -> Any:
        """
        Crear caché.
        
        Args:
            name: Nombre del caché
            cache_type: Tipo ("lru" o "ttl")
            **kwargs: Parámetros del caché
            
        Returns:
            Caché creado
        """
        if cache_type == "lru":
            cache = LRUCache(**kwargs)
        elif cache_type == "ttl":
            cache = TTLCache(**kwargs)
        else:
            raise ValueError(f"Unknown cache type: {cache_type}")
        
        self.caches[name] = cache
        logger.info(f"Created {cache_type} cache: {name}")
        return cache
    
    def get_cache(self, name: str) -> Optional[Any]:
        """Obtener caché por nombre."""
        return self.caches.get(name)
    
    def get_or_create_cache(
        self,
        name: str,
        cache_type: str = "lru",
        **kwargs
    ) -> Any:
        """Obtener caché o crear si no existe."""
        if name not in self.caches:
            return self.create_cache(name, cache_type, **kwargs)
        return self.caches[name]
    
    def clear_all(self) -> None:
        """Limpiar todos los cachés."""
        for cache in self.caches.values():
            if hasattr(cache, 'clear'):
                cache.clear()
    
    def get_all_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Obtener estadísticas de todos los cachés."""
        return {
            name: cache.get_statistics()
            for name, cache in self.caches.items()
            if hasattr(cache, 'get_statistics')
        }


# Instancia global
_cache_manager: Optional[CacheManager] = None


def get_cache_manager() -> CacheManager:
    """Obtener instancia global del gestor de cachés."""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager


def cached(
    cache_name: str = "default",
    cache_type: str = "lru",
    maxsize: int = 128,
    ttl_seconds: Optional[float] = None
):
    """
    Decorador para cachear resultados de función.
    
    Args:
        cache_name: Nombre del caché
        cache_type: Tipo de caché ("lru" o "ttl")
        maxsize: Tamaño máximo
        ttl_seconds: TTL en segundos (solo para cache_type="ttl")
    
    Usage:
        @cached(cache_name="my_cache", maxsize=256)
        def expensive_function(x, y):
            ...
    """
    def decorator(func: Callable) -> Callable:
        import functools
        
        # Crear o obtener caché
        manager = get_cache_manager()
        kwargs = {"maxsize": maxsize}
        if cache_type == "ttl" and ttl_seconds:
            kwargs["ttl_seconds"] = ttl_seconds
        
        cache = manager.get_or_create_cache(
            cache_name,
            cache_type=cache_type,
            **kwargs
        )
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Crear clave de caché
            key_data = {
                "func": func.__name__,
                "args": args,
                "kwargs": kwargs
            }
            key_str = pickle.dumps(key_data)
            key_hash = hashlib.md5(key_str).hexdigest()
            
            # Verificar caché
            result = cache.get(key_hash)
            if result is not None:
                return result
            
            # Ejecutar función
            result = func(*args, **kwargs)
            
            # Guardar en caché
            cache.set(key_hash, result)
            
            return result
        
        return wrapper
    return decorator






