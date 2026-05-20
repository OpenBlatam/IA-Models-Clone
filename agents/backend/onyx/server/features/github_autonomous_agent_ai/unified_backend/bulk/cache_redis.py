"""
Cache Redis para BUL API
=========================
"""

import json
import pickle
from typing import Any, Optional
from datetime import timedelta
import redis
from redis.exceptions import RedisError

try:
    redis_client = redis.Redis(
        host='localhost',
        port=6379,
        db=0,
        decode_responses=False,
        socket_connect_timeout=5,
        socket_timeout=5
    )
    redis_client.ping()
    REDIS_AVAILABLE = True
except (RedisError, ConnectionError, Exception):
    REDIS_AVAILABLE = False
    redis_client = None


class CacheManager:
    """Gestor de cache Redis."""
    
    def __init__(self, default_ttl: int = 3600):
        """
        Inicializa el gestor de cache.
        
        Args:
            default_ttl: TTL por defecto en segundos
        """
        self.default_ttl = default_ttl
        self.available = REDIS_AVAILABLE
    
    def get(self, key: str) -> Optional[Any]:
        """
        Obtiene un valor del cache.
        
        Args:
            key: Clave del cache
            
        Returns:
            Valor deserializado o None
        """
        if not self.available:
            return None
        
        try:
            value = redis_client.get(key)
            if value is None:
                return None
            return pickle.loads(value)
        except Exception as e:
            print(f"Error getting cache key {key}: {e}")
            return None
    
    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """
        Establece un valor en el cache.
        
        Args:
            key: Clave del cache
            value: Valor a cachear
            ttl: TTL en segundos (opcional)
            
        Returns:
            True si se guardó exitosamente
        """
        if not self.available:
            return False
        
        try:
            ttl = ttl or self.default_ttl
            serialized = pickle.dumps(value)
            redis_client.setex(key, ttl, serialized)
            return True
        except Exception as e:
            print(f"Error setting cache key {key}: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """
        Elimina una clave del cache.
        
        Args:
            key: Clave a eliminar
            
        Returns:
            True si se eliminó exitosamente
        """
        if not self.available:
            return False
        
        try:
            redis_client.delete(key)
            return True
        except Exception as e:
            print(f"Error deleting cache key {key}: {e}")
            return False
    
    def exists(self, key: str) -> bool:
        """
        Verifica si una clave existe.
        
        Args:
            key: Clave a verificar
            
        Returns:
            True si existe
        """
        if not self.available:
            return False
        
        try:
            return redis_client.exists(key) > 0
        except Exception:
            return False
    
    def clear_pattern(self, pattern: str) -> int:
        """
        Elimina todas las claves que coincidan con un patrón.
        
        Args:
            pattern: Patrón (ej: "task:*")
            
        Returns:
            Número de claves eliminadas
        """
        if not self.available:
            return 0
        
        try:
            keys = redis_client.keys(pattern)
            if keys:
                return redis_client.delete(*keys)
            return 0
        except Exception as e:
            print(f"Error clearing pattern {pattern}: {e}")
            return 0
    
    def get_stats(self) -> dict:
        """
        Obtiene estadísticas del cache.
        
        Returns:
            Dict con estadísticas
        """
        if not self.available:
            return {
                "available": False,
                "connected": False
            }
        
        try:
            info = redis_client.info()
            return {
                "available": True,
                "connected": True,
                "used_memory": info.get("used_memory_human", "N/A"),
                "connected_clients": info.get("connected_clients", 0),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0)
            }
        except Exception:
            return {
                "available": True,
                "connected": False
            }


cache_manager = CacheManager(default_ttl=3600)


def cache_key(prefix: str, *args) -> str:
    """Genera una clave de cache."""
    parts = [prefix] + [str(arg) for arg in args]
    return ":".join(parts)


def cache_result(ttl: int = 3600, key_prefix: str = "cache"):
    """
    Decorator para cachear resultados de funciones.
    
    Usage:
        @cache_result(ttl=1800, key_prefix="documents")
        def get_document(id: str):
            ...
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            key = cache_key(key_prefix, func.__name__, *args, *sorted(kwargs.items()))
            
            cached = cache_manager.get(key)
            if cached is not None:
                return cached
            
            result = func(*args, **kwargs)
            
            cache_manager.set(key, result, ttl=ttl)
            
            return result
        return wrapper
    return decorator
































