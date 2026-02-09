"""
Caché Distribuido con Redis

Proporciona:
- Caché distribuido entre múltiples instancias
- TTL configurable
- Invalidación de caché
- Patrones de caché (cache-aside, write-through)
"""

import logging
import json
import pickle
from typing import Optional, Any, Dict, List
from datetime import timedelta
import hashlib

logger = logging.getLogger(__name__)

# Intentar importar Redis
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("Redis not available, using in-memory fallback")


class DistributedCache:
    """Caché distribuido usando Redis"""
    
    def __init__(
        self,
        redis_url: Optional[str] = None,
        default_ttl: int = 3600,
        key_prefix: str = "suno_clone:"
    ):
        """
        Args:
            redis_url: URL de conexión a Redis
            default_ttl: TTL por defecto en segundos
            key_prefix: Prefijo para las claves
        """
        self.default_ttl = default_ttl
        self.key_prefix = key_prefix
        self.redis_client = None
        self._fallback_cache: Dict[str, Any] = {}
        
        if REDIS_AVAILABLE:
            try:
                if redis_url:
                    self.redis_client = redis.from_url(redis_url, decode_responses=False)
                else:
                    self.redis_client = redis.Redis(
                        host='localhost',
                        port=6379,
                        db=0,
                        decode_responses=False
                    )
                # Test connection
                self.redis_client.ping()
                logger.info("Redis connection established")
            except Exception as e:
                logger.warning(f"Could not connect to Redis: {e}, using fallback")
                self.redis_client = None
        else:
            logger.warning("Redis not available, using in-memory cache")
    
    def _make_key(self, key: str) -> str:
        """Crea una clave completa con prefijo"""
        return f"{self.key_prefix}{key}"
    
    def _serialize(self, value: Any) -> bytes:
        """Serializa un valor para almacenamiento"""
        try:
            return pickle.dumps(value)
        except Exception as e:
            logger.error(f"Error serializing value: {e}")
            raise
    
    def _deserialize(self, data: bytes) -> Any:
        """Deserializa un valor desde almacenamiento"""
        try:
            return pickle.loads(data)
        except Exception as e:
            logger.error(f"Error deserializing value: {e}")
            raise
    
    def get(self, key: str) -> Optional[Any]:
        """Obtiene un valor del caché"""
        full_key = self._make_key(key)
        
        if self.redis_client:
            try:
                data = self.redis_client.get(full_key)
                if data:
                    return self._deserialize(data)
            except Exception as e:
                logger.error(f"Error getting from Redis: {e}")
        
        # Fallback a caché en memoria
        return self._fallback_cache.get(full_key)
    
    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """
        Almacena un valor en el caché
        
        Args:
            key: Clave
            value: Valor
            ttl: Tiempo de vida en segundos (None = default_ttl)
        
        Returns:
            True si se almacenó correctamente
        """
        full_key = self._make_key(key)
        ttl = ttl or self.default_ttl
        
        if self.redis_client:
            try:
                data = self._serialize(value)
                self.redis_client.setex(full_key, ttl, data)
                return True
            except Exception as e:
                logger.error(f"Error setting in Redis: {e}")
                return False
        
        # Fallback a caché en memoria
        self._fallback_cache[full_key] = value
        return True
    
    def delete(self, key: str) -> bool:
        """Elimina una clave del caché"""
        full_key = self._make_key(key)
        
        if self.redis_client:
            try:
                return bool(self.redis_client.delete(full_key))
            except Exception as e:
                logger.error(f"Error deleting from Redis: {e}")
        
        # Fallback
        if full_key in self._fallback_cache:
            del self._fallback_cache[full_key]
            return True
        return False
    
    def exists(self, key: str) -> bool:
        """Verifica si una clave existe"""
        full_key = self._make_key(key)
        
        if self.redis_client:
            try:
                return bool(self.redis_client.exists(full_key))
            except Exception as e:
                logger.error(f"Error checking existence in Redis: {e}")
        
        return full_key in self._fallback_cache
    
    def clear_pattern(self, pattern: str) -> int:
        """
        Elimina todas las claves que coincidan con un patrón
        
        Args:
            pattern: Patrón de búsqueda (ej: "user:*")
        
        Returns:
            Número de claves eliminadas
        """
        full_pattern = self._make_key(pattern)
        deleted = 0
        
        if self.redis_client:
            try:
                keys = self.redis_client.keys(full_pattern)
                if keys:
                    deleted = self.redis_client.delete(*keys)
            except Exception as e:
                logger.error(f"Error clearing pattern in Redis: {e}")
        
        # Fallback
        keys_to_delete = [k for k in self._fallback_cache.keys() if full_pattern.replace('*', '') in k]
        for key in keys_to_delete:
            del self._fallback_cache[key]
            deleted += len(keys_to_delete)
        
        return deleted
    
    def increment(self, key: str, amount: int = 1) -> int:
        """Incrementa un valor numérico"""
        full_key = self._make_key(key)
        
        if self.redis_client:
            try:
                return self.redis_client.incrby(full_key, amount)
            except Exception as e:
                logger.error(f"Error incrementing in Redis: {e}")
        
        # Fallback
        current = self._fallback_cache.get(full_key, 0)
        new_value = current + amount
        self._fallback_cache[full_key] = new_value
        return new_value
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del caché"""
        stats = {
            "type": "redis" if self.redis_client else "memory",
            "key_prefix": self.key_prefix,
            "default_ttl": self.default_ttl
        }
        
        if self.redis_client:
            try:
                info = self.redis_client.info()
                stats.update({
                    "connected_clients": info.get("connected_clients", 0),
                    "used_memory_human": info.get("used_memory_human", "0B"),
                    "keyspace_hits": info.get("keyspace_hits", 0),
                    "keyspace_misses": info.get("keyspace_misses", 0)
                })
            except Exception as e:
                logger.error(f"Error getting Redis stats: {e}")
        else:
            stats["memory_keys"] = len(self._fallback_cache)
        
        return stats


# Instancia global
_distributed_cache: Optional[DistributedCache] = None


def get_distributed_cache(redis_url: Optional[str] = None) -> DistributedCache:
    """Obtiene la instancia global del caché distribuido"""
    global _distributed_cache
    if _distributed_cache is None:
        _distributed_cache = DistributedCache(redis_url=redis_url)
    return _distributed_cache

