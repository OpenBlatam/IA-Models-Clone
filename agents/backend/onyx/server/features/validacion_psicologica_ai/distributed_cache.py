"""
Caché Distribuido
=================
Sistema de caché distribuido con Redis
"""

from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import structlog
import json
import hashlib

logger = structlog.get_logger()

try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("redis not available, using in-memory cache")


class DistributedCache:
    """Caché distribuido"""
    
    def __init__(
        self,
        redis_url: Optional[str] = None,
        default_ttl: int = 3600
    ):
        """
        Inicializar caché
        
        Args:
            redis_url: URL de Redis (opcional)
            default_ttl: TTL por defecto en segundos
        """
        self.default_ttl = default_ttl
        self._redis_client: Optional[redis.Redis] = None
        self._fallback_cache: Dict[str, Dict[str, Any]] = {}
        
        if REDIS_AVAILABLE and redis_url:
            try:
                self._redis_client = redis.from_url(redis_url)
                logger.info("DistributedCache initialized with Redis", url=redis_url)
            except Exception as e:
                logger.warning("Failed to connect to Redis, using fallback", error=str(e))
                self._redis_client = None
        else:
            logger.info("DistributedCache initialized with in-memory fallback")
    
    def _generate_key(self, prefix: str, *args) -> str:
        """Generar clave de caché"""
        key_parts = [prefix] + [str(arg) for arg in args]
        key_string = ":".join(key_parts)
        return f"psych_val:{key_string}"
    
    async def get(
        self,
        key: str
    ) -> Optional[Any]:
        """
        Obtener valor del caché
        
        Args:
            key: Clave del caché
            
        Returns:
            Valor o None
        """
        if self._redis_client:
            try:
                value = await self._redis_client.get(key)
                if value:
                    return json.loads(value)
            except Exception as e:
                logger.error("Error getting from Redis", error=str(e))
        
        # Fallback a caché en memoria
        if key in self._fallback_cache:
            entry = self._fallback_cache[key]
            if entry["expires_at"] > datetime.utcnow():
                return entry["value"]
            else:
                del self._fallback_cache[key]
        
        return None
    
    async def set(
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
            ttl: Tiempo de vida en segundos (opcional)
        """
        ttl = ttl or self.default_ttl
        
        if self._redis_client:
            try:
                json_value = json.dumps(value, default=str)
                await self._redis_client.setex(key, ttl, json_value)
                return
            except Exception as e:
                logger.error("Error setting in Redis", error=str(e))
        
        # Fallback a caché en memoria
        expires_at = datetime.utcnow() + timedelta(seconds=ttl)
        self._fallback_cache[key] = {
            "value": value,
            "expires_at": expires_at
        }
    
    async def delete(self, key: str) -> None:
        """Eliminar entrada del caché"""
        if self._redis_client:
            try:
                await self._redis_client.delete(key)
            except Exception as e:
                logger.error("Error deleting from Redis", error=str(e))
        
        if key in self._fallback_cache:
            del self._fallback_cache[key]
    
    async def clear_pattern(self, pattern: str) -> int:
        """
        Eliminar entradas que coincidan con patrón
        
        Args:
            pattern: Patrón a buscar
            
        Returns:
            Número de entradas eliminadas
        """
        deleted = 0
        
        if self._redis_client:
            try:
                keys = await self._redis_client.keys(pattern)
                if keys:
                    deleted = await self._redis_client.delete(*keys)
            except Exception as e:
                logger.error("Error clearing pattern in Redis", error=str(e))
        
        # Limpiar fallback
        keys_to_delete = [k for k in self._fallback_cache.keys() if pattern in k]
        for key in keys_to_delete:
            del self._fallback_cache[key]
            deleted += 1
        
        return deleted
    
    async def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del caché"""
        stats = {
            "backend": "redis" if self._redis_client else "memory",
            "default_ttl": self.default_ttl
        }
        
        if self._redis_client:
            try:
                info = await self._redis_client.info("memory")
                stats["redis_memory"] = info.get("used_memory_human", "N/A")
            except Exception as e:
                logger.error("Error getting Redis stats", error=str(e))
        
        stats["fallback_cache_size"] = len(self._fallback_cache)
        
        return stats


# Instancia global del caché distribuido
distributed_cache = DistributedCache()




