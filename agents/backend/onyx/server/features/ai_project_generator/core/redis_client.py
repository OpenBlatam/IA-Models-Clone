"""
Redis Client - Cliente Redis para cache y state management
==========================================================

Cliente Redis optimizado para:
- Cache distribuido
- State management stateless
- Session storage
- Pub/Sub para eventos
"""

import json
import logging
from typing import Optional, Any, Dict, List
from datetime import timedelta
import redis.asyncio as aioredis
import redis
from redis.exceptions import ConnectionError, TimeoutError

from .microservices_config import get_cache_config

logger = logging.getLogger(__name__)


class RedisClient:
    """Cliente Redis para operaciones síncronas y asíncronas"""
    
    def __init__(self, url: Optional[str] = None, **kwargs):
        """
        Args:
            url: URL de conexión Redis (ej: redis://localhost:6379)
            **kwargs: Argumentos adicionales para Redis
        """
        config = get_cache_config()
        self.url = url or config.get("url", "redis://localhost:6379")
        self.default_ttl = config.get("ttl", 3600)
        
        # Cliente síncrono
        try:
            self.sync_client = redis.from_url(
                self.url,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
                **kwargs
            )
            # Test connection
            self.sync_client.ping()
            logger.info(f"Redis sync client connected to {self.url}")
        except (ConnectionError, TimeoutError) as e:
            logger.warning(f"Redis sync client not available: {e}")
            self.sync_client = None
        
        # Cliente asíncrono (lazy initialization)
        self._async_client: Optional[aioredis.Redis] = None
    
    async def get_async_client(self) -> aioredis.Redis:
        """Obtiene o crea el cliente async"""
        if self._async_client is None:
            try:
                self._async_client = await aioredis.from_url(
                    self.url,
                    decode_responses=True,
                    socket_connect_timeout=5,
                    socket_timeout=5,
                )
                await self._async_client.ping()
                logger.info(f"Redis async client connected to {self.url}")
            except (ConnectionError, TimeoutError) as e:
                logger.warning(f"Redis async client not available: {e}")
                raise
        
        return self._async_client
    
    # Sync methods
    def get(self, key: str) -> Optional[Any]:
        """Obtiene valor del cache"""
        if not self.sync_client:
            return None
        
        try:
            value = self.sync_client.get(key)
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            logger.error(f"Error getting key {key}: {e}")
            return None
    
    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """Establece valor en cache"""
        if not self.sync_client:
            return False
        
        try:
            ttl = ttl or self.default_ttl
            serialized = json.dumps(value)
            return self.sync_client.setex(key, ttl, serialized)
        except Exception as e:
            logger.error(f"Error setting key {key}: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Elimina clave del cache"""
        if not self.sync_client:
            return False
        
        try:
            return bool(self.sync_client.delete(key))
        except Exception as e:
            logger.error(f"Error deleting key {key}: {e}")
            return False
    
    def exists(self, key: str) -> bool:
        """Verifica si una clave existe"""
        if not self.sync_client:
            return False
        
        try:
            return bool(self.sync_client.exists(key))
        except Exception as e:
            logger.error(f"Error checking key {key}: {e}")
            return False
    
    def clear_pattern(self, pattern: str) -> int:
        """Elimina todas las claves que coinciden con el patrón"""
        if not self.sync_client:
            return 0
        
        try:
            keys = self.sync_client.keys(pattern)
            if keys:
                return self.sync_client.delete(*keys)
            return 0
        except Exception as e:
            logger.error(f"Error clearing pattern {pattern}: {e}")
            return 0
    
    # Async methods
    async def aget(self, key: str) -> Optional[Any]:
        """Obtiene valor del cache (async)"""
        try:
            client = await self.get_async_client()
            value = await client.get(key)
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            logger.error(f"Error getting key {key} (async): {e}")
            return None
    
    async def aset(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """Establece valor en cache (async)"""
        try:
            client = await self.get_async_client()
            ttl = ttl or self.default_ttl
            serialized = json.dumps(value)
            return await client.setex(key, ttl, serialized)
        except Exception as e:
            logger.error(f"Error setting key {key} (async): {e}")
            return False
    
    async def adelete(self, key: str) -> bool:
        """Elimina clave del cache (async)"""
        try:
            client = await self.get_async_client()
            return bool(await client.delete(key))
        except Exception as e:
            logger.error(f"Error deleting key {key} (async): {e}")
            return False
    
    async def aexists(self, key: str) -> bool:
        """Verifica si una clave existe (async)"""
        try:
            client = await self.get_async_client()
            return bool(await client.exists(key))
        except Exception as e:
            logger.error(f"Error checking key {key} (async): {e}")
            return False
    
    # Pub/Sub
    async def publish(self, channel: str, message: Dict[str, Any]) -> int:
        """Publica mensaje en canal"""
        try:
            client = await self.get_async_client()
            serialized = json.dumps(message)
            return await client.publish(channel, serialized)
        except Exception as e:
            logger.error(f"Error publishing to channel {channel}: {e}")
            return 0
    
    async def subscribe(self, channel: str):
        """Suscribe a un canal (generator)"""
        try:
            client = await self.get_async_client()
            pubsub = client.pubsub()
            await pubsub.subscribe(channel)
            
            async for message in pubsub.listen():
                if message["type"] == "message":
                    try:
                        data = json.loads(message["data"])
                        yield data
                    except json.JSONDecodeError:
                        logger.warning(f"Invalid JSON in message: {message['data']}")
        except Exception as e:
            logger.error(f"Error subscribing to channel {channel}: {e}")
    
    def close(self):
        """Cierra conexiones"""
        if self.sync_client:
            try:
                self.sync_client.close()
            except Exception as e:
                logger.error(f"Error closing sync client: {e}")
    
    async def aclose(self):
        """Cierra conexiones async"""
        if self._async_client:
            try:
                await self._async_client.close()
                self._async_client = None
            except Exception as e:
                logger.error(f"Error closing async client: {e}")


# Instancia global
_redis_client: Optional[RedisClient] = None


def get_redis_client() -> RedisClient:
    """Obtiene instancia global de Redis client"""
    global _redis_client
    if _redis_client is None:
        _redis_client = RedisClient()
    return _redis_client















