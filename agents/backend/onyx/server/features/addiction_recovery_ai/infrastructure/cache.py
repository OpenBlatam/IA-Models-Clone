"""
Cache Service Implementations
Provides implementations for different cache backends
"""

import logging
from typing import Any, Optional
import json

from core.interfaces import ICacheService, IServiceFactory
from config.aws_settings import get_aws_settings

logger = logging.getLogger(__name__)


class RedisCacheService(ICacheService):
    """Redis implementation of cache service"""
    
    def __init__(self):
        self.settings = get_aws_settings()
        self._client = None
    
    @property
    def client(self):
        """Lazy initialization of Redis client"""
        if self._client is None:
            import redis
            self._client = redis.Redis(
                host=self.settings.redis_endpoint or "localhost",
                port=self.settings.redis_port,
                decode_responses=True
            )
        return self._client
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from Redis"""
        try:
            value = self.client.get(key)
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            logger.error(f"Error getting from Redis: {str(e)}")
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in Redis"""
        try:
            serialized = json.dumps(value)
            if ttl:
                self.client.setex(key, ttl, serialized)
            else:
                self.client.set(key, serialized)
        except Exception as e:
            logger.error(f"Error setting in Redis: {str(e)}")
    
    async def delete(self, key: str) -> None:
        """Delete key from Redis"""
        try:
            self.client.delete(key)
        except Exception as e:
            logger.error(f"Error deleting from Redis: {str(e)}")
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in Redis"""
        try:
            return bool(self.client.exists(key))
        except Exception as e:
            logger.error(f"Error checking existence in Redis: {str(e)}")
            return False


class InMemoryCacheService(ICacheService):
    """In-memory cache implementation for local development"""
    
    def __init__(self):
        self._cache: dict = {}
        self._ttl: dict = {}
        import time
        self._time = time
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from in-memory cache"""
        if key in self._cache:
            # Check TTL
            if key in self._ttl:
                if self._time.time() > self._ttl[key]:
                    del self._cache[key]
                    del self._ttl[key]
                    return None
            return self._cache[key]
        return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in in-memory cache"""
        self._cache[key] = value
        if ttl:
            self._ttl[key] = self._time.time() + ttl
    
    async def delete(self, key: str) -> None:
        """Delete key from in-memory cache"""
        self._cache.pop(key, None)
        self._ttl.pop(key, None)
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in in-memory cache"""
        return key in self._cache


class CacheServiceFactory(IServiceFactory):
    """Factory for creating cache services"""
    
    @staticmethod
    def create(backend: str = "redis") -> ICacheService:
        """
        Create cache service based on backend type
        
        Args:
            backend: Cache backend type (redis, memory)
            
        Returns:
            Cache service instance
        """
        if backend == "redis":
            return RedisCacheService()
        elif backend == "memory":
            return InMemoryCacheService()
        else:
            raise ValueError(f"Unsupported cache backend: {backend}")
    
    def create_cache_service(self) -> ICacheService:
        """Create cache service (factory method)"""
        settings = get_aws_settings()
        backend = "redis" if settings.redis_endpoint else "memory"
        return self.create(backend)
    
    def create_storage_service(self):
        """Not implemented in cache factory"""
        raise NotImplementedError
    
    def create_file_storage_service(self):
        """Not implemented in cache factory"""
        raise NotImplementedError
    
    def create_message_queue_service(self):
        """Not implemented in cache factory"""
        raise NotImplementedError
    
    def create_notification_service(self):
        """Not implemented in cache factory"""
        raise NotImplementedError
    
    def create_metrics_service(self):
        """Not implemented in cache factory"""
        raise NotImplementedError
    
    def create_tracing_service(self):
        """Not implemented in cache factory"""
        raise NotImplementedError
    
    def create_authentication_service(self):
        """Not implemented in cache factory"""
        raise NotImplementedError















