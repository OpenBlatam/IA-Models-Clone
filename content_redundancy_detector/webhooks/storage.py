"""
Stateless Storage Backend for Webhooks
Supports Redis and in-memory for serverless/microservices
"""

import json
import time
import logging
from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class StorageBackend(ABC):
    """Abstract storage backend for stateless webhook system"""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get value by key"""
        pass
    
    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value with optional TTL"""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> None:
        """Delete key"""
        pass
    
    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists"""
        pass
    
    @abstractmethod
    async def list_keys(self, pattern: str = "*") -> List[str]:
        """List keys matching pattern"""
        pass


class RedisStorageBackend(StorageBackend):
    """Redis backend for production microservices"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379", **kwargs):
        """
        Initialize Redis backend
        
        Args:
            redis_url: Redis connection URL
            **kwargs: Additional Redis client parameters
        """
        self.redis_url = redis_url
        self._redis = None
        self._client_kwargs = kwargs
    
    async def _get_client(self):
        """Lazy initialization of Redis client"""
        if self._redis is None:
            try:
                import redis.asyncio as aioredis
                self._redis = aioredis.from_url(
                    self.redis_url,
                    decode_responses=True,
                    **self._client_kwargs
                )
            except ImportError:
                logger.warning("Redis not available, falling back to in-memory")
                return None
        return self._redis
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from Redis"""
        client = await self._get_client()
        if not client:
            return None
        
        try:
            value = await client.get(key)
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            logger.error(f"Redis GET error for {key}: {e}")
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in Redis with optional TTL"""
        client = await self._get_client()
        if not client:
            return
        
        try:
            serialized = json.dumps(value, default=str)
            if ttl:
                await client.setex(key, ttl, serialized)
            else:
                await client.set(key, serialized)
        except Exception as e:
            logger.error(f"Redis SET error for {key}: {e}")
    
    async def delete(self, key: str) -> None:
        """Delete key from Redis"""
        client = await self._get_client()
        if not client:
            return
        
        try:
            await client.delete(key)
        except Exception as e:
            logger.error(f"Redis DELETE error for {key}: {e}")
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in Redis"""
        client = await self._get_client()
        if not client:
            return False
        
        try:
            return bool(await client.exists(key))
        except Exception as e:
            logger.error(f"Redis EXISTS error for {key}: {e}")
            return False
    
    async def list_keys(self, pattern: str = "*") -> List[str]:
        """List keys matching pattern"""
        client = await self._get_client()
        if not client:
            return []
        
        try:
            return [key async for key in client.scan_iter(match=pattern)]
        except Exception as e:
            logger.error(f"Redis LIST KEYS error: {e}")
            return []
    
    async def close(self) -> None:
        """Close Redis connection"""
        if self._redis:
            await self._redis.aclose()
            self._redis = None


class InMemoryStorageBackend(StorageBackend):
    """In-memory backend for serverless/local development"""
    
    def __init__(self):
        """Initialize in-memory storage"""
        self._storage: Dict[str, tuple] = {}  # key -> (value, expire_time)
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from memory"""
        if key not in self._storage:
            return None
        
        value, expire_time = self._storage[key]
        
        # Check expiration
        if expire_time and time.time() > expire_time:
            del self._storage[key]
            return None
        
        return value
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in memory with optional TTL"""
        expire_time = None
        if ttl:
            expire_time = time.time() + ttl
        
        self._storage[key] = (value, expire_time)
    
    async def delete(self, key: str) -> None:
        """Delete key from memory"""
        self._storage.pop(key, None)
    
    async def exists(self, key: str) -> bool:
        """Check if key exists"""
        if key not in self._storage:
            return False
        
        # Check expiration
        value, expire_time = self._storage[key]
        if expire_time and time.time() > expire_time:
            del self._storage[key]
            return False
        
        return True
    
    async def list_keys(self, pattern: str = "*") -> List[str]:
        """List keys matching pattern (simple implementation)"""
        import fnmatch
        keys = list(self._storage.keys())
        
        # Clean expired keys
        now = time.time()
        expired = [
            k for k, (_, exp) in self._storage.items()
            if exp and now > exp
        ]
        for k in expired:
            del self._storage[k]
        
        if pattern == "*":
            return keys
        
        return [k for k in keys if fnmatch.fnmatch(k, pattern)]


class StorageFactory:
    """Factory for creating storage backends"""
    
    @staticmethod
    def create(storage_type: str = "auto", **kwargs) -> StorageBackend:
        """
        Create storage backend
        
        Args:
            storage_type: "redis", "memory", or "auto"
            **kwargs: Backend-specific configuration
        
        Returns:
            StorageBackend instance
        """
        if storage_type == "redis" or (storage_type == "auto" and "redis_url" in kwargs):
            redis_url = kwargs.get("redis_url", "redis://localhost:6379")
            return RedisStorageBackend(redis_url, **kwargs)
        
        # Default to in-memory for serverless/local
        return InMemoryStorageBackend()






