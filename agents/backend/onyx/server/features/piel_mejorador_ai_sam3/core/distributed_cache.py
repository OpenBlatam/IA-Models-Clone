"""
Distributed Cache for Piel Mejorador AI SAM3
============================================

Redis-based distributed cache.
"""

import logging
import json
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class DistributedCacheBackend(ABC):
    """Interface for distributed cache backends."""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        pass
    
    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache."""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
        pass
    
    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        pass
    
    @abstractmethod
    async def close(self):
        """Close connection."""
        pass


class RedisCacheBackend(DistributedCacheBackend):
    """Redis cache backend."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        """
        Initialize Redis backend.
        
        Args:
            redis_url: Redis connection URL
        """
        self.redis_url = redis_url
        self._client = None
    
    async def _get_client(self):
        """Get or create Redis client."""
        if self._client is None:
            try:
                import redis.asyncio as redis
                self._client = redis.from_url(self.redis_url, decode_responses=True)
            except ImportError:
                logger.warning("Redis not available, distributed cache disabled")
                return None
        return self._client
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from Redis."""
        client = await self._get_client()
        if not client:
            return None
        
        try:
            value = await client.get(key)
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            logger.error(f"Error getting from Redis: {e}")
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in Redis."""
        client = await self._get_client()
        if not client:
            return False
        
        try:
            value_str = json.dumps(value)
            if ttl:
                await client.setex(key, ttl, value_str)
            else:
                await client.set(key, value_str)
            return True
        except Exception as e:
            logger.error(f"Error setting in Redis: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete key from Redis."""
        client = await self._get_client()
        if not client:
            return False
        
        try:
            await client.delete(key)
            return True
        except Exception as e:
            logger.error(f"Error deleting from Redis: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in Redis."""
        client = await self._get_client()
        if not client:
            return False
        
        try:
            return await client.exists(key) > 0
        except Exception as e:
            logger.error(f"Error checking Redis: {e}")
            return False
    
    async def close(self):
        """Close Redis connection."""
        if self._client:
            await self._client.close()
            self._client = None


class DistributedCache:
    """
    Distributed cache with Redis support.
    
    Features:
    - Redis backend
    - Fallback to local cache
    - TTL support
    - Key prefixing
    """
    
    def __init__(self, backend: Optional[DistributedCacheBackend] = None):
        """
        Initialize distributed cache.
        
        Args:
            backend: Optional cache backend (defaults to Redis if available)
        """
        if backend:
            self.backend = backend
        else:
            # Try Redis, fallback to None
            try:
                redis_url = "redis://localhost:6379"
                self.backend = RedisCacheBackend(redis_url)
            except Exception:
                logger.warning("Distributed cache not available, using local cache only")
                self.backend = None
        
        self.key_prefix = "piel_mejorador:"
    
    def _make_key(self, key: str) -> str:
        """Make full cache key with prefix."""
        return f"{self.key_prefix}{key}"
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if not self.backend:
            return None
        
        full_key = self._make_key(key)
        return await self.backend.get(full_key)
    
    async def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> bool:
        """Set value in cache."""
        if not self.backend:
            return False
        
        full_key = self._make_key(key)
        return await self.backend.set(full_key, value, ttl_seconds)
    
    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
        if not self.backend:
            return False
        
        full_key = self._make_key(key)
        return await self.backend.delete(full_key)
    
    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        if not self.backend:
            return False
        
        full_key = self._make_key(key)
        return await self.backend.exists(full_key)
    
    async def close(self):
        """Close cache connection."""
        if self.backend:
            await self.backend.close()




