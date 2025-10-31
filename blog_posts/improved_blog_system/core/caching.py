"""
Caching service using Redis
"""

import json
import asyncio
from typing import Optional, Any, List
import redis.asyncio as redis
from functools import wraps


class CacheService:
    """Redis-based caching service."""
    
    def __init__(self, redis_url: str):
        self.redis_url = redis_url
        self.redis = None
        self._connection_lock = asyncio.Lock()
    
    async def _get_connection(self) -> redis.Redis:
        """Get Redis connection with lazy initialization."""
        if self.redis is None:
            async with self._connection_lock:
                if self.redis is None:
                    self.redis = redis.from_url(self.redis_url, decode_responses=True)
        return self.redis
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        try:
            redis_client = await self._get_connection()
            value = await redis_client.get(key)
            if value:
                return json.loads(value)
            return None
        except Exception:
            # If Redis is unavailable, return None (graceful degradation)
            return None
    
    async def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """Set value in cache with TTL."""
        try:
            redis_client = await self._get_connection()
            serialized_value = json.dumps(value, default=str)
            await redis_client.setex(key, ttl, serialized_value)
            return True
        except Exception:
            # If Redis is unavailable, continue without caching
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete value from cache."""
        try:
            redis_client = await self._get_connection()
            await redis_client.delete(key)
            return True
        except Exception:
            return False
    
    async def delete_pattern(self, pattern: str) -> int:
        """Delete all keys matching pattern."""
        try:
            redis_client = await self._get_connection()
            keys = await redis_client.keys(pattern)
            if keys:
                return await redis_client.delete(*keys)
            return 0
        except Exception:
            return 0
    
    async def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate all keys matching pattern."""
        return await self.delete_pattern(pattern)
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        try:
            redis_client = await self._get_connection()
            return await redis_client.exists(key) > 0
        except Exception:
            return False
    
    async def get_ttl(self, key: str) -> int:
        """Get TTL for a key."""
        try:
            redis_client = await self._get_connection()
            return await redis_client.ttl(key)
        except Exception:
            return -1
    
    async def increment(self, key: str, amount: int = 1) -> int:
        """Increment a numeric value in cache."""
        try:
            redis_client = await self._get_connection()
            return await redis_client.incrby(key, amount)
        except Exception:
            return 0
    
    async def close(self):
        """Close Redis connection."""
        if self.redis:
            await self.redis.close()
            self.redis = None


def cache_result(ttl: int = 3600, key_prefix: str = ""):
    """Decorator for caching function results."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = f"{key_prefix}{func.__name__}:{hash(str(args) + str(kwargs))}"
            
            # Try to get from cache
            cache_service = CacheService("redis://localhost:6379")  # This should be injected
            cached_result = await cache_service.get(cache_key)
            
            if cached_result is not None:
                return cached_result
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Cache result
            await cache_service.set(cache_key, result, ttl)
            
            return result
        return wrapper
    return decorator






























