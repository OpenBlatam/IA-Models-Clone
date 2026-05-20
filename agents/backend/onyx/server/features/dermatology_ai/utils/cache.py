"""
Advanced Cache Manager with Redis support for stateless microservices
Falls back to in-memory cache if Redis is not available
"""

import json
import hashlib
from typing import Any, Optional, Dict
from datetime import timedelta
import logging

logger = logging.getLogger(__name__)

# Try to import Redis
try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("Redis not available. Install with: pip install redis[hiredis]")

# Fallback: in-memory cache
from functools import lru_cache
from collections import OrderedDict
from time import time as current_time


class InMemoryCache:
    """In-memory LRU cache with TTL support"""
    
    def __init__(self, maxsize: int = 1000, default_ttl: int = 3600):
        self.maxsize = maxsize
        self.default_ttl = default_ttl
        self.cache: OrderedDict = OrderedDict()
        self.expiry: Dict[str, float] = {}
    
    def _is_expired(self, key: str) -> bool:
        """Check if key is expired"""
        if key not in self.expiry:
            return False
        return current_time() > self.expiry[key]
    
    def _cleanup_expired(self):
        """Remove expired entries"""
        now = current_time()
        expired_keys = [
            key for key, expiry_time in self.expiry.items()
            if now > expiry_time
        ]
        for key in expired_keys:
            self.cache.pop(key, None)
            self.expiry.pop(key, None)
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        self._cleanup_expired()
        
        if key in self.cache and not self._is_expired(key):
            # Move to end (most recently used)
            value = self.cache.pop(key)
            self.cache[key] = value
            return value
        return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache"""
        self._cleanup_expired()
        
        ttl = ttl or self.default_ttl
        
        # Remove if exists
        self.cache.pop(key, None)
        self.expiry.pop(key, None)
        
        # Add new entry
        self.cache[key] = value
        self.expiry[key] = current_time() + ttl
        
        # Enforce maxsize
        if len(self.cache) > self.maxsize:
            # Remove oldest
            oldest_key = next(iter(self.cache))
            self.cache.pop(oldest_key)
            self.expiry.pop(oldest_key, None)
        
        return True
    
    async def delete(self, key: str) -> bool:
        """Delete key from cache"""
        deleted = key in self.cache
        self.cache.pop(key, None)
        self.expiry.pop(key, None)
        return deleted
    
    async def clear(self) -> bool:
        """Clear all cache"""
        self.cache.clear()
        self.expiry.clear()
        return True
    
    async def ping(self) -> bool:
        """Check if cache is available"""
        return True
    
    async def close(self):
        """Close cache (no-op for in-memory)"""
        pass


class RedisCache:
    """Redis-backed cache for distributed systems"""
    
    def __init__(self, redis_url: Optional[str] = None, default_ttl: int = 3600):
        self.redis_url = redis_url or "redis://localhost:6379/0"
        self.default_ttl = default_ttl
        self.client: Optional[redis.Redis] = None
        self._connected = False
    
    async def initialize(self):
        """Initialize Redis connection"""
        try:
            self.client = redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True,
                socket_connect_timeout=2,
                socket_timeout=2,
                retry_on_timeout=True,
            )
            await self.client.ping()
            self._connected = True
            logger.info("✅ Redis cache connected")
        except Exception as e:
            logger.warning(f"Redis connection failed, using fallback: {e}")
            self._connected = False
            self.client = None
    
    def _serialize(self, value: Any) -> str:
        """Serialize value to JSON string"""
        return json.dumps(value, default=str)
    
    def _deserialize(self, value: str) -> Any:
        """Deserialize JSON string to value"""
        return json.loads(value)
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if not self._connected or not self.client:
            return None
        
        try:
            value = await self.client.get(key)
            if value:
                return self._deserialize(value)
            return None
        except Exception as e:
            logger.error(f"Redis get error: {e}")
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache"""
        if not self._connected or not self.client:
            return False
        
        try:
            ttl = ttl or self.default_ttl
            serialized = self._serialize(value)
            await self.client.setex(key, ttl, serialized)
            return True
        except Exception as e:
            logger.error(f"Redis set error: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete key from cache"""
        if not self._connected or not self.client:
            return False
        
        try:
            result = await self.client.delete(key)
            return result > 0
        except Exception as e:
            logger.error(f"Redis delete error: {e}")
            return False
    
    async def clear(self) -> bool:
        """Clear all cache (use with caution!)"""
        if not self._connected or not self.client:
            return False
        
        try:
            await self.client.flushdb()
            return True
        except Exception as e:
            logger.error(f"Redis clear error: {e}")
            return False
    
    async def ping(self) -> bool:
        """Check if Redis is available"""
        if not self._connected or not self.client:
            return False
        
        try:
            await self.client.ping()
            return True
        except Exception:
            return False
    
    async def close(self):
        """Close Redis connection"""
        if self.client:
            await self.client.close()
            self._connected = False


class CacheManager:
    """Unified cache manager with Redis and fallback"""
    
    def __init__(self, redis_url: Optional[str] = None, use_redis: bool = True):
        self.redis_url = redis_url
        self.use_redis = use_redis and REDIS_AVAILABLE
        self.cache: Optional[Any] = None
    
    async def initialize(self):
        """Initialize cache backend"""
        if self.use_redis:
            self.cache = RedisCache(self.redis_url)
            await self.cache.initialize()
            
            # If Redis failed, fallback to in-memory
            if not await self.cache.ping():
                logger.warning("Redis unavailable, using in-memory cache")
                self.cache = InMemoryCache()
        else:
            self.cache = InMemoryCache()
            logger.info("Using in-memory cache")
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if not self.cache:
            await self.initialize()
        return await self.cache.get(key)
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache"""
        if not self.cache:
            await self.initialize()
        return await self.cache.set(key, value, ttl)
    
    async def delete(self, key: str) -> bool:
        """Delete key from cache"""
        if not self.cache:
            await self.initialize()
        return await self.cache.delete(key)
    
    async def clear(self) -> bool:
        """Clear all cache"""
        if not self.cache:
            await self.initialize()
        return await self.cache.clear()
    
    async def ping(self) -> bool:
        """Check if cache is available"""
        if not self.cache:
            await self.initialize()
        return await self.cache.ping()
    
    async def close(self):
        """Close cache connection"""
        if self.cache:
            await self.cache.close()


# Global cache manager instance
_cache_manager: Optional[CacheManager] = None


def get_cache_manager(redis_url: Optional[str] = None) -> CacheManager:
    """Get or create global cache manager"""
    global _cache_manager
    if _cache_manager is None:
        import os
        redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
        use_redis = os.getenv("USE_REDIS", "true").lower() == "true"
        _cache_manager = CacheManager(redis_url=redis_url, use_redis=use_redis)
    return _cache_manager


def cache_key(*args, **kwargs) -> str:
    """Generate cache key from arguments"""
    key_parts = [str(arg) for arg in args]
    key_parts.extend([f"{k}:{v}" for k, v in sorted(kwargs.items())])
    key_string = "|".join(key_parts)
    return hashlib.md5(key_string.encode()).hexdigest()
