"""
Caching utilities with optimized serialization

This module provides a multi-level caching system with:
- Redis cache (L2) for distributed caching
- In-memory cache (L1) as fallback
- Automatic fallback when Redis is unavailable
"""

import logging
from typing import Optional, Any, Dict
import time

try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None

from config.settings import settings
from utils.json_serializer import json_dumps, json_loads

logger = logging.getLogger(__name__)


class CacheService:
    """
    Cache service for Redis and in-memory caching
    
    Provides multi-level caching with automatic fallback:
    - L2: Redis cache (distributed, persistent)
    - L1: In-memory cache (local, fast)
    """
    
    def __init__(self):
        """
        Initialize cache service
        
        Sets up in-memory cache and attempts Redis connection.
        """
        self.redis_client: Optional[redis.Redis] = None
        self.memory_cache: Dict[str, Any] = {}
        self._redis_connected = False
        self._stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "deletes": 0
        }
        logger.info("CacheService initialized")
    
    async def connect(self) -> None:
        """
        Connect to Redis if available
        
        Attempts to establish Redis connection. Falls back to
        in-memory cache if Redis is unavailable.
        """
        if not REDIS_AVAILABLE:
            logger.info("Redis library not available, using in-memory cache only")
            return
        
        if not hasattr(settings, 'REDIS_URL') or not settings.REDIS_URL:
            logger.info("Redis URL not configured, using in-memory cache only")
            return
        
        try:
            self.redis_client = redis.from_url(
                settings.REDIS_URL,
                decode_responses=False,  # We handle encoding ourselves
                socket_connect_timeout=5,
                socket_timeout=5
            )
            await self.redis_client.ping()
            self._redis_connected = True
            logger.info("Successfully connected to Redis")
        except Exception as e:
            logger.warning(
                f"Failed to connect to Redis: {e}, "
                f"using in-memory cache only"
            )
            self._redis_connected = False
            self.redis_client = None
    
    async def disconnect(self) -> None:
        """Disconnect from Redis"""
        if not self.redis_client:
            return
        
        await self.redis_client.close()
        self._redis_connected = False
    
    async def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache
        
        Args:
            key: Cache key
            
        Returns:
            Cached value if found, None otherwise
            
        Note:
            Checks Redis first (L2), then in-memory cache (L1)
        """
        if not key:
            return None
        
        # Try Redis first (L2)
        if self._redis_connected and self.redis_client:
            try:
                value = await self.redis_client.get(key)
                if value:
                    try:
                        result = json_loads(value)
                        self._stats["hits"] += 1
                        logger.debug(f"Cache hit (Redis): {key}")
                        # Record metrics
                        try:
                            from utils.metrics import get_metrics_collector
                            get_metrics_collector().record_cache_hit("redis")
                        except Exception:
                            pass
                        return result
                    except Exception as e:
                        logger.warning(f"Error deserializing cached value: {e}")
            except Exception as e:
                logger.warning(f"Redis get error: {e}")
        
        # Fallback to memory cache (L1)
        if key in self.memory_cache:
            self._stats["hits"] += 1
            logger.debug(f"Cache hit (memory): {key}")
            # Record metrics
            try:
                from utils.metrics import get_metrics_collector
                get_metrics_collector().record_cache_hit("memory")
            except Exception:
                pass
            return self.memory_cache[key]
        
        self._stats["misses"] += 1
        logger.debug(f"Cache miss: {key}")
        # Record metrics
        try:
            from utils.metrics import get_metrics_collector
            get_metrics_collector().record_cache_miss("default")
        except Exception:
            pass
        return None
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> None:
        """
        Set value in cache
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (None for no expiration)
            
        Note:
            Sets in both Redis (L2) and memory cache (L1) if available
        """
        if not key:
            logger.warning("Attempted to cache with empty key")
            return
        
        try:
            serialized = json_dumps(value).encode('utf-8')
        except Exception as e:
            logger.error(f"Error serializing value for cache: {e}")
            return
        
        # Set in Redis (L2)
        if self._redis_connected and self.redis_client:
            try:
                if ttl and ttl > 0:
                    await self.redis_client.setex(key, ttl, serialized)
                else:
                    await self.redis_client.set(key, serialized)
                logger.debug(f"Cached in Redis: {key} (TTL: {ttl}s)")
            except Exception as e:
                logger.warning(f"Redis set error: {e}, using memory cache only")
        
        # Always set in memory cache (L1)
        self.memory_cache[key] = value
        self._stats["sets"] += 1
        logger.debug(f"Cached in memory: {key}")
        # Update cache size metric
        try:
            from utils.metrics import get_metrics_collector
            get_metrics_collector().set_cache_size(len(self.memory_cache), "memory")
        except Exception:
            pass
    
    async def delete(self, key: str) -> None:
        """
        Delete key from cache
        
        Args:
            key: Cache key to delete
            
        Note:
            Deletes from both Redis (L2) and memory cache (L1)
        """
        if not key:
            return
        
        # Delete from Redis (L2)
        if self._redis_connected and self.redis_client:
            try:
                await self.redis_client.delete(key)
                logger.debug(f"Deleted from Redis: {key}")
            except Exception as e:
                logger.warning(f"Redis delete error: {e}")
        
        # Delete from memory cache (L1)
        if key in self.memory_cache:
            del self.memory_cache[key]
            self._stats["deletes"] += 1
            logger.debug(f"Deleted from memory: {key}")
    
    async def clear_pattern(self, pattern: str) -> None:
        """Clear keys matching pattern (optimized for Redis)"""
        if self._redis_connected and self.redis_client:
            try:
                cursor = 0
                deleted_count = 0
                while True:
                    cursor, keys = await self.redis_client.scan(
                        cursor,
                        match=pattern,
                        count=100
                    )
                    if keys:
                        await self.redis_client.delete(*keys)
                        deleted_count += len(keys)
                    if cursor == 0:
                        break
                logger.debug(f"Cleared {deleted_count} keys matching pattern {pattern}")
            except Exception as e:
                logger.warning(f"Redis clear pattern error: {e}")
        
        # Clear from memory cache
        pattern_clean = pattern.replace("*", "")
        keys_to_delete = [
            k for k in self.memory_cache.keys()
            if pattern_clean in k
        ]
        for key in keys_to_delete:
            del self.memory_cache[key]
    
    async def get_or_set(
        self,
        key: str,
        factory: Any,
        ttl: Optional[int] = None
    ) -> Any:
        """
        Get from cache or set using factory function
        
        Args:
            key: Cache key
            factory: Factory function or value to cache if not found
            ttl: Time to live in seconds
            
        Returns:
            Cached value or result from factory
            
        Note:
            Implements cache-aside pattern. Factory can be:
            - Async function (coroutine)
            - Sync function
            - Direct value
        """
        # Try to get from cache first
        cached = await self.get(key)
        if cached is not None:
            return cached
        
        # Generate value using factory
        import asyncio
        if callable(factory):
            if asyncio.iscoroutinefunction(factory):
                value = await factory()
            else:
                value = factory()
        else:
            value = factory
        
        # Cache the generated value
        await self.set(key, value, ttl)
        return value
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics
        
        Returns:
            Dictionary with cache statistics
        """
        total_requests = self._stats["hits"] + self._stats["misses"]
        hit_rate = (
            (self._stats["hits"] / total_requests * 100)
            if total_requests > 0 else 0
        )
        
        return {
            "hits": self._stats["hits"],
            "misses": self._stats["misses"],
            "sets": self._stats["sets"],
            "deletes": self._stats["deletes"],
            "hit_rate": round(hit_rate, 2),
            "total_requests": total_requests,
            "memory_cache_size": len(self.memory_cache),
            "redis_connected": self._redis_connected
        }
    
    def reset_stats(self) -> None:
        """Reset cache statistics"""
        self._stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "deletes": 0
        }
        logger.debug("Cache statistics reset")


# Global cache instance
cache_service = CacheService()
