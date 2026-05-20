"""
Redis Cache Implementation
===========================

High-performance Redis cache for microservices architecture.
"""

import json
import logging
from typing import Optional, Any
import redis.asyncio as aioredis
from redis.asyncio import Redis

logger = logging.getLogger(__name__)


class RedisCache:
    """Redis cache implementation with async support."""

    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        self.redis_url = redis_url
        self.client: Optional[Redis] = None

    async def connect(self):
        """Connect to Redis with retry and health check."""
        max_retries = 3
        retry_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                self.client = await aioredis.from_url(
                    self.redis_url,
                    decode_responses=True,
                    socket_connect_timeout=5,
                    socket_timeout=5,
                    socket_keepalive=True,
                    socket_keepalive_options={},
                    retry_on_timeout=True,
                    health_check_interval=30,
                    max_connections=50,
                )
                # Test connection
                await self.client.ping()
                logger.info("Connected to Redis")
                return
            except Exception as e:
                logger.warning(f"Redis connection attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    import asyncio
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    logger.error(f"Failed to connect to Redis after {max_retries} attempts: {e}")
                    self.client = None
                    from ..core.exceptions import CacheError
                    raise CacheError(f"Failed to connect to Redis: {e}", "CACHE_CONNECTION_ERROR")

    async def disconnect(self):
        """Disconnect from Redis."""
        if self.client:
            await self.client.close()
            logger.info("Disconnected from Redis")

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache with error handling."""
        if not self.client:
            return None

        try:
            value = await self.client.get(key)
            if value:
                return json.loads(value)
            return None
        except aioredis.ConnectionError as e:
            logger.error(f"Redis connection error: {e}")
            # Try to reconnect
            await self.connect()
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding cache value: {e}")
            # Delete corrupted cache entry
            await self.delete(key)
            return None
        except Exception as e:
            logger.error(f"Error getting from cache: {e}")
            return None

    async def set(
        self, key: str, value: Any, ttl: Optional[int] = None
    ) -> bool:
        """Set value in cache with error handling."""
        if not self.client:
            return False

        try:
            serialized = json.dumps(value)
            if ttl:
                await self.client.setex(key, ttl, serialized)
            else:
                await self.client.set(key, serialized)
            return True
        except aioredis.ConnectionError as e:
            logger.error(f"Redis connection error: {e}")
            # Try to reconnect
            await self.connect()
            return False
        except (TypeError, ValueError) as e:
            logger.error(f"Error serializing cache value: {e}")
            return False
        except Exception as e:
            logger.error(f"Error setting cache: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
        if not self.client:
            return False

        try:
            await self.client.delete(key)
            return True
        except Exception as e:
            logger.error(f"Error deleting from cache: {e}")
            return False

    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        if not self.client:
            return False

        try:
            return await self.client.exists(key) > 0
        except Exception as e:
            logger.error(f"Error checking cache: {e}")
            return False

    async def clear_pattern(self, pattern: str) -> int:
        """Clear all keys matching pattern."""
        if not self.client:
            return 0

        try:
            keys = []
            async for key in self.client.scan_iter(match=pattern):
                keys.append(key)
            
            if keys:
                return await self.client.delete(*keys)
            return 0
        except Exception as e:
            logger.error(f"Error clearing cache pattern: {e}")
            return 0


# Global cache instance
_cache_instance: Optional[RedisCache] = None


async def get_cache() -> RedisCache:
    """Get or create cache instance."""
    global _cache_instance
    if _cache_instance is None:
        from ..config.settings import get_settings
        settings = get_settings()
        _cache_instance = RedisCache(redis_url=settings.redis_url)
        await _cache_instance.connect()
    return _cache_instance

