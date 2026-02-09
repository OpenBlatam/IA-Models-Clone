"""
L2 Cache Layer - Redis cache
"""

import asyncio
import logging
from typing import Optional, List, Dict, Any

try:
    import redis.asyncio as redis
    from redis.asyncio.connection import ConnectionPool
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from ..constants import REDIS_TIMEOUT, REDIS_SCAN_COUNT

logger = logging.getLogger(__name__)


class L2Cache:
    """L2 Redis cache layer"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379/0", ttl: int = 3600):
        """Initialize L2 cache
        
        Args:
            redis_url: Redis connection URL
            ttl: Default TTL in seconds
        """
        self.enabled = REDIS_AVAILABLE
        self.ttl = ttl
        self.redis_url = redis_url
        self.redis_client: Optional[redis.Redis] = None
        self.redis_pool: Optional[ConnectionPool] = None
        
        if self.enabled:
            self._initialize_redis()
    
    def _initialize_redis(self):
        """Initialize Redis connection"""
        try:
            self.redis_pool = ConnectionPool.from_url(
                self.redis_url,
                decode_responses=False,
                max_connections=50,
                retry_on_timeout=True,
                health_check_interval=30
            )
            self.redis_client = redis.Redis(connection_pool=self.redis_pool)
            logger.info("Redis cache initialized with connection pooling")
        except Exception as e:
            logger.warning(f"Failed to initialize Redis: {e}")
            self.enabled = False
            self.redis_client = None
            self.redis_pool = None
    
    async def health_check(self, retry: bool = True) -> bool:
        """Check Redis connection health with retry logic
        
        Args:
            retry: Whether to retry on failure
            
        Returns:
            True if healthy, False otherwise
        """
        if not self.enabled or not self.redis_client:
            return False
        
        max_retries = 2 if retry else 1
        for attempt in range(max_retries):
            try:
                await asyncio.wait_for(self.redis_client.ping(), timeout=0.5)
                return True
            except Exception as e:
                if attempt < max_retries - 1:
                    await asyncio.sleep(0.1)
                    continue
                logger.debug(f"Redis health check failed: {e}")
                return False
    
    async def get(self, key: str, retry: bool = True) -> Optional[bytes]:
        """Get value from Redis with retry logic
        
        Args:
            key: Cache key
            retry: Whether to retry on failure
            
        Returns:
            Cached value as bytes or None
        """
        if not self.enabled or not self.redis_client:
            return None
        
        if not await self.health_check(retry=retry):
            return None
        
        max_retries = 2 if retry else 1
        for attempt in range(max_retries):
            try:
                cached = await asyncio.wait_for(
                    self.redis_client.get(f"cache:{key}"),
                    timeout=REDIS_TIMEOUT
                )
                return cached
            except asyncio.TimeoutError:
                if attempt < max_retries - 1:
                    await asyncio.sleep(0.1)
                    continue
                logger.warning(f"L2 cache get timeout for key: {key}")
            except Exception as e:
                if attempt < max_retries - 1:
                    await asyncio.sleep(0.1)
                    # Try to reconnect
                    try:
                        await self.health_check(retry=False)
                    except Exception:
                        pass
                    continue
                logger.debug(f"L2 cache get error for key {key}: {e}")
        
        return None
    
    async def set(self, key: str, value: bytes, ttl: Optional[int] = None, retry: bool = True) -> bool:
        """Set value in Redis with retry logic
        
        Args:
            key: Cache key
            value: Value as bytes
            ttl: Time-to-live in seconds
            retry: Whether to retry on failure
            
        Returns:
            True if successful, False otherwise
        """
        if not self.enabled or not self.redis_client:
            return False
        
        if not await self.health_check(retry=retry):
            return False
        
        ttl = ttl or self.ttl
        max_retries = 2 if retry else 1
        
        for attempt in range(max_retries):
            try:
                await asyncio.wait_for(
                    self.redis_client.setex(f"cache:{key}", ttl, value),
                    timeout=REDIS_TIMEOUT
                )
                return True
            except asyncio.TimeoutError:
                if attempt < max_retries - 1:
                    await asyncio.sleep(0.1)
                    continue
                logger.warning(f"L2 cache set timeout for key: {key}")
            except Exception as e:
                if attempt < max_retries - 1:
                    await asyncio.sleep(0.1)
                    # Try to reconnect
                    try:
                        await self.health_check(retry=False)
                    except Exception:
                        pass
                    continue
                logger.debug(f"L2 cache set error for key {key}: {e}")
        
        return False
    
    async def delete(self, key: str) -> bool:
        """Delete key from Redis
        
        Args:
            key: Cache key
            
        Returns:
            True if deleted, False otherwise
        """
        if not self.enabled or not self.redis_client:
            return False
        
        if not await self.health_check():
            return False
        
        try:
            await asyncio.wait_for(
                self.redis_client.delete(f"cache:{key}"),
                timeout=REDIS_TIMEOUT
            )
            return True
        except asyncio.TimeoutError:
            logger.warning(f"L2 cache delete timeout for key: {key}")
        except Exception as e:
            logger.error(f"L2 cache delete error: {e}")
        
        return False
    
    async def mget(self, keys: List[str]) -> List[Optional[bytes]]:
        """Get multiple values
        
        Args:
            keys: List of cache keys
            
        Returns:
            List of values (None for missing keys)
        """
        if not self.enabled or not self.redis_client or not keys:
            return [None] * len(keys)
        
        if not await self.health_check():
            return [None] * len(keys)
        
        try:
            redis_keys = [f"cache:{k}" for k in keys]
            values = await asyncio.wait_for(
                self.redis_client.mget(redis_keys),
                timeout=REDIS_TIMEOUT * 2
            )
            return values
        except Exception as e:
            logger.error(f"L2 cache mget error: {e}")
            return [None] * len(keys)
    
    async def mset(self, items: Dict[str, bytes], ttl: Optional[int] = None) -> bool:
        """Set multiple values using pipeline
        
        Args:
            items: Dictionary of key-value pairs
            ttl: Time-to-live in seconds
            
        Returns:
            True if successful, False otherwise
        """
        if not self.enabled or not self.redis_client or not items:
            return False
        
        if not await self.health_check():
            return False
        
        ttl = ttl or self.ttl
        
        try:
            pipe = self.redis_client.pipeline()
            for key, value in items.items():
                pipe.setex(f"cache:{key}", ttl, value)
            
            await asyncio.wait_for(pipe.execute(), timeout=REDIS_TIMEOUT * 2)
            return True
        except Exception as e:
            logger.error(f"L2 cache mset error: {e}")
            return False
    
    async def clear(self) -> bool:
        """Clear all cache entries
        
        Returns:
            True if successful, False otherwise
        """
        if not self.enabled or not self.redis_client:
            return False
        
        if not await self.health_check():
            return False
        
        try:
            cursor = 0
            while True:
                cursor, keys = await asyncio.wait_for(
                    self.redis_client.scan(cursor, match="cache:*", count=REDIS_SCAN_COUNT),
                    timeout=REDIS_TIMEOUT * 2
                )
                if keys:
                    await self.redis_client.delete(*keys)
                if cursor == 0:
                    break
            return True
        except Exception as e:
            logger.error(f"L2 cache clear error: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists
        
        Args:
            key: Cache key
            
        Returns:
            True if exists, False otherwise
        """
        if not self.enabled or not self.redis_client:
            return False
        
        if not await self.health_check():
            return False
        
        try:
            return bool(await self.redis_client.exists(f"cache:{key}"))
        except Exception:
            return False
    
    async def expire(self, key: str, ttl: int) -> bool:
        """Set TTL for existing key
        
        Args:
            key: Cache key
            ttl: Time-to-live in seconds
            
        Returns:
            True if successful, False otherwise
        """
        if not self.enabled or not self.redis_client:
            return False
        
        if not await self.health_check():
            return False
        
        try:
            await self.redis_client.expire(f"cache:{key}", ttl)
            return True
        except Exception as e:
            logger.error(f"L2 cache expire error: {e}")
            return False
    
    async def get_keys(self, pattern: str = "*") -> List[str]:
        """Get all keys matching pattern
        
        Args:
            pattern: Pattern to match
            
        Returns:
            List of keys (without cache: prefix)
        """
        if not self.enabled or not self.redis_client:
            return []
        
        if not await self.health_check():
            return []
        
        keys = []
        try:
            cursor = 0
            while True:
                cursor, redis_keys = await self.redis_client.scan(
                    cursor, match=f"cache:{pattern}", count=REDIS_SCAN_COUNT
                )
                for key in redis_keys:
                    clean_key = key.replace("cache:", "", 1)
                    keys.append(clean_key)
                if cursor == 0:
                    break
        except Exception as e:
            logger.error(f"Error getting keys from L2: {e}")
        
        return keys
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get Redis statistics
        
        Returns:
            Dictionary with statistics
        """
        stats = {
            "enabled": self.enabled,
            "healthy": await self.health_check() if self.enabled else False
        }
        
        if self.enabled and self.redis_client:
            try:
                info = await self.redis_client.info("memory")
                stats["memory_usage"] = info.get("used_memory_human", "N/A")
                stats["keys"] = await self.redis_client.dbsize()
            except Exception:
                pass
        
        return stats
    
    async def close(self):
        """Close Redis connections"""
        if self.redis_client:
            try:
                await self.redis_client.close()
            except Exception as e:
                logger.warning(f"Error closing Redis client: {e}")
        
        if self.redis_pool:
            try:
                await self.redis_pool.disconnect()
            except Exception as e:
                logger.warning(f"Error disconnecting Redis pool: {e}")

