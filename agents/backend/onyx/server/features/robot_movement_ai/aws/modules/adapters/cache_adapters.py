"""
Cache Adapters
==============

Implementations of CachePort with different backends.
"""

import logging
from typing import Any, Optional
import os
from aws.modules.ports.cache_port import CachePort

logger = logging.getLogger(__name__)


class RedisCacheAdapter(CachePort):
    """Redis implementation of CachePort."""
    
    def __init__(self, redis_url: Optional[str] = None):
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self._client = None
    
    async def _get_client(self):
        """Get Redis client."""
        if self._client is None:
            try:
                import redis.asyncio as aioredis
                self._client = await aioredis.from_url(
                    self.redis_url,
                    encoding="utf-8",
                    decode_responses=True
                )
            except ImportError:
                logger.warning("aioredis not installed")
            except Exception as e:
                logger.error(f"Failed to connect to Redis: {e}")
        return self._client
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from Redis."""
        client = await self._get_client()
        if not client:
            return None
        
        try:
            import json
            value = await client.get(key)
            if value:
                return json.loads(value)
        except Exception as e:
            logger.error(f"Redis get error: {e}")
        return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in Redis."""
        client = await self._get_client()
        if not client:
            return False
        
        try:
            import json
            serialized = json.dumps(value)
            if ttl:
                await client.setex(key, ttl, serialized)
            else:
                await client.set(key, serialized)
            return True
        except Exception as e:
            logger.error(f"Redis set error: {e}")
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
            logger.error(f"Redis delete error: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in Redis."""
        client = await self._get_client()
        if not client:
            return False
        
        try:
            return await client.exists(key) > 0
        except Exception as e:
            logger.error(f"Redis exists error: {e}")
            return False
    
    async def clear(self, pattern: Optional[str] = None) -> int:
        """Clear cache in Redis."""
        client = await self._get_client()
        if not client:
            return 0
        
        try:
            if pattern:
                keys = await client.keys(pattern)
                if keys:
                    return await client.delete(*keys)
            else:
                return await client.flushdb()
        except Exception as e:
            logger.error(f"Redis clear error: {e}")
            return 0


class MemcachedCacheAdapter(CachePort):
    """Memcached implementation of CachePort."""
    
    def __init__(self, servers: Optional[str] = None):
        self.servers = servers or os.getenv("MEMCACHED_SERVERS", "localhost:11211")
        self._client = None
    
    async def _get_client(self):
        """Get Memcached client."""
        if self._client is None:
            try:
                import aiomcache
                host, port = self.servers.split(":")
                self._client = aiomcache.Client(host, int(port))
            except ImportError:
                logger.warning("aiomcache not installed")
            except Exception as e:
                logger.error(f"Failed to connect to Memcached: {e}")
        return self._client
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from Memcached."""
        client = await self._get_client()
        if not client:
            return None
        
        try:
            import json
            value = await client.get(key.encode())
            if value:
                return json.loads(value.decode())
        except Exception as e:
            logger.error(f"Memcached get error: {e}")
        return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in Memcached."""
        client = await self._get_client()
        if not client:
            return False
        
        try:
            import json
            serialized = json.dumps(value).encode()
            ttl = ttl or 0
            return await client.set(key.encode(), serialized, exptime=ttl)
        except Exception as e:
            logger.error(f"Memcached set error: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete key from Memcached."""
        client = await self._get_client()
        if not client:
            return False
        
        try:
            return await client.delete(key.encode())
        except Exception as e:
            logger.error(f"Memcached delete error: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in Memcached."""
        value = await self.get(key)
        return value is not None
    
    async def clear(self, pattern: Optional[str] = None) -> int:
        """Clear cache in Memcached (not supported)."""
        logger.warning("Memcached does not support pattern-based clearing")
        return 0


class InMemoryCacheAdapter(CachePort):
    """In-memory implementation of CachePort (for testing)."""
    
    def __init__(self):
        from typing import Dict, Tuple
        import time
        self._storage: Dict[str, Tuple[Any, Optional[float]]] = {}
        self._time = time
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from memory."""
        if key not in self._storage:
            return None
        
        value, expiry = self._storage[key]
        if expiry and self._time.time() > expiry:
            del self._storage[key]
            return None
        
        return value
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in memory."""
        expiry = None
        if ttl:
            expiry = self._time.time() + ttl
        self._storage[key] = (value, expiry)
        return True
    
    async def delete(self, key: str) -> bool:
        """Delete key from memory."""
        if key in self._storage:
            del self._storage[key]
            return True
        return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in memory."""
        return key in self._storage
    
    async def clear(self, pattern: Optional[str] = None) -> int:
        """Clear cache in memory."""
        if pattern:
            import fnmatch
            keys_to_delete = [k for k in self._storage.keys() if fnmatch.fnmatch(k, pattern)]
            for key in keys_to_delete:
                del self._storage[key]
            return len(keys_to_delete)
        else:
            count = len(self._storage)
            self._storage.clear()
            return count

