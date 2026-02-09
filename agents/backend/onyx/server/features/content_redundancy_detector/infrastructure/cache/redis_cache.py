"""
Redis Cache Implementation
High-performance caching layer for stateless microservices
"""

import json
import logging
import hashlib
from typing import Any, Optional, Union, List
from datetime import timedelta

logger = logging.getLogger(__name__)


class RedisCache:
    """
    Redis cache implementation with connection pooling
    Optimized for high-throughput read-heavy workloads
    """
    
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        default_ttl: int = 3600,
        max_connections: int = 50
    ):
        self.redis_url = redis_url
        self.default_ttl = default_ttl
        self.max_connections = max_connections
        self._pool = None
        self._client = None
    
    async def _get_client(self):
        """Get Redis client with connection pooling"""
        if self._client is None:
            try:
                import redis.asyncio as aioredis
                
                # Create connection pool for better performance
                self._pool = aioredis.ConnectionPool.from_url(
                    self.redis_url,
                    max_connections=self.max_connections,
                    decode_responses=True
                )
                self._client = aioredis.Redis(connection_pool=self._pool)
                
                # Test connection
                await self._client.ping()
                logger.info("Redis cache connected with connection pooling")
            except ImportError:
                logger.error("redis.asyncio not available")
                raise
            except Exception as e:
                logger.error(f"Failed to connect to Redis: {e}")
                raise
        
        return self._client
    
    def _serialize(self, value: Any) -> str:
        """Serialize value for storage"""
        if isinstance(value, str):
            return value
        return json.dumps(value)
    
    def _deserialize(self, value: str) -> Any:
        """Deserialize value from storage"""
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return value
    
    def _make_key(self, key: str, prefix: str = "cache") -> str:
        """Create namespaced cache key"""
        return f"{prefix}:{key}"
    
    async def get(
        self,
        key: str,
        default: Optional[Any] = None,
        prefix: str = "cache"
    ) -> Optional[Any]:
        """
        Get value from cache
        
        Args:
            key: Cache key
            default: Default value if not found
            prefix: Key prefix for namespacing
            
        Returns:
            Cached value or default
        """
        client = await self._get_client()
        cache_key = self._make_key(key, prefix)
        
        try:
            value = await client.get(cache_key)
            if value is None:
                return default
            return self._deserialize(value)
        except Exception as e:
            logger.error(f"Cache get error for key {key}: {e}")
            return default
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        prefix: str = "cache"
    ) -> bool:
        """
        Set value in cache
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (defaults to default_ttl)
            prefix: Key prefix for namespacing
            
        Returns:
            True if successful
        """
        client = await self._get_client()
        cache_key = self._make_key(key, prefix)
        
        try:
            serialized = self._serialize(value)
            ttl = ttl or self.default_ttl
            
            await client.setex(cache_key, ttl, serialized)
            return True
        except Exception as e:
            logger.error(f"Cache set error for key {key}: {e}")
            return False
    
    async def delete(self, key: str, prefix: str = "cache") -> bool:
        """Delete key from cache"""
        client = await self._get_client()
        cache_key = self._make_key(key, prefix)
        
        try:
            deleted = await client.delete(cache_key)
            return deleted > 0
        except Exception as e:
            logger.error(f"Cache delete error for key {key}: {e}")
            return False
    
    async def delete_pattern(self, pattern: str, prefix: str = "cache") -> int:
        """Delete all keys matching pattern"""
        client = await self._get_client()
        full_pattern = self._make_key(pattern, prefix)
        
        try:
            count = 0
            async for key in client.scan_iter(match=full_pattern):
                await client.delete(key)
                count += 1
            return count
        except Exception as e:
            logger.error(f"Cache delete pattern error: {e}")
            return 0
    
    async def exists(self, key: str, prefix: str = "cache") -> bool:
        """Check if key exists"""
        client = await self._get_client()
        cache_key = self._make_key(key, prefix)
        
        try:
            exists = await client.exists(cache_key)
            return exists > 0
        except Exception as e:
            logger.error(f"Cache exists check error: {e}")
            return False
    
    async def increment(self, key: str, amount: int = 1, prefix: str = "cache") -> Optional[int]:
        """Increment counter"""
        client = await self._get_client()
        cache_key = self._make_key(key, prefix)
        
        try:
            return await client.incrby(cache_key, amount)
        except Exception as e:
            logger.error(f"Cache increment error: {e}")
            return None
    
    async def get_or_set(
        self,
        key: str,
        callable: callable,
        ttl: Optional[int] = None,
        prefix: str = "cache"
    ) -> Any:
        """
        Get value from cache or set using callable
        
        Args:
            key: Cache key
            callable: Async callable to generate value if not cached
            ttl: Time to live
            prefix: Key prefix
            
        Returns:
            Cached or newly generated value
        """
        value = await self.get(key, prefix=prefix)
        
        if value is not None:
            return value
        
        # Generate value
        if callable:
            value = await callable() if hasattr(callable, '__call__') else callable
            
            await self.set(key, value, ttl, prefix)
            return value
        
        return None
    
    async def clear_namespace(self, prefix: str = "cache") -> int:
        """Clear all keys in namespace"""
        pattern = f"{prefix}:*"
        return await self.delete_pattern("*", prefix)
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        client = await self._get_client()
        
        try:
            info = await client.info("stats")
            return {
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
                "total_commands": info.get("total_commands_processed", 0),
                "hit_rate": (
                    info.get("keyspace_hits", 0) / 
                    max(info.get("keyspace_hits", 0) + info.get("keyspace_misses", 1), 1)
                ) * 100
            }
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {}






