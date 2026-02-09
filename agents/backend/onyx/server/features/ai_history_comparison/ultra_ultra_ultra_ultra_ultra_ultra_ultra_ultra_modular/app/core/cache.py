"""
Advanced caching system with Redis and in-memory fallback.
"""

import asyncio
import json
import pickle
from typing import Any, Optional, Union, Dict, List
from datetime import datetime, timedelta
from functools import wraps
import hashlib

import redis.asyncio as redis
from ..core.config import get_settings
from ..core.logging import get_logger

logger = get_logger(__name__)

# Global cache instance
_cache_instance: Optional['CacheManager'] = None


class CacheManager:
    """Advanced cache manager with Redis and in-memory fallback."""
    
    def __init__(self):
        self.redis_client: Optional[redis.Redis] = None
        self.memory_cache: Dict[str, Dict[str, Any]] = {}
        self.settings = get_settings()
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize cache manager."""
        try:
            # Try to connect to Redis
            self.redis_client = redis.from_url(
                self.settings.redis_url,
                encoding="utf-8",
                decode_responses=False,  # We'll handle encoding manually
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
                health_check_interval=30
            )
            
            # Test connection
            await self.redis_client.ping()
            logger.info("Connected to Redis cache")
            
        except Exception as e:
            logger.warning(f"Redis connection failed, using in-memory cache: {e}")
            self.redis_client = None
        
        self._initialized = True
    
    async def shutdown(self) -> None:
        """Shutdown cache manager."""
        if self.redis_client:
            await self.redis_client.close()
        self.memory_cache.clear()
        self._initialized = False
    
    async def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache."""
        if not self._initialized:
            return default
        
        try:
            # Try Redis first
            if self.redis_client:
                value = await self.redis_client.get(key)
                if value is not None:
                    return self._deserialize(value)
            
            # Fallback to memory cache
            if key in self.memory_cache:
                cache_entry = self.memory_cache[key]
                if self._is_expired(cache_entry):
                    del self.memory_cache[key]
                    return default
                return cache_entry["value"]
            
            return default
            
        except Exception as e:
            logger.error(f"Error getting cache key {key}: {e}")
            return default
    
    async def set(
        self, 
        key: str, 
        value: Any, 
        ttl: Optional[int] = None,
        tags: Optional[List[str]] = None
    ) -> bool:
        """Set value in cache."""
        if not self._initialized:
            return False
        
        try:
            ttl = ttl or self.settings.redis_ttl
            serialized_value = self._serialize(value)
            
            # Set in Redis
            if self.redis_client:
                await self.redis_client.setex(key, ttl, serialized_value)
                
                # Store tags for invalidation
                if tags:
                    for tag in tags:
                        await self.redis_client.sadd(f"tag:{tag}", key)
                        await self.redis_client.expire(f"tag:{tag}", ttl)
            
            # Set in memory cache
            self.memory_cache[key] = {
                "value": value,
                "expires_at": datetime.utcnow() + timedelta(seconds=ttl),
                "tags": tags or []
            }
            
            return True
            
        except Exception as e:
            logger.error(f"Error setting cache key {key}: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
        if not self._initialized:
            return False
        
        try:
            # Delete from Redis
            if self.redis_client:
                await self.redis_client.delete(key)
            
            # Delete from memory cache
            if key in self.memory_cache:
                del self.memory_cache[key]
            
            return True
            
        except Exception as e:
            logger.error(f"Error deleting cache key {key}: {e}")
            return False
    
    async def invalidate_by_tags(self, tags: List[str]) -> int:
        """Invalidate cache entries by tags."""
        if not self._initialized:
            return 0
        
        invalidated_count = 0
        
        try:
            if self.redis_client:
                for tag in tags:
                    # Get keys with this tag
                    keys = await self.redis_client.smembers(f"tag:{tag}")
                    if keys:
                        # Delete keys
                        await self.redis_client.delete(*keys)
                        # Delete tag set
                        await self.redis_client.delete(f"tag:{tag}")
                        invalidated_count += len(keys)
            
            # Invalidate memory cache
            keys_to_delete = []
            for key, cache_entry in self.memory_cache.items():
                if cache_entry.get("tags") and any(tag in cache_entry["tags"] for tag in tags):
                    keys_to_delete.append(key)
            
            for key in keys_to_delete:
                del self.memory_cache[key]
                invalidated_count += 1
            
            return invalidated_count
            
        except Exception as e:
            logger.error(f"Error invalidating cache by tags {tags}: {e}")
            return 0
    
    async def clear(self) -> bool:
        """Clear all cache."""
        if not self._initialized:
            return False
        
        try:
            # Clear Redis
            if self.redis_client:
                await self.redis_client.flushdb()
            
            # Clear memory cache
            self.memory_cache.clear()
            
            return True
            
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return False
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if not self._initialized:
            return {"status": "not_initialized"}
        
        stats = {
            "memory_cache_size": len(self.memory_cache),
            "redis_connected": self.redis_client is not None
        }
        
        if self.redis_client:
            try:
                info = await self.redis_client.info()
                stats.update({
                    "redis_used_memory": info.get("used_memory_human"),
                    "redis_connected_clients": info.get("connected_clients"),
                    "redis_total_commands_processed": info.get("total_commands_processed"),
                    "redis_keyspace_hits": info.get("keyspace_hits"),
                    "redis_keyspace_misses": info.get("keyspace_misses")
                })
            except Exception as e:
                logger.error(f"Error getting Redis stats: {e}")
        
        return stats
    
    def _serialize(self, value: Any) -> bytes:
        """Serialize value for storage."""
        try:
            # Try JSON first for simple types
            if isinstance(value, (str, int, float, bool, list, dict, type(None))):
                return json.dumps(value).encode('utf-8')
            else:
                # Use pickle for complex objects
                return pickle.dumps(value)
        except Exception:
            # Fallback to pickle
            return pickle.dumps(value)
    
    def _deserialize(self, value: bytes) -> Any:
        """Deserialize value from storage."""
        try:
            # Try JSON first
            return json.loads(value.decode('utf-8'))
        except Exception:
            # Fallback to pickle
            return pickle.loads(value)
    
    def _is_expired(self, cache_entry: Dict[str, Any]) -> bool:
        """Check if cache entry is expired."""
        expires_at = cache_entry.get("expires_at")
        if not expires_at:
            return False
        return datetime.utcnow() > expires_at


async def get_cache_manager() -> CacheManager:
    """Get global cache manager instance."""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = CacheManager()
        await _cache_instance.initialize()
    return _cache_instance


def cache_key(*args, **kwargs) -> str:
    """Generate cache key from arguments."""
    key_data = {"args": args, "kwargs": sorted(kwargs.items())}
    key_string = json.dumps(key_data, sort_keys=True)
    return hashlib.md5(key_string.encode()).hexdigest()


def cached(
    ttl: Optional[int] = None,
    tags: Optional[List[str]] = None,
    key_prefix: str = ""
) -> callable:
    """Decorator for caching function results."""
    def decorator(func: callable) -> callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Generate cache key
            key = f"{key_prefix}:{func.__name__}:{cache_key(*args, **kwargs)}"
            
            # Get cache manager
            cache = await get_cache_manager()
            
            # Try to get from cache
            cached_result = await cache.get(key)
            if cached_result is not None:
                logger.debug(f"Cache hit for {key}")
                return cached_result
            
            # Execute function
            logger.debug(f"Cache miss for {key}")
            result = await func(*args, **kwargs)
            
            # Store in cache
            await cache.set(key, result, ttl=ttl, tags=tags)
            
            return result
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # For sync functions, we'll store the result but not retrieve from cache
            # This is a limitation of the current implementation
            result = func(*args, **kwargs)
            
            # Store in cache asynchronously (fire and forget)
            async def store_result():
                key = f"{key_prefix}:{func.__name__}:{cache_key(*args, **kwargs)}"
                cache = await get_cache_manager()
                await cache.set(key, result, ttl=ttl, tags=tags)
            
            asyncio.create_task(store_result())
            return result
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


# Cache utilities
async def invalidate_analysis_cache() -> int:
    """Invalidate all analysis-related cache."""
    cache = await get_cache_manager()
    return await cache.invalidate_by_tags(["analysis", "content"])


async def invalidate_plugin_cache() -> int:
    """Invalidate all plugin-related cache."""
    cache = await get_cache_manager()
    return await cache.invalidate_by_tags(["plugins", "extensions"])


async def warm_up_cache() -> None:
    """Warm up cache with frequently accessed data."""
    cache = await get_cache_manager()
    
    # Cache plugin list
    from ..services.plugin_service import list_plugins
    plugins = await list_plugins()
    await cache.set("plugins:list", [p.dict() for p in plugins], ttl=300, tags=["plugins"])
    
    # Cache system stats
    from ..services.plugin_service import get_plugin_stats
    stats = await get_plugin_stats()
    await cache.set("system:stats", stats, ttl=60, tags=["stats"])
    
    logger.info("Cache warmed up successfully")


