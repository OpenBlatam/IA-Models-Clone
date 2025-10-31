"""
Gamma App - Advanced Cache Service
Multi-level caching system with Redis and in-memory cache
"""

import asyncio
import json
import logging
import pickle
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, asdict
from functools import wraps
import redis
from redis.asyncio import Redis
import aioredis

logger = logging.getLogger(__name__)

@dataclass
class CacheConfig:
    """Cache configuration"""
    default_ttl: int = 3600  # 1 hour
    max_memory: str = "100mb"
    compression: bool = True
    serialization: str = "json"  # json, pickle, msgpack
    key_prefix: str = "gamma_app"

@dataclass
class CacheStats:
    """Cache statistics"""
    hits: int = 0
    misses: int = 0
    sets: int = 0
    deletes: int = 0
    memory_usage: int = 0
    key_count: int = 0

class AdvancedCacheService:
    """
    Advanced multi-level caching service
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize cache service"""
        self.config = config or {}
        self.redis_client: Optional[Redis] = None
        self.local_cache: Dict[str, Any] = {}
        self.cache_stats = CacheStats()
        self.cache_config = CacheConfig(**self.config.get('cache', {}))
        
        # Initialize Redis
        self._init_redis()
        
        # Cache warming tasks
        self.warming_tasks: Dict[str, asyncio.Task] = {}
        
        logger.info("Advanced Cache Service initialized successfully")

    def _init_redis(self):
        """Initialize Redis connection"""
        try:
            redis_url = self.config.get('redis_url', 'redis://localhost:6379')
            self.redis_client = Redis.from_url(redis_url, decode_responses=False)
            logger.info("Redis connection established for caching")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}")

    def _generate_cache_key(self, key: str, namespace: str = "default") -> str:
        """Generate cache key with namespace and prefix"""
        full_key = f"{self.cache_config.key_prefix}:{namespace}:{key}"
        return hashlib.md5(full_key.encode()).hexdigest()

    def _serialize_data(self, data: Any) -> bytes:
        """Serialize data for storage"""
        if self.cache_config.serialization == "json":
            return json.dumps(data, default=str).encode('utf-8')
        elif self.cache_config.serialization == "pickle":
            return pickle.dumps(data)
        else:
            return json.dumps(data, default=str).encode('utf-8')

    def _deserialize_data(self, data: bytes) -> Any:
        """Deserialize data from storage"""
        try:
            if self.cache_config.serialization == "json":
                return json.loads(data.decode('utf-8'))
            elif self.cache_config.serialization == "pickle":
                return pickle.loads(data)
            else:
                return json.loads(data.decode('utf-8'))
        except Exception as e:
            logger.error(f"Error deserializing data: {e}")
            return None

    async def get(self, key: str, namespace: str = "default") -> Optional[Any]:
        """Get value from cache"""
        cache_key = self._generate_cache_key(key, namespace)
        
        try:
            # Try local cache first
            if cache_key in self.local_cache:
                self.cache_stats.hits += 1
                logger.debug(f"Cache hit (local): {key}")
                return self.local_cache[cache_key]
            
            # Try Redis cache
            if self.redis_client:
                data = await self.redis_client.get(cache_key)
                if data:
                    value = self._deserialize_data(data)
                    # Store in local cache for faster access
                    self.local_cache[cache_key] = value
                    self.cache_stats.hits += 1
                    logger.debug(f"Cache hit (redis): {key}")
                    return value
            
            self.cache_stats.misses += 1
            logger.debug(f"Cache miss: {key}")
            return None
            
        except Exception as e:
            logger.error(f"Error getting from cache: {e}")
            self.cache_stats.misses += 1
            return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None, 
                  namespace: str = "default") -> bool:
        """Set value in cache"""
        cache_key = self._generate_cache_key(key, namespace)
        ttl = ttl or self.cache_config.default_ttl
        
        try:
            # Store in local cache
            self.local_cache[cache_key] = value
            
            # Store in Redis
            if self.redis_client:
                serialized_data = self._serialize_data(value)
                await self.redis_client.setex(cache_key, ttl, serialized_data)
            
            self.cache_stats.sets += 1
            logger.debug(f"Cache set: {key}")
            return True
            
        except Exception as e:
            logger.error(f"Error setting cache: {e}")
            return False

    async def delete(self, key: str, namespace: str = "default") -> bool:
        """Delete value from cache"""
        cache_key = self._generate_cache_key(key, namespace)
        
        try:
            # Remove from local cache
            if cache_key in self.local_cache:
                del self.local_cache[cache_key]
            
            # Remove from Redis
            if self.redis_client:
                await self.redis_client.delete(cache_key)
            
            self.cache_stats.deletes += 1
            logger.debug(f"Cache delete: {key}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting from cache: {e}")
            return False

    async def clear_namespace(self, namespace: str) -> bool:
        """Clear all keys in a namespace"""
        try:
            pattern = f"{self.cache_config.key_prefix}:{namespace}:*"
            
            # Clear local cache
            keys_to_remove = [k for k in self.local_cache.keys() if k.startswith(pattern)]
            for key in keys_to_remove:
                del self.local_cache[key]
            
            # Clear Redis cache
            if self.redis_client:
                keys = await self.redis_client.keys(pattern)
                if keys:
                    await self.redis_client.delete(*keys)
            
            logger.info(f"Cleared namespace: {namespace}")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing namespace: {e}")
            return False

    async def get_or_set(self, key: str, factory: Callable, ttl: Optional[int] = None,
                        namespace: str = "default") -> Any:
        """Get value from cache or set it using factory function"""
        value = await self.get(key, namespace)
        if value is not None:
            return value
        
        # Generate value using factory
        if asyncio.iscoroutinefunction(factory):
            value = await factory()
        else:
            value = factory()
        
        # Store in cache
        await self.set(key, value, ttl, namespace)
        return value

    async def invalidate_pattern(self, pattern: str, namespace: str = "default") -> int:
        """Invalidate all keys matching pattern"""
        try:
            full_pattern = f"{self.cache_config.key_prefix}:{namespace}:{pattern}"
            invalidated_count = 0
            
            # Invalidate local cache
            keys_to_remove = [k for k in self.local_cache.keys() if pattern in k]
            for key in keys_to_remove:
                del self.local_cache[key]
                invalidated_count += 1
            
            # Invalidate Redis cache
            if self.redis_client:
                keys = await self.redis_client.keys(full_pattern)
                if keys:
                    await self.redis_client.delete(*keys)
                    invalidated_count += len(keys)
            
            logger.info(f"Invalidated {invalidated_count} keys matching pattern: {pattern}")
            return invalidated_count
            
        except Exception as e:
            logger.error(f"Error invalidating pattern: {e}")
            return 0

    async def warm_cache(self, warming_functions: Dict[str, Callable], 
                        namespace: str = "default") -> Dict[str, bool]:
        """Warm cache with predefined functions"""
        results = {}
        
        for key, func in warming_functions.items():
            try:
                if asyncio.iscoroutinefunction(func):
                    value = await func()
                else:
                    value = func()
                
                await self.set(key, value, namespace=namespace)
                results[key] = True
                logger.info(f"Cache warmed: {key}")
                
            except Exception as e:
                logger.error(f"Error warming cache for {key}: {e}")
                results[key] = False
        
        return results

    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        try:
            total_requests = self.cache_stats.hits + self.cache_stats.misses
            hit_rate = (self.cache_stats.hits / total_requests * 100) if total_requests > 0 else 0
            
            stats = {
                "hits": self.cache_stats.hits,
                "misses": self.cache_stats.misses,
                "sets": self.cache_stats.sets,
                "deletes": self.cache_stats.deletes,
                "hit_rate": round(hit_rate, 2),
                "local_cache_size": len(self.local_cache),
                "redis_connected": self.redis_client is not None
            }
            
            # Get Redis info if available
            if self.redis_client:
                try:
                    redis_info = await self.redis_client.info()
                    stats.update({
                        "redis_memory_usage": redis_info.get("used_memory_human", "0B"),
                        "redis_connected_clients": redis_info.get("connected_clients", 0),
                        "redis_total_commands_processed": redis_info.get("total_commands_processed", 0)
                    })
                except Exception as e:
                    logger.error(f"Error getting Redis info: {e}")
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {}

    async def cleanup_expired(self) -> int:
        """Clean up expired entries from local cache"""
        try:
            # This is a simple implementation
            # In a real scenario, you'd track TTL for local cache entries
            expired_count = 0
            
            # For now, just limit local cache size
            max_local_size = 1000
            if len(self.local_cache) > max_local_size:
                # Remove oldest entries (simple FIFO)
                keys_to_remove = list(self.local_cache.keys())[:len(self.local_cache) - max_local_size]
                for key in keys_to_remove:
                    del self.local_cache[key]
                    expired_count += 1
            
            logger.info(f"Cleaned up {expired_count} expired cache entries")
            return expired_count
            
        except Exception as e:
            logger.error(f"Error cleaning up expired cache: {e}")
            return 0

    def cache_decorator(self, ttl: int = 3600, namespace: str = "default", 
                       key_func: Optional[Callable] = None):
        """Decorator for caching function results"""
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Generate cache key
                if key_func:
                    cache_key = key_func(*args, **kwargs)
                else:
                    cache_key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"
                
                # Try to get from cache
                cached_result = await self.get(cache_key, namespace)
                if cached_result is not None:
                    return cached_result
                
                # Execute function
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                # Store in cache
                await self.set(cache_key, result, ttl, namespace)
                return result
            
            return wrapper
        return decorator

    async def close(self):
        """Close cache service"""
        try:
            if self.redis_client:
                await self.redis_client.close()
            
            # Cancel warming tasks
            for task in self.warming_tasks.values():
                if not task.done():
                    task.cancel()
            
            logger.info("Cache service closed")
            
        except Exception as e:
            logger.error(f"Error closing cache service: {e}")

# Global cache instance
cache_service = AdvancedCacheService()

# Convenience functions
async def get_cached(key: str, namespace: str = "default") -> Optional[Any]:
    """Get value from cache"""
    return await cache_service.get(key, namespace)

async def set_cached(key: str, value: Any, ttl: Optional[int] = None, 
                    namespace: str = "default") -> bool:
    """Set value in cache"""
    return await cache_service.set(key, value, ttl, namespace)

async def delete_cached(key: str, namespace: str = "default") -> bool:
    """Delete value from cache"""
    return await cache_service.delete(key, namespace)

def cached(ttl: int = 3600, namespace: str = "default", 
          key_func: Optional[Callable] = None):
    """Cache decorator"""
    return cache_service.cache_decorator(ttl, namespace, key_func)



























