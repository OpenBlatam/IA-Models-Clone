"""
Advanced Cache Manager with Redis and Memory Fallback
Supports intelligent caching, TTL management, and cache invalidation
"""

import json
import time
import hashlib
from typing import Any, Optional, Dict, Union
from functools import wraps
import logging

try:
    import redis
    from redis.exceptions import RedisError
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

logger = logging.getLogger(__name__)


class CacheManager:
    """Advanced cache manager with Redis and memory fallback"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379/0", 
                 memory_cache_size: int = 1000):
        self.redis_client = None
        self.memory_cache: Dict[str, Dict[str, Any]] = {}
        self.memory_cache_size = memory_cache_size
        self.cache_stats = {
            "hits": 0,
            "misses": 0,
            "memory_hits": 0,
            "redis_hits": 0,
            "errors": 0
        }
        
        # Initialize Redis if available
        if REDIS_AVAILABLE:
            try:
                self.redis_client = redis.from_url(redis_url, decode_responses=True)
                self.redis_client.ping()
                logger.info("Redis cache initialized successfully")
            except RedisError as e:
                logger.warning(f"Redis not available, using memory cache: {e}")
                self.redis_client = None
        else:
            logger.warning("Redis not installed, using memory cache only")
    
    def _generate_key(self, prefix: str, *args, **kwargs) -> str:
        """Generate cache key from arguments"""
        key_data = {
            "args": args,
            "kwargs": sorted(kwargs.items()) if kwargs else {}
        }
        key_string = json.dumps(key_data, sort_keys=True)
        key_hash = hashlib.md5(key_string.encode()).hexdigest()
        return f"{prefix}:{key_hash}"
    
    def _is_expired(self, cache_entry: Dict[str, Any]) -> bool:
        """Check if cache entry is expired"""
        if "expires_at" not in cache_entry:
            return False
        return time.time() > cache_entry["expires_at"]
    
    def _cleanup_memory_cache(self):
        """Clean up old entries from memory cache"""
        if len(self.memory_cache) <= self.memory_cache_size:
            return
        
        # Remove expired entries first
        expired_keys = [
            key for key, entry in self.memory_cache.items()
            if self._is_expired(entry)
        ]
        for key in expired_keys:
            del self.memory_cache[key]
        
        # If still over limit, remove oldest entries
        if len(self.memory_cache) > self.memory_cache_size:
            sorted_items = sorted(
                self.memory_cache.items(),
                key=lambda x: x[1].get("created_at", 0)
            )
            items_to_remove = len(self.memory_cache) - self.memory_cache_size
            for key, _ in sorted_items[:items_to_remove]:
                del self.memory_cache[key]
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        try:
            # Try Redis first
            if self.redis_client:
                try:
                    cached_data = self.redis_client.get(key)
                    if cached_data:
                        entry = json.loads(cached_data)
                        if not self._is_expired(entry):
                            self.cache_stats["hits"] += 1
                            self.cache_stats["redis_hits"] += 1
                            return entry["data"]
                        else:
                            # Remove expired entry
                            self.redis_client.delete(key)
                except RedisError as e:
                    logger.warning(f"Redis get error: {e}")
                    self.cache_stats["errors"] += 1
            
            # Fallback to memory cache
            if key in self.memory_cache:
                entry = self.memory_cache[key]
                if not self._is_expired(entry):
                    self.cache_stats["hits"] += 1
                    self.cache_stats["memory_hits"] += 1
                    return entry["data"]
                else:
                    del self.memory_cache[key]
            
            self.cache_stats["misses"] += 1
            return None
            
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            self.cache_stats["errors"] += 1
            return None
    
    async def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """Set value in cache with TTL"""
        try:
            entry = {
                "data": value,
                "created_at": time.time(),
                "expires_at": time.time() + ttl
            }
            
            # Try Redis first
            if self.redis_client:
                try:
                    self.redis_client.setex(
                        key, 
                        ttl, 
                        json.dumps(entry, default=str)
                    )
                    return True
                except RedisError as e:
                    logger.warning(f"Redis set error: {e}")
                    self.cache_stats["errors"] += 1
            
            # Fallback to memory cache
            self.memory_cache[key] = entry
            self._cleanup_memory_cache()
            return True
            
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            self.cache_stats["errors"] += 1
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete key from cache"""
        try:
            deleted = False
            
            # Delete from Redis
            if self.redis_client:
                try:
                    if self.redis_client.delete(key):
                        deleted = True
                except RedisError as e:
                    logger.warning(f"Redis delete error: {e}")
            
            # Delete from memory cache
            if key in self.memory_cache:
                del self.memory_cache[key]
                deleted = True
            
            return deleted
            
        except Exception as e:
            logger.error(f"Cache delete error: {e}")
            return False
    
    async def clear(self) -> bool:
        """Clear all cache"""
        try:
            # Clear Redis
            if self.redis_client:
                try:
                    self.redis_client.flushdb()
                except RedisError as e:
                    logger.warning(f"Redis clear error: {e}")
            
            # Clear memory cache
            self.memory_cache.clear()
            return True
            
        except Exception as e:
            logger.error(f"Cache clear error: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.cache_stats["hits"] + self.cache_stats["misses"]
        hit_rate = (self.cache_stats["hits"] / total_requests * 100) if total_requests > 0 else 0
        
        return {
            **self.cache_stats,
            "total_requests": total_requests,
            "hit_rate": round(hit_rate, 2),
            "memory_cache_size": len(self.memory_cache),
            "redis_available": self.redis_client is not None
        }


# Global cache manager instance
cache_manager = CacheManager()


def cached(prefix: str = "default", ttl: int = 3600):
    """Decorator for caching function results"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = cache_manager._generate_key(prefix, func.__name__, *args, **kwargs)
            
            # Try to get from cache
            cached_result = await cache_manager.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = await func(*args, **kwargs)
            await cache_manager.set(cache_key, result, ttl)
            
            return result
        return wrapper
    return decorator


def cache_invalidate(prefix: str):
    """Decorator for invalidating cache on function execution"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Execute function
            result = await func(*args, **kwargs)
            
            # Invalidate cache entries with matching prefix
            try:
                if cache_manager.redis_client:
                    pattern = f"{prefix}:*"
                    keys = cache_manager.redis_client.keys(pattern)
                    if keys:
                        cache_manager.redis_client.delete(*keys)
                
                # Remove from memory cache
                keys_to_remove = [
                    key for key in cache_manager.memory_cache.keys()
                    if key.startswith(f"{prefix}:")
                ]
                for key in keys_to_remove:
                    del cache_manager.memory_cache[key]
                    
            except Exception as e:
                logger.warning(f"Cache invalidation error: {e}")
            
            return result
        return wrapper
    return decorator





