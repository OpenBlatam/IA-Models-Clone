"""
Advanced caching system for Facebook Posts API
Multi-level caching with Redis and in-memory support
"""

import json
import time
import hashlib
import asyncio
from typing import Any, Dict, Optional, Union, List, Callable
from datetime import datetime, timedelta
import structlog
from abc import ABC, abstractmethod

try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from ..core.config import get_settings

logger = structlog.get_logger(__name__)


class CacheBackend(ABC):
    """Abstract base class for cache backends"""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        pass
    
    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache"""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete value from cache"""
        pass
    
    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        pass
    
    @abstractmethod
    async def clear(self) -> bool:
        """Clear all cache entries"""
        pass


class InMemoryCacheBackend(CacheBackend):
    """In-memory cache backend"""
    
    def __init__(self, default_ttl: int = 3600):
        self.cache = {}
        self.ttl = {}
        self.default_ttl = default_ttl
        self.cleanup_interval = 300  # 5 minutes
        self.last_cleanup = time.time()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if key in self.cache:
            if time.time() < self.ttl.get(key, 0):
                return self.cache[key]
            else:
                # Expired
                del self.cache[key]
                del self.ttl[key]
        return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache"""
        try:
            self.cache[key] = value
            self.ttl[key] = time.time() + (ttl or self.default_ttl)
            
            # Periodic cleanup
            if time.time() - self.last_cleanup > self.cleanup_interval:
                await self._cleanup_expired()
            
            return True
        except Exception as e:
            logger.error("Failed to set cache value", key=key, error=str(e))
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete value from cache"""
        try:
            self.cache.pop(key, None)
            self.ttl.pop(key, None)
            return True
        except Exception as e:
            logger.error("Failed to delete cache value", key=key, error=str(e))
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        if key in self.cache:
            if time.time() < self.ttl.get(key, 0):
                return True
            else:
                # Expired
                del self.cache[key]
                del self.ttl[key]
        return False
    
    async def clear(self) -> bool:
        """Clear all cache entries"""
        try:
            self.cache.clear()
            self.ttl.clear()
            return True
        except Exception as e:
            logger.error("Failed to clear cache", error=str(e))
            return False
    
    async def _cleanup_expired(self):
        """Clean up expired entries"""
        current_time = time.time()
        expired_keys = [
            key for key, expiry in self.ttl.items()
            if expiry <= current_time
        ]
        
        for key in expired_keys:
            self.cache.pop(key, None)
            self.ttl.pop(key, None)
        
        self.last_cleanup = current_time
        
        if expired_keys:
            logger.debug("Cleaned up expired cache entries", count=len(expired_keys))


class RedisCacheBackend(CacheBackend):
    """Redis cache backend"""
    
    def __init__(self, redis_url: str, default_ttl: int = 3600):
        if not REDIS_AVAILABLE:
            raise ImportError("Redis is not available. Install redis package.")
        
        self.redis_url = redis_url
        self.default_ttl = default_ttl
        self._redis = None
        self._connection_pool = None
    
    async def _get_redis(self) -> redis.Redis:
        """Get Redis connection"""
        if self._redis is None:
            self._connection_pool = redis.ConnectionPool.from_url(
                self.redis_url,
                max_connections=20,
                retry_on_timeout=True
            )
            self._redis = redis.Redis(connection_pool=self._connection_pool)
        return self._redis
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        try:
            redis_client = await self._get_redis()
            value = await redis_client.get(key)
            
            if value is None:
                return None
            
            # Try to deserialize JSON
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                return value.decode('utf-8') if isinstance(value, bytes) else value
                
        except Exception as e:
            logger.error("Failed to get cache value", key=key, error=str(e))
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache"""
        try:
            redis_client = await self._get_redis()
            
            # Serialize value
            if isinstance(value, (dict, list)):
                serialized_value = json.dumps(value)
            else:
                serialized_value = str(value)
            
            # Set with TTL
            ttl_seconds = ttl or self.default_ttl
            await redis_client.setex(key, ttl_seconds, serialized_value)
            
            return True
            
        except Exception as e:
            logger.error("Failed to set cache value", key=key, error=str(e))
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete value from cache"""
        try:
            redis_client = await self._get_redis()
            result = await redis_client.delete(key)
            return result > 0
            
        except Exception as e:
            logger.error("Failed to delete cache value", key=key, error=str(e))
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        try:
            redis_client = await self._get_redis()
            result = await redis_client.exists(key)
            return result > 0
            
        except Exception as e:
            logger.error("Failed to check cache existence", key=key, error=str(e))
            return False
    
    async def clear(self) -> bool:
        """Clear all cache entries"""
        try:
            redis_client = await self._get_redis()
            await redis_client.flushdb()
            return True
            
        except Exception as e:
            logger.error("Failed to clear cache", error=str(e))
            return False
    
    async def close(self):
        """Close Redis connection"""
        if self._redis:
            await self._redis.close()
        if self._connection_pool:
            await self._connection_pool.disconnect()


class MultiLevelCache:
    """Multi-level cache with L1 (in-memory) and L2 (Redis)"""
    
    def __init__(self, l1_ttl: int = 300, l2_ttl: int = 3600, redis_url: Optional[str] = None):
        self.l1_cache = InMemoryCacheBackend(l1_ttl)
        self.l2_cache = None
        
        if redis_url and REDIS_AVAILABLE:
            try:
                self.l2_cache = RedisCacheBackend(redis_url, l2_ttl)
                logger.info("Multi-level cache initialized with Redis")
            except Exception as e:
                logger.warning("Failed to initialize Redis cache", error=str(e))
                self.l2_cache = None
        
        if not self.l2_cache:
            logger.info("Multi-level cache initialized with in-memory only")
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache (L1 first, then L2)"""
        # Try L1 cache first
        value = await self.l1_cache.get(key)
        if value is not None:
            logger.debug("Cache hit L1", key=key)
            return value
        
        # Try L2 cache if available
        if self.l2_cache:
            value = await self.l2_cache.get(key)
            if value is not None:
                # Promote to L1 cache
                await self.l1_cache.set(key, value)
                logger.debug("Cache hit L2", key=key)
                return value
        
        logger.debug("Cache miss", key=key)
        return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache (both L1 and L2)"""
        success = True
        
        # Set in L1 cache
        if not await self.l1_cache.set(key, value, ttl):
            success = False
        
        # Set in L2 cache if available
        if self.l2_cache:
            if not await self.l2_cache.set(key, value, ttl):
                success = False
        
        if success:
            logger.debug("Cache set", key=key)
        else:
            logger.warning("Cache set failed", key=key)
        
        return success
    
    async def delete(self, key: str) -> bool:
        """Delete value from cache (both L1 and L2)"""
        success = True
        
        # Delete from L1 cache
        if not await self.l1_cache.delete(key):
            success = False
        
        # Delete from L2 cache if available
        if self.l2_cache:
            if not await self.l2_cache.delete(key):
                success = False
        
        return success
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        # Check L1 first
        if await self.l1_cache.exists(key):
            return True
        
        # Check L2 if available
        if self.l2_cache:
            return await self.l2_cache.exists(key)
        
        return False
    
    async def clear(self) -> bool:
        """Clear all cache entries"""
        success = True
        
        # Clear L1 cache
        if not await self.l1_cache.clear():
            success = False
        
        # Clear L2 cache if available
        if self.l2_cache:
            if not await self.l2_cache.clear():
                success = False
        
        return success
    
    async def close(self):
        """Close cache connections"""
        if self.l2_cache:
            await self.l2_cache.close()


class CacheManager:
    """Advanced cache manager with key generation and invalidation"""
    
    def __init__(self, cache: MultiLevelCache):
        self.cache = cache
        self.key_prefix = "facebook_posts"
        self.key_separator = ":"
    
    def _generate_key(self, *parts: str) -> str:
        """Generate cache key from parts"""
        key_parts = [self.key_prefix] + list(parts)
        return self.key_separator.join(key_parts)
    
    def _generate_hash_key(self, data: Any) -> str:
        """Generate hash key from data"""
        if isinstance(data, dict):
            # Sort keys for consistent hashing
            data_str = json.dumps(data, sort_keys=True)
        else:
            data_str = str(data)
        
        return hashlib.md5(data_str.encode()).hexdigest()[:8]
    
    async def get_post(self, post_id: str) -> Optional[Dict[str, Any]]:
        """Get post from cache"""
        key = self._generate_key("post", post_id)
        return await self.cache.get(key)
    
    async def set_post(self, post_id: str, post_data: Dict[str, Any], ttl: int = 3600) -> bool:
        """Set post in cache"""
        key = self._generate_key("post", post_id)
        return await self.cache.set(key, post_data, ttl)
    
    async def delete_post(self, post_id: str) -> bool:
        """Delete post from cache"""
        key = self._generate_key("post", post_id)
        return await self.cache.delete(key)
    
    async def get_posts_list(self, filters: Dict[str, Any], skip: int, limit: int) -> Optional[List[Dict[str, Any]]]:
        """Get posts list from cache"""
        filter_hash = self._generate_hash_key(filters)
        key = self._generate_key("posts_list", filter_hash, str(skip), str(limit))
        return await self.cache.get(key)
    
    async def set_posts_list(self, filters: Dict[str, Any], skip: int, limit: int, posts: List[Dict[str, Any]], ttl: int = 300) -> bool:
        """Set posts list in cache"""
        filter_hash = self._generate_hash_key(filters)
        key = self._generate_key("posts_list", filter_hash, str(skip), str(limit))
        return await self.cache.set(key, posts, ttl)
    
    async def get_analytics(self, post_id: str) -> Optional[Dict[str, Any]]:
        """Get analytics from cache"""
        key = self._generate_key("analytics", post_id)
        return await self.cache.get(key)
    
    async def set_analytics(self, post_id: str, analytics_data: Dict[str, Any], ttl: int = 1800) -> bool:
        """Set analytics in cache"""
        key = self._generate_key("analytics", post_id)
        return await self.cache.set(key, analytics_data, ttl)
    
    async def get_metrics(self) -> Optional[Dict[str, Any]]:
        """Get system metrics from cache"""
        key = self._generate_key("metrics")
        return await self.cache.get(key)
    
    async def set_metrics(self, metrics_data: Dict[str, Any], ttl: int = 60) -> bool:
        """Set system metrics in cache"""
        key = self._generate_key("metrics")
        return await self.cache.set(key, metrics_data, ttl)
    
    async def invalidate_post_related(self, post_id: str) -> bool:
        """Invalidate all cache entries related to a post"""
        success = True
        
        # Delete post itself
        if not await self.delete_post(post_id):
            success = False
        
        # Delete analytics
        analytics_key = self._generate_key("analytics", post_id)
        if not await self.cache.delete(analytics_key):
            success = False
        
        # Note: Posts list cache invalidation would require more complex logic
        # For now, we'll let it expire naturally
        
        return success
    
    async def invalidate_all_posts(self) -> bool:
        """Invalidate all posts-related cache entries"""
        # This is a simplified implementation
        # In production, you might want to use Redis patterns or maintain a list of keys
        return await self.cache.clear()


def cache_key(*args, **kwargs) -> str:
    """Generate cache key from function arguments"""
    key_parts = []
    
    # Add positional arguments
    for arg in args:
        if isinstance(arg, (dict, list)):
            key_parts.append(json.dumps(arg, sort_keys=True))
        else:
            key_parts.append(str(arg))
    
    # Add keyword arguments
    for key, value in sorted(kwargs.items()):
        if isinstance(value, (dict, list)):
            key_parts.append(f"{key}={json.dumps(value, sort_keys=True)}")
        else:
            key_parts.append(f"{key}={value}")
    
    return hashlib.md5(":".join(key_parts).encode()).hexdigest()


def cached(ttl: int = 3600, key_func: Optional[Callable] = None):
    """Decorator for caching function results"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = f"{func.__name__}:{cache_key(*args, **kwargs)}"
            
            # Try to get from cache
            cache_manager = get_cache_manager()
            cached_result = await cache_manager.cache.get(cache_key)
            
            if cached_result is not None:
                logger.debug("Cache hit", function=func.__name__, key=cache_key)
                return cached_result
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Cache result
            await cache_manager.cache.set(cache_key, result, ttl)
            logger.debug("Cache set", function=func.__name__, key=cache_key)
            
            return result
        
        return wrapper
    return decorator


# Global cache manager instance
_cache_manager: Optional[CacheManager] = None


def get_cache_manager() -> CacheManager:
    """Get global cache manager instance"""
    global _cache_manager
    
    if _cache_manager is None:
        settings = get_settings()
        
        # Initialize multi-level cache
        cache = MultiLevelCache(
            l1_ttl=300,  # 5 minutes
            l2_ttl=settings.cache_default_ttl,
            redis_url=settings.redis_url if settings.enable_caching else None
        )
        
        _cache_manager = CacheManager(cache)
    
    return _cache_manager


async def close_cache():
    """Close cache connections"""
    global _cache_manager
    
    if _cache_manager:
        await _cache_manager.cache.close()
        _cache_manager = None


# Export all classes and functions
__all__ = [
    # Cache backends
    'CacheBackend',
    'InMemoryCacheBackend',
    'RedisCacheBackend',
    
    # Cache implementations
    'MultiLevelCache',
    'CacheManager',
    
    # Utility functions
    'cache_key',
    'cached',
    'get_cache_manager',
    'close_cache',
]






























