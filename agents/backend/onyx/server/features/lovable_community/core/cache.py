"""
Advanced caching utilities with Redis support and async capabilities

Supports:
- Redis for distributed caching (production)
- In-memory cache as fallback (development)
- Async operations for better performance
- Automatic serialization/deserialization
"""

import time
import json
import logging
from typing import Optional, Any, Callable, Union
from functools import wraps
from threading import Lock
from enum import Enum

logger = logging.getLogger(__name__)


class CacheBackend(str, Enum):
    """Cache backend types."""
    MEMORY = "memory"
    REDIS = "redis"
    AUTO = "auto"


class CacheEntry:
    """Cache entry with expiration time."""
    
    def __init__(self, value: Any, ttl: int):
        """
        Initialize cache entry.
        
        Args:
            value: Cached value
            ttl: Time to live in seconds
        """
        self.value = value
        self.expires_at = time.time() + ttl
        self.created_at = time.time()
    
    def is_expired(self) -> bool:
        """Check if entry is expired."""
        return time.time() > self.expires_at


class SimpleCache:
    """
    Simple in-memory cache with TTL.
    
    Thread-safe implementation using locks.
    """
    
    def __init__(self, default_ttl: int = 300):
        """
        Initialize cache.
        
        Args:
            default_ttl: Default time to live in seconds
        """
        self._cache: dict[str, CacheEntry] = {}
        self._lock = Lock()
        self.default_ttl = default_ttl
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return None
            
            if entry.is_expired():
                del self._cache[key]
                return None
            
            return entry.value
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (uses default if None)
        """
        with self._lock:
            ttl = ttl or self.default_ttl
            self._cache[key] = CacheEntry(value, ttl)
    
    def delete(self, key: str) -> bool:
        """
        Delete key from cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if deleted, False if not found
        """
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
    
    def cleanup_expired(self) -> int:
        """
        Remove expired entries.
        
        Returns:
            Number of entries removed
        """
        with self._lock:
            expired_keys = [
                key for key, entry in self._cache.items()
                if entry.is_expired()
            ]
            for key in expired_keys:
                del self._cache[key]
            return len(expired_keys)
    
    def size(self) -> int:
        """Get current cache size."""
        with self._lock:
            return len(self._cache)


class RedisCache:
    """
    Redis-based cache with async support.
    
    Falls back to in-memory cache if Redis is unavailable.
    """
    
    def __init__(
        self,
        redis_url: Optional[str] = None,
        default_ttl: int = 300,
        fallback_to_memory: bool = True
    ):
        """
        Initialize Redis cache.
        
        Args:
            redis_url: Redis connection URL (e.g., redis://localhost:6379)
            default_ttl: Default time to live in seconds
            fallback_to_memory: Fallback to in-memory cache if Redis fails
        """
        self.redis_url = redis_url
        self.default_ttl = default_ttl
        self.fallback_to_memory = fallback_to_memory
        self._redis_client: Optional[Any] = None
        self._async_redis_client: Optional[Any] = None
        self._memory_cache: Optional[SimpleCache] = None
        self._initialized = False
        self._async_initialized = False
    
    def _init_redis(self) -> bool:
        """Initialize Redis client (sync)."""
        if self._initialized:
            return self._redis_client is not None
        
        try:
            import redis
            from redis import ConnectionPool
            
            if not self.redis_url:
                logger.warning("Redis URL not provided, using in-memory cache")
                if self.fallback_to_memory:
                    self._memory_cache = SimpleCache(default_ttl=self.default_ttl)
                return False
            
            pool = ConnectionPool.from_url(
                self.redis_url,
                max_connections=50,
                decode_responses=True
            )
            self._redis_client = redis.Redis(connection_pool=pool)
            
            # Test connection
            self._redis_client.ping()
            self._initialized = True
            logger.info("Redis cache initialized successfully")
            return True
        except Exception as e:
            logger.warning(f"Failed to initialize Redis: {e}. Using fallback.")
            if self.fallback_to_memory:
                self._memory_cache = SimpleCache(default_ttl=self.default_ttl)
            self._initialized = True
            return False
    
    async def _init_async_redis(self) -> bool:
        """Initialize async Redis client."""
        if self._async_initialized:
            return self._async_redis_client is not None
        
        try:
            import aioredis
            
            if not self.redis_url:
                logger.warning("Redis URL not provided, using in-memory cache")
                if self.fallback_to_memory:
                    self._memory_cache = SimpleCache(default_ttl=self.default_ttl)
                return False
            
            self._async_redis_client = await aioredis.from_url(
                self.redis_url,
                max_connections=50,
                decode_responses=True
            )
            
            # Test connection
            await self._async_redis_client.ping()
            self._async_initialized = True
            logger.info("Async Redis cache initialized successfully")
            return True
        except Exception as e:
            logger.warning(f"Failed to initialize async Redis: {e}. Using fallback.")
            if self.fallback_to_memory:
                self._memory_cache = SimpleCache(default_ttl=self.default_ttl)
            self._async_initialized = True
            return False
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache (sync).
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        if not self._init_redis():
            if self._memory_cache:
                return self._memory_cache.get(key)
            return None
        
        try:
            value = self._redis_client.get(key)
            if value is None:
                return None
            
            # Deserialize JSON
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                return value
        except Exception as e:
            logger.error(f"Redis get error: {e}")
            if self._memory_cache:
                return self._memory_cache.get(key)
            return None
    
    async def aget(self, key: str) -> Optional[Any]:
        """
        Get value from cache (async).
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        if not await self._init_async_redis():
            if self._memory_cache:
                return self._memory_cache.get(key)
            return None
        
        try:
            value = await self._async_redis_client.get(key)
            if value is None:
                return None
            
            # Deserialize JSON
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                return value
        except Exception as e:
            logger.error(f"Async Redis get error: {e}")
            if self._memory_cache:
                return self._memory_cache.get(key)
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Set value in cache (sync).
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (uses default if None)
        """
        if not self._init_redis():
            if self._memory_cache:
                self._memory_cache.set(key, value, ttl)
            return
        
        try:
            ttl = ttl or self.default_ttl
            
            # Serialize to JSON if needed
            if not isinstance(value, (str, int, float, bool)):
                value = json.dumps(value)
            
            self._redis_client.setex(key, ttl, value)
        except Exception as e:
            logger.error(f"Redis set error: {e}")
            if self._memory_cache:
                self._memory_cache.set(key, value, ttl)
    
    async def aset(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Set value in cache (async).
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (uses default if None)
        """
        if not await self._init_async_redis():
            if self._memory_cache:
                self._memory_cache.set(key, value, ttl)
            return
        
        try:
            ttl = ttl or self.default_ttl
            
            # Serialize to JSON if needed
            if not isinstance(value, (str, int, float, bool)):
                value = json.dumps(value)
            
            await self._async_redis_client.setex(key, ttl, value)
        except Exception as e:
            logger.error(f"Async Redis set error: {e}")
            if self._memory_cache:
                self._memory_cache.set(key, value, ttl)
    
    def delete(self, key: str) -> bool:
        """
        Delete key from cache (sync).
        
        Args:
            key: Cache key
            
        Returns:
            True if deleted, False if not found
        """
        if not self._init_redis():
            if self._memory_cache:
                return self._memory_cache.delete(key)
            return False
        
        try:
            result = self._redis_client.delete(key)
            return bool(result)
        except Exception as e:
            logger.error(f"Redis delete error: {e}")
            if self._memory_cache:
                return self._memory_cache.delete(key)
            return False
    
    async def adelete(self, key: str) -> bool:
        """
        Delete key from cache (async).
        
        Args:
            key: Cache key
            
        Returns:
            True if deleted, False if not found
        """
        if not await self._init_async_redis():
            if self._memory_cache:
                return self._memory_cache.delete(key)
            return False
        
        try:
            result = await self._async_redis_client.delete(key)
            return bool(result)
        except Exception as e:
            logger.error(f"Async Redis delete error: {e}")
            if self._memory_cache:
                return self._memory_cache.delete(key)
            return False
    
    def clear(self) -> None:
        """Clear all cache entries."""
        if self._init_redis():
            try:
                self._redis_client.flushdb()
            except Exception as e:
                logger.error(f"Redis clear error: {e}")
        
        if self._memory_cache:
            self._memory_cache.clear()
    
    async def aclear(self) -> None:
        """Clear all cache entries (async)."""
        if await self._init_async_redis():
            try:
                await self._async_redis_client.flushdb()
            except Exception as e:
                logger.error(f"Async Redis clear error: {e}")
        
        if self._memory_cache:
            self._memory_cache.clear()


class CacheManager:
    """
    Unified cache manager that supports multiple backends.
    
    Automatically selects the best available backend.
    """
    
    def __init__(
        self,
        backend: Union[CacheBackend, str] = CacheBackend.AUTO,
        redis_url: Optional[str] = None,
        default_ttl: int = 300
    ):
        """
        Initialize cache manager.
        
        Args:
            backend: Cache backend to use
            redis_url: Redis connection URL
            default_ttl: Default time to live in seconds
        """
        self.backend = CacheBackend(backend) if isinstance(backend, str) else backend
        self.default_ttl = default_ttl
        self._cache: Optional[Union[SimpleCache, RedisCache]] = None
        
        if self.backend == CacheBackend.REDIS or self.backend == CacheBackend.AUTO:
            self._cache = RedisCache(
                redis_url=redis_url,
                default_ttl=default_ttl,
                fallback_to_memory=True
            )
        else:
            self._cache = SimpleCache(default_ttl=default_ttl)
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache (sync)."""
        return self._cache.get(key)
    
    async def aget(self, key: str) -> Optional[Any]:
        """Get value from cache (async)."""
        if isinstance(self._cache, RedisCache):
            return await self._cache.aget(key)
        return self._cache.get(key)
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache (sync)."""
        self._cache.set(key, value, ttl)
    
    async def aset(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache (async)."""
        if isinstance(self._cache, RedisCache):
            await self._cache.aset(key, value, ttl)
        else:
            self._cache.set(key, value, ttl)
    
    def delete(self, key: str) -> bool:
        """Delete key from cache (sync)."""
        return self._cache.delete(key)
    
    async def adelete(self, key: str) -> bool:
        """Delete key from cache (async)."""
        if isinstance(self._cache, RedisCache):
            return await self._cache.adelete(key)
        return self._cache.delete(key)
    
    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()
    
    async def aclear(self) -> None:
        """Clear all cache entries (async)."""
        if isinstance(self._cache, RedisCache):
            await self._cache.aclear()
        else:
            self._cache.clear()


# Global cache instance
_cache_manager: Optional[CacheManager] = None


def get_cache() -> CacheManager:
    """
    Get global cache manager instance.
    
    Returns:
        CacheManager instance
    """
    global _cache_manager
    if _cache_manager is None:
        from ..config import settings
        
        redis_url = getattr(settings, 'redis_url', None)
        cache_backend = getattr(settings, 'cache_backend', CacheBackend.AUTO)
        cache_ttl = getattr(settings, 'cache_ttl', 300)
        
        _cache_manager = CacheManager(
            backend=cache_backend,
            redis_url=redis_url,
            default_ttl=cache_ttl
        )
    return _cache_manager


def cached(key_prefix: str = "", ttl: Optional[int] = None):
    """
    Decorator for caching function results (sync).
    
    Args:
        key_prefix: Prefix for cache keys
        ttl: Time to live in seconds
        
    Usage:
        @cached(key_prefix="chat", ttl=300)
        def get_chat(chat_id: str):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache = get_cache()
            
            # Generate cache key
            key_parts = [key_prefix, func.__name__]
            key_parts.extend(str(arg) for arg in args)
            key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
            cache_key = ":".join(key_parts)
            
            # Try to get from cache
            cached_value = cache.get(cache_key)
            if cached_value is not None:
                logger.debug(f"Cache hit: {cache_key}")
                return cached_value
            
            # Execute function and cache result
            logger.debug(f"Cache miss: {cache_key}")
            result = func(*args, **kwargs)
            cache.set(cache_key, result, ttl)
            return result
        
        return wrapper
    return decorator


def async_cached(key_prefix: str = "", ttl: Optional[int] = None):
    """
    Decorator for caching async function results.
    
    Args:
        key_prefix: Prefix for cache keys
        ttl: Time to live in seconds
        
    Usage:
        @async_cached(key_prefix="chat", ttl=300)
        async def get_chat(chat_id: str):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            cache = get_cache()
            
            # Generate cache key
            key_parts = [key_prefix, func.__name__]
            key_parts.extend(str(arg) for arg in args)
            key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
            cache_key = ":".join(key_parts)
            
            # Try to get from cache
            cached_value = await cache.aget(cache_key)
            if cached_value is not None:
                logger.debug(f"Cache hit: {cache_key}")
                return cached_value
            
            # Execute function and cache result
            logger.debug(f"Cache miss: {cache_key}")
            result = await func(*args, **kwargs)
            await cache.aset(cache_key, result, ttl)
            return result
        
        return wrapper
    return decorator
