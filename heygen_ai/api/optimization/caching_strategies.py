from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

# Constants
BUFFER_SIZE = 1024

import asyncio
import json
import hashlib
import pickle
from typing import (
from datetime import datetime, timezone, timedelta
from functools import wraps
import structlog
from dataclasses import dataclass, asdict
from enum import Enum
import redis.asyncio as redis
from typing import Any, List, Dict, Optional
import logging
#!/usr/bin/env python3
"""
Caching Strategies for HeyGen AI API
Comprehensive caching implementation with Redis, in-memory caching, and cache warming.
"""

    Dict, List, Any, Optional, Union, Callable, Awaitable,
    TypeVar, Generic, Tuple, Set
)

logger = structlog.get_logger()

T = TypeVar('T')
R = TypeVar('R')

# =============================================================================
# Cache Types
# =============================================================================

class CacheType(Enum):
    """Cache types."""
    REDIS = "redis"
    MEMORY = "memory"
    HYBRID = "hybrid"

class CacheStrategy(Enum):
    """Cache strategies."""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    TTL = "ttl"  # Time To Live
    WRITE_THROUGH = "write_through"
    WRITE_BEHIND = "write_behind"
    WRITE_AROUND = "write_around"

# =============================================================================
# Cache Configuration
# =============================================================================

@dataclass
class CacheConfig:
    """Cache configuration."""
    cache_type: CacheType = CacheType.HYBRID
    strategy: CacheStrategy = CacheStrategy.TTL
    default_ttl: int = 300  # 5 minutes
    max_size: int = 1000
    enable_compression: bool = True
    enable_serialization: bool = True
    redis_url: Optional[str] = None
    redis_prefix: str = "heygen:"
    memory_cache_size: int = 100
    compression_threshold: int = 1024  # Compress if larger than 1KB

# =============================================================================
# Cache Entry
# =============================================================================

@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    value: Any
    created_at: datetime
    expires_at: Optional[datetime] = None
    access_count: int = 0
    last_accessed: datetime = None
    size_bytes: int = 0
    compressed: bool = False
    
    def __post_init__(self) -> Any:
        if self.last_accessed is None:
            self.last_accessed = self.created_at
    
    def is_expired(self) -> bool:
        """Check if entry is expired."""
        if self.expires_at is None:
            return False
        return datetime.now(timezone.utc) > self.expires_at
    
    def access(self) -> Any:
        """Record access to entry."""
        self.access_count += 1
        self.last_accessed = datetime.now(timezone.utc)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "key": self.key,
            "value": self.value,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed.isoformat(),
            "size_bytes": self.size_bytes,
            "compressed": self.compressed
        }

# =============================================================================
# Base Cache Interface
# =============================================================================

class BaseCache:
    """Base cache interface."""
    
    def __init__(self, config: CacheConfig):
        
    """__init__ function."""
self.config = config
        self.stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "deletes": 0,
            "evictions": 0
        }
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        raise NotImplementedError
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache."""
        raise NotImplementedError
    
    async def delete(self, key: str) -> bool:
        """Delete value from cache."""
        raise NotImplementedError
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        raise NotImplementedError
    
    async def clear(self) -> bool:
        """Clear all cache entries."""
        raise NotImplementedError
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.stats["hits"] + self.stats["misses"]
        hit_rate = self.stats["hits"] / total_requests if total_requests > 0 else 0
        
        return {
            **self.stats,
            "hit_rate": hit_rate,
            "total_requests": total_requests
        }

# =============================================================================
# Redis Cache Implementation
# =============================================================================

class RedisCache(BaseCache):
    """Redis-based cache implementation."""
    
    def __init__(self, config: CacheConfig):
        
    """__init__ function."""
super().__init__(config)
        self.redis_client: Optional[redis.Redis] = None
        self._connection_lock = asyncio.Lock()
    
    async def _get_redis_client(self) -> redis.Redis:
        """Get Redis client with connection pooling."""
        if self.redis_client is None:
            async with self._connection_lock:
                if self.redis_client is None:
                    self.redis_client = redis.from_url(
                        self.config.redis_url or "redis://localhost:6379",
                        encoding="utf-8",
                        decode_responses=False,  # We handle serialization ourselves
                        max_connections=20
                    )
        return self.redis_client
    
    def _make_key(self, key: str) -> str:
        """Make Redis key with prefix."""
        return f"{self.config.redis_prefix}{key}"
    
    def _serialize(self, value: Any) -> bytes:
        """Serialize value for Redis storage."""
        if self.config.enable_serialization:
            return pickle.dumps(value)
        else:
            if isinstance(value, (dict, list)):
                return json.dumps(value).encode('utf-8')
            else:
                return str(value).encode('utf-8')
    
    def _deserialize(self, data: bytes) -> Any:
        """Deserialize value from Redis storage."""
        if self.config.enable_serialization:
            return pickle.loads(data)
        else:
            try:
                return json.loads(data.decode('utf-8'))
            except (json.JSONDecodeError, UnicodeDecodeError):
                return data.decode('utf-8')
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from Redis cache."""
        try:
            redis_client = await self._get_redis_client()
            redis_key = self._make_key(key)
            
            data = await redis_client.get(redis_key)
            
            if data is not None:
                self.stats["hits"] += 1
                return self._deserialize(data)
            else:
                self.stats["misses"] += 1
                return None
                
        except Exception as e:
            logger.error("Redis get error", key=key, error=str(e))
            self.stats["misses"] += 1
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in Redis cache."""
        try:
            redis_client = await self._get_redis_client()
            redis_key = self._make_key(key)
            
            data = self._serialize(value)
            ttl = ttl or self.config.default_ttl
            
            await redis_client.setex(redis_key, ttl, data)
            self.stats["sets"] += 1
            
            return True
            
        except Exception as e:
            logger.error("Redis set error", key=key, error=str(e))
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete value from Redis cache."""
        try:
            redis_client = await self._get_redis_client()
            redis_key = self._make_key(key)
            
            result = await redis_client.delete(redis_key)
            self.stats["deletes"] += 1
            
            return result > 0
            
        except Exception as e:
            logger.error("Redis delete error", key=key, error=str(e))
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in Redis cache."""
        try:
            redis_client = await self._get_redis_client()
            redis_key = self._make_key(key)
            
            return await redis_client.exists(redis_key) > 0
            
        except Exception as e:
            logger.error("Redis exists error", key=key, error=str(e))
            return False
    
    async def clear(self) -> bool:
        """Clear all cache entries."""
        try:
            redis_client = await self._get_redis_client()
            pattern = f"{self.config.redis_prefix}*"
            
            keys = await redis_client.keys(pattern)
            if keys:
                await redis_client.delete(*keys)
            
            return True
            
        except Exception as e:
            logger.error("Redis clear error", error=str(e))
            return False
    
    async def close(self) -> Any:
        """Close Redis connection."""
        if self.redis_client:
            await self.redis_client.close()
            self.redis_client = None

# =============================================================================
# Memory Cache Implementation
# =============================================================================

class MemoryCache(BaseCache):
    """In-memory cache implementation with LRU eviction."""
    
    def __init__(self, config: CacheConfig):
        
    """__init__ function."""
super().__init__(config)
        self._cache: Dict[str, CacheEntry] = {}
        self._access_order: List[str] = []
        self._cleanup_task: Optional[asyncio.Task] = None
    
    async def start(self) -> Any:
        """Start memory cache with cleanup task."""
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
    
    async def stop(self) -> Any:
        """Stop memory cache."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
    
    async def _cleanup_loop(self) -> Any:
        """Background cleanup loop."""
        while True:
            try:
                await asyncio.sleep(60)  # Cleanup every minute
                await self._cleanup_expired()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Memory cache cleanup error", error=str(e))
    
    async def _cleanup_expired(self) -> Any:
        """Remove expired entries."""
        expired_keys = [
            key for key, entry in self._cache.items()
            if entry.is_expired()
        ]
        
        for key in expired_keys:
            await self.delete(key)
            self.stats["evictions"] += 1
    
    def _evict_if_needed(self) -> Any:
        """Evict entries if cache is full."""
        while len(self._cache) >= self.config.max_size:
            # Remove least recently used entry
            if self._access_order:
                lru_key = self._access_order.pop(0)
                if lru_key in self._cache:
                    del self._cache[lru_key]
                    self.stats["evictions"] += 1
    
    def _update_access_order(self, key: str):
        """Update access order for LRU eviction."""
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from memory cache."""
        entry = self._cache.get(key)
        
        if entry is None:
            self.stats["misses"] += 1
            return None
        
        if entry.is_expired():
            await self.delete(key)
            self.stats["misses"] += 1
            return None
        
        # Update access metadata
        entry.access()
        self._update_access_order(key)
        
        self.stats["hits"] += 1
        return entry.value
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in memory cache."""
        try:
            # Evict if needed
            self._evict_if_needed()
            
            # Create cache entry
            created_at = datetime.now(timezone.utc)
            expires_at = None
            
            if ttl is not None:
                expires_at = created_at + timedelta(seconds=ttl)
            elif self.config.default_ttl > 0:
                expires_at = created_at + timedelta(seconds=self.config.default_ttl)
            
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=created_at,
                expires_at=expires_at,
                size_bytes=len(str(value))
            )
            
            self._cache[key] = entry
            self._update_access_order(key)
            
            self.stats["sets"] += 1
            return True
            
        except Exception as e:
            logger.error("Memory cache set error", key=key, error=str(e))
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete value from memory cache."""
        if key in self._cache:
            del self._cache[key]
            if key in self._access_order:
                self._access_order.remove(key)
            self.stats["deletes"] += 1
            return True
        return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in memory cache."""
        entry = self._cache.get(key)
        if entry and not entry.is_expired():
            return True
        return False
    
    async def clear(self) -> bool:
        """Clear all cache entries."""
        self._cache.clear()
        self._access_order.clear()
        return True

# =============================================================================
# Hybrid Cache Implementation
# =============================================================================

class HybridCache(BaseCache):
    """Hybrid cache using both Redis and memory cache."""
    
    def __init__(self, config: CacheConfig):
        
    """__init__ function."""
super().__init__(config)
        self.redis_cache = RedisCache(config)
        self.memory_cache = MemoryCache(config)
        self.memory_ttl = 60  # Shorter TTL for memory cache
    
    async def start(self) -> Any:
        """Start hybrid cache."""
        await self.memory_cache.start()
    
    async def stop(self) -> Any:
        """Stop hybrid cache."""
        await self.memory_cache.stop()
        await self.redis_cache.close()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from hybrid cache (memory first, then Redis)."""
        # Try memory cache first
        value = await self.memory_cache.get(key)
        if value is not None:
            return value
        
        # Try Redis cache
        value = await self.redis_cache.get(key)
        if value is not None:
            # Store in memory cache for faster access
            await self.memory_cache.set(key, value, self.memory_ttl)
            return value
        
        return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in both caches."""
        # Set in Redis (primary storage)
        redis_success = await self.redis_cache.set(key, value, ttl)
        
        # Set in memory cache (shorter TTL)
        memory_ttl = min(ttl or self.config.default_ttl, self.memory_ttl)
        memory_success = await self.memory_cache.set(key, value, memory_ttl)
        
        return redis_success and memory_success
    
    async def delete(self, key: str) -> bool:
        """Delete value from both caches."""
        redis_success = await self.redis_cache.delete(key)
        memory_success = await self.memory_cache.delete(key)
        
        return redis_success or memory_success
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in either cache."""
        return await self.memory_cache.exists(key) or await self.redis_cache.exists(key)
    
    async def clear(self) -> bool:
        """Clear both caches."""
        redis_success = await self.redis_cache.clear()
        memory_success = await self.memory_cache.clear()
        
        return redis_success and memory_success
    
    def get_stats(self) -> Dict[str, Any]:
        """Get combined cache statistics."""
        redis_stats = self.redis_cache.get_stats()
        memory_stats = self.memory_cache.get_stats()
        
        return {
            "redis": redis_stats,
            "memory": memory_stats,
            "combined": {
                "total_hits": redis_stats["hits"] + memory_stats["hits"],
                "total_misses": redis_stats["misses"] + memory_stats["misses"],
                "total_sets": redis_stats["sets"] + memory_stats["sets"],
                "total_deletes": redis_stats["deletes"] + memory_stats["deletes"],
            }
        }

# =============================================================================
# Cache Decorators
# =============================================================================

def cached(
    ttl: Optional[int] = None,
    key_prefix: str = "",
    cache_type: CacheType = CacheType.HYBRID,
    key_generator: Optional[Callable] = None
):
    """Decorator for caching function results."""
    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            # Generate cache key
            if key_generator:
                cache_key = key_generator(*args, **kwargs)
            else:
                # Default key generation
                key_parts = [key_prefix, func.__name__]
                key_parts.extend([str(arg) for arg in args])
                key_parts.extend([f"{k}:{v}" for k, v in sorted(kwargs.items())])
                cache_key = hashlib.md5("|".join(key_parts).encode()).hexdigest()
            
            # Get cache instance
            cache = get_cache_instance(cache_type)
            
            # Try to get from cache
            cached_result = await cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Store in cache
            await cache.set(cache_key, result, ttl)
            
            return result
        
        return wrapper
    return decorator

def cache_invalidate(pattern: str, cache_type: CacheType = CacheType.HYBRID):
    """Decorator for invalidating cache after function execution."""
    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            # Execute function
            result = await func(*args, **kwargs)
            
            # Invalidate cache
            cache = get_cache_instance(cache_type)
            if hasattr(cache, 'redis_cache'):
                await cache.redis_cache.delete(pattern)
            
            return result
        
        return wrapper
    return decorator

# =============================================================================
# Cache Warming
# =============================================================================

class CacheWarmer:
    """Cache warming utility for preloading frequently accessed data."""
    
    def __init__(self, cache: BaseCache):
        
    """__init__ function."""
self.cache = cache
        self.warming_tasks: List[asyncio.Task] = []
    
    async def warm_cache(
        self,
        data_sources: List[Tuple[str, Callable[[], Awaitable[Any]], Optional[int]]]
    ):
        """Warm cache with data from multiple sources."""
        tasks = []
        
        for key, loader_func, ttl in data_sources:
            task = asyncio.create_task(self._warm_single_key(key, loader_func, ttl))
            tasks.append(task)
        
        # Execute warming tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Log results
        successful = sum(1 for r in results if not isinstance(r, Exception))
        failed = len(results) - successful
        
        logger.info(
            "Cache warming completed",
            total=len(data_sources),
            successful=successful,
            failed=failed
        )
        
        return results
    
    async def _warm_single_key(
        self,
        key: str,
        loader_func: Callable[[], Awaitable[Any]],
        ttl: Optional[int]
    ):
        """Warm a single cache key."""
        try:
            # Load data
            data = await loader_func()
            
            # Store in cache
            await self.cache.set(key, data, ttl)
            
            logger.info("Cache warmed successfully", key=key)
            
        except Exception as e:
            logger.error("Cache warming failed", key=key, error=str(e))
            raise

# =============================================================================
# Cache Factory
# =============================================================================

_cache_instances: Dict[CacheType, BaseCache] = {}

def get_cache_instance(cache_type: CacheType, config: Optional[CacheConfig] = None) -> BaseCache:
    """Get or create cache instance."""
    if cache_type not in _cache_instances:
        if config is None:
            config = CacheConfig(cache_type=cache_type)
        
        if cache_type == CacheType.REDIS:
            _cache_instances[cache_type] = RedisCache(config)
        elif cache_type == CacheType.MEMORY:
            _cache_instances[cache_type] = MemoryCache(config)
        elif cache_type == CacheType.HYBRID:
            _cache_instances[cache_type] = HybridCache(config)
    
    return _cache_instances[cache_type]

async def initialize_caches(configs: Dict[CacheType, CacheConfig]):
    """Initialize all cache instances."""
    for cache_type, config in configs.items():
        cache = get_cache_instance(cache_type, config)
        if hasattr(cache, 'start'):
            await cache.start()

async def shutdown_caches():
    """Shutdown all cache instances."""
    for cache in _cache_instances.values():
        if hasattr(cache, 'stop'):
            await cache.stop()
        elif hasattr(cache, 'close'):
            await cache.close()

# =============================================================================
# Usage Examples
# =============================================================================

@cached(ttl=300, key_prefix="user")
async def get_user_profile(user_id: int) -> Dict[str, Any]:
    """Get user profile with caching."""
    # Simulate database query
    await asyncio.sleep(0.1)
    return {"id": user_id, "name": f"User {user_id}", "email": f"user{user_id}@example.com"}

@cached(ttl=600, key_prefix="video")
async def get_video_metadata(video_id: int) -> Dict[str, Any]:
    """Get video metadata with caching."""
    # Simulate external API call
    await asyncio.sleep(0.2)
    return {"id": video_id, "title": f"Video {video_id}", "duration": 120}

@cache_invalidate("user:*")
async def update_user_profile(user_id: int, data: Dict[str, Any]) -> bool:
    """Update user profile and invalidate cache."""
    # Simulate database update
    await asyncio.sleep(0.1)
    return True

async def example_cache_usage():
    """Example of cache usage."""
    
    # Initialize cache
    config = CacheConfig(
        cache_type=CacheType.HYBRID,
        redis_url="redis://localhost:6379"
    )
    
    cache = get_cache_instance(CacheType.HYBRID, config)
    await cache.start()
    
    try:
        # Use cached functions
        user1 = await get_user_profile(1)
        user2 = await get_user_profile(2)
        
        # Get cache statistics
        stats = cache.get_stats()
        print(f"Cache stats: {stats}")
        
        # Warm cache
        warmer = CacheWarmer(cache)
        await warmer.warm_cache([
            ("user:3", lambda: get_user_profile(3), 300),
            ("video:1", lambda: get_video_metadata(1), 600),
        ])
        
    finally:
        await cache.stop()

# =============================================================================
# Export
# =============================================================================

__all__ = [
    "CacheType",
    "CacheStrategy",
    "CacheConfig",
    "CacheEntry",
    "BaseCache",
    "RedisCache",
    "MemoryCache",
    "HybridCache",
    "cached",
    "cache_invalidate",
    "CacheWarmer",
    "get_cache_instance",
    "initialize_caches",
    "shutdown_caches",
    "example_cache_usage",
] 