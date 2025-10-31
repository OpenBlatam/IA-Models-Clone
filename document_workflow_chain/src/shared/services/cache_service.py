"""
Cache Service
=============

Advanced caching service with multiple backends and intelligent invalidation.
"""

from __future__ import annotations
import asyncio
import logging
import json
import pickle
import hashlib
from typing import Dict, List, Optional, Any, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import redis.asyncio as redis
import memcached
from collections import OrderedDict
import threading
import time

from ..utils.helpers import DateTimeHelpers
from ..utils.decorators import log_execution, measure_performance


logger = logging.getLogger(__name__)


class CacheBackend(str, Enum):
    """Cache backend enumeration"""
    MEMORY = "memory"
    REDIS = "redis"
    MEMCACHED = "memcached"
    HYBRID = "hybrid"


class CacheStrategy(str, Enum):
    """Cache strategy enumeration"""
    LRU = "lru"
    LFU = "lfu"
    TTL = "ttl"
    WRITE_THROUGH = "write_through"
    WRITE_BEHIND = "write_behind"
    WRITE_AROUND = "write_around"


class CacheCompression(str, Enum):
    """Cache compression enumeration"""
    NONE = "none"
    GZIP = "gzip"
    LZ4 = "lz4"
    ZSTD = "zstd"


@dataclass
class CacheConfig:
    """Cache configuration"""
    backend: CacheBackend = CacheBackend.MEMORY
    strategy: CacheStrategy = CacheStrategy.LRU
    compression: CacheCompression = CacheCompression.NONE
    max_size: int = 1000
    default_ttl: int = 3600  # seconds
    redis_url: str = "redis://localhost:6379"
    memcached_servers: List[str] = field(default_factory=lambda: ["localhost:11211"])
    enable_compression: bool = False
    compression_threshold: int = 1024  # bytes
    enable_metrics: bool = True
    enable_invalidation: bool = True


@dataclass
class CacheEntry:
    """Cache entry representation"""
    key: str
    value: Any
    created_at: datetime
    expires_at: Optional[datetime] = None
    access_count: int = 0
    last_accessed: datetime = field(default_factory=DateTimeHelpers.now_utc)
    size_bytes: int = 0
    compressed: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CacheMetrics:
    """Cache metrics representation"""
    hits: int = 0
    misses: int = 0
    sets: int = 0
    deletes: int = 0
    evictions: int = 0
    total_size_bytes: int = 0
    entry_count: int = 0
    hit_rate: float = 0.0
    miss_rate: float = 0.0


class MemoryCache:
    """In-memory cache implementation"""
    
    def __init__(self, max_size: int = 1000, strategy: CacheStrategy = CacheStrategy.LRU):
        self.max_size = max_size
        self.strategy = strategy
        self._cache: Dict[str, CacheEntry] = {}
        self._access_order: List[str] = []
        self._lock = asyncio.Lock()
        self._metrics = CacheMetrics()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        async with self._lock:
            if key not in self._cache:
                self._metrics.misses += 1
                return None
            
            entry = self._cache[key]
            
            # Check expiration
            if entry.expires_at and DateTimeHelpers.now_utc() > entry.expires_at:
                del self._cache[key]
                if key in self._access_order:
                    self._access_order.remove(key)
                self._metrics.misses += 1
                return None
            
            # Update access tracking
            entry.access_count += 1
            entry.last_accessed = DateTimeHelpers.now_utc()
            
            if self.strategy == CacheStrategy.LRU:
                if key in self._access_order:
                    self._access_order.remove(key)
                self._access_order.append(key)
            
            self._metrics.hits += 1
            self._update_metrics()
            
            return entry.value
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache"""
        async with self._lock:
            # Calculate expiration
            expires_at = None
            if ttl:
                expires_at = DateTimeHelpers.now_utc() + timedelta(seconds=ttl)
            
            # Create entry
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=DateTimeHelpers.now_utc(),
                expires_at=expires_at,
                size_bytes=len(str(value).encode('utf-8'))
            )
            
            # Check if we need to evict
            if len(self._cache) >= self.max_size and key not in self._cache:
                await self._evict_entry()
            
            self._cache[key] = entry
            
            if self.strategy == CacheStrategy.LRU:
                if key in self._access_order:
                    self._access_order.remove(key)
                self._access_order.append(key)
            
            self._metrics.sets += 1
            self._update_metrics()
    
    async def delete(self, key: str) -> bool:
        """Delete key from cache"""
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                if key in self._access_order:
                    self._access_order.remove(key)
                self._metrics.deletes += 1
                self._update_metrics()
                return True
            return False
    
    async def clear(self) -> None:
        """Clear all cache entries"""
        async with self._lock:
            self._cache.clear()
            self._access_order.clear()
            self._metrics = CacheMetrics()
    
    async def _evict_entry(self) -> None:
        """Evict entry based on strategy"""
        if not self._cache:
            return
        
        if self.strategy == CacheStrategy.LRU:
            # Remove least recently used
            if self._access_order:
                oldest_key = self._access_order.pop(0)
                if oldest_key in self._cache:
                    del self._cache[oldest_key]
                    self._metrics.evictions += 1
        
        elif self.strategy == CacheStrategy.LFU:
            # Remove least frequently used
            least_used_key = min(self._cache.keys(), key=lambda k: self._cache[k].access_count)
            del self._cache[least_used_key]
            if least_used_key in self._access_order:
                self._access_order.remove(least_used_key)
            self._metrics.evictions += 1
        
        elif self.strategy == CacheStrategy.TTL:
            # Remove expired entries
            now = DateTimeHelpers.now_utc()
            expired_keys = [
                key for key, entry in self._cache.items()
                if entry.expires_at and entry.expires_at < now
            ]
            for key in expired_keys:
                del self._cache[key]
                if key in self._access_order:
                    self._access_order.remove(key)
                self._metrics.evictions += 1
    
    def _update_metrics(self) -> None:
        """Update cache metrics"""
        total_requests = self._metrics.hits + self._metrics.misses
        if total_requests > 0:
            self._metrics.hit_rate = self._metrics.hits / total_requests
            self._metrics.miss_rate = self._metrics.misses / total_requests
        
        self._metrics.entry_count = len(self._cache)
        self._metrics.total_size_bytes = sum(entry.size_bytes for entry in self._cache.values())
    
    def get_metrics(self) -> CacheMetrics:
        """Get cache metrics"""
        return self._metrics


class RedisCache:
    """Redis cache implementation"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self._redis: Optional[redis.Redis] = None
        self._metrics = CacheMetrics()
    
    async def connect(self) -> None:
        """Connect to Redis"""
        if not self._redis:
            self._redis = redis.from_url(self.redis_url)
            await self._redis.ping()
            logger.info("Connected to Redis cache")
    
    async def disconnect(self) -> None:
        """Disconnect from Redis"""
        if self._redis:
            await self._redis.close()
            self._redis = None
            logger.info("Disconnected from Redis cache")
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from Redis cache"""
        await self.connect()
        
        try:
            value = await self._redis.get(key)
            if value is None:
                self._metrics.misses += 1
                return None
            
            # Deserialize value
            data = json.loads(value)
            self._metrics.hits += 1
            return data
        
        except Exception as e:
            logger.error(f"Redis get error: {e}")
            self._metrics.misses += 1
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in Redis cache"""
        await self.connect()
        
        try:
            # Serialize value
            serialized_value = json.dumps(value)
            
            if ttl:
                await self._redis.setex(key, ttl, serialized_value)
            else:
                await self._redis.set(key, serialized_value)
            
            self._metrics.sets += 1
        
        except Exception as e:
            logger.error(f"Redis set error: {e}")
    
    async def delete(self, key: str) -> bool:
        """Delete key from Redis cache"""
        await self.connect()
        
        try:
            result = await self._redis.delete(key)
            if result:
                self._metrics.deletes += 1
                return True
            return False
        
        except Exception as e:
            logger.error(f"Redis delete error: {e}")
            return False
    
    async def clear(self) -> None:
        """Clear all cache entries"""
        await self.connect()
        
        try:
            await self._redis.flushdb()
            self._metrics = CacheMetrics()
        except Exception as e:
            logger.error(f"Redis clear error: {e}")
    
    def get_metrics(self) -> CacheMetrics:
        """Get cache metrics"""
        return self._metrics


class CacheService:
    """Advanced cache service with multiple backends"""
    
    def __init__(self, config: Optional[CacheConfig] = None):
        self.config = config or CacheConfig()
        self._memory_cache: Optional[MemoryCache] = None
        self._redis_cache: Optional[RedisCache] = None
        self._invalidation_patterns: Dict[str, List[str]] = {}
        self._lock = asyncio.Lock()
        self._is_running = False
        self._cleanup_task: Optional[asyncio.Task] = None
        self._initialize_backends()
    
    def _initialize_backends(self):
        """Initialize cache backends"""
        if self.config.backend in [CacheBackend.MEMORY, CacheBackend.HYBRID]:
            self._memory_cache = MemoryCache(
                max_size=self.config.max_size,
                strategy=self.config.strategy
            )
        
        if self.config.backend in [CacheBackend.REDIS, CacheBackend.HYBRID]:
            self._redis_cache = RedisCache(self.config.redis_url)
        
        logger.info(f"Cache service initialized with backend: {self.config.backend.value}")
    
    async def start(self):
        """Start the cache service"""
        if self._is_running:
            return
        
        self._is_running = True
        
        # Connect to Redis if needed
        if self._redis_cache:
            await self._redis_cache.connect()
        
        # Start cleanup task
        self._cleanup_task = asyncio.create_task(self._cleanup_worker())
        
        logger.info("Cache service started")
    
    async def stop(self):
        """Stop the cache service"""
        if not self._is_running:
            return
        
        self._is_running = False
        
        # Stop cleanup task
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Disconnect from Redis
        if self._redis_cache:
            await self._redis_cache.disconnect()
        
        logger.info("Cache service stopped")
    
    @measure_performance
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        # Try memory cache first (if available)
        if self._memory_cache:
            value = await self._memory_cache.get(key)
            if value is not None:
                return value
        
        # Try Redis cache
        if self._redis_cache:
            value = await self._redis_cache.get(key)
            if value is not None:
                # Store in memory cache for faster access
                if self._memory_cache:
                    await self._memory_cache.set(key, value, self.config.default_ttl)
                return value
        
        return None
    
    @measure_performance
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache"""
        ttl = ttl or self.config.default_ttl
        
        # Set in memory cache
        if self._memory_cache:
            await self._memory_cache.set(key, value, ttl)
        
        # Set in Redis cache
        if self._redis_cache:
            await self._redis_cache.set(key, value, ttl)
    
    async def delete(self, key: str) -> bool:
        """Delete key from cache"""
        deleted = False
        
        # Delete from memory cache
        if self._memory_cache:
            if await self._memory_cache.delete(key):
                deleted = True
        
        # Delete from Redis cache
        if self._redis_cache:
            if await self._redis_cache.delete(key):
                deleted = True
        
        return deleted
    
    async def clear(self) -> None:
        """Clear all cache entries"""
        # Clear memory cache
        if self._memory_cache:
            await self._memory_cache.clear()
        
        # Clear Redis cache
        if self._redis_cache:
            await self._redis_cache.clear()
    
    async def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate cache entries matching pattern"""
        if not self.config.enable_invalidation:
            return 0
        
        invalidated_count = 0
        
        # Invalidate in memory cache
        if self._memory_cache:
            async with self._memory_cache._lock:
                keys_to_delete = []
                for key in self._memory_cache._cache.keys():
                    if self._match_pattern(key, pattern):
                        keys_to_delete.append(key)
                
                for key in keys_to_delete:
                    await self._memory_cache.delete(key)
                    invalidated_count += 1
        
        # Invalidate in Redis cache
        if self._redis_cache:
            try:
                await self._redis_cache.connect()
                keys = await self._redis_cache._redis.keys(pattern)
                if keys:
                    await self._redis_cache._redis.delete(*keys)
                    invalidated_count += len(keys)
            except Exception as e:
                logger.error(f"Redis pattern invalidation error: {e}")
        
        logger.info(f"Invalidated {invalidated_count} cache entries matching pattern: {pattern}")
        return invalidated_count
    
    def _match_pattern(self, key: str, pattern: str) -> bool:
        """Match key against pattern"""
        import fnmatch
        return fnmatch.fnmatch(key, pattern)
    
    async def register_invalidation_pattern(self, pattern: str, trigger_keys: List[str]) -> None:
        """Register invalidation pattern"""
        async with self._lock:
            self._invalidation_patterns[pattern] = trigger_keys
    
    async def _cleanup_worker(self):
        """Cleanup expired entries periodically"""
        while self._is_running:
            try:
                # Cleanup memory cache
                if self._memory_cache:
                    await self._memory_cache._evict_entry()
                
                await asyncio.sleep(60)  # Cleanup every minute
            
            except Exception as e:
                logger.error(f"Cache cleanup error: {e}")
                await asyncio.sleep(60)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get cache metrics"""
        metrics = {
            "backend": self.config.backend.value,
            "strategy": self.config.strategy.value,
            "compression": self.config.compression.value,
            "timestamp": DateTimeHelpers.now_utc().isoformat()
        }
        
        # Memory cache metrics
        if self._memory_cache:
            memory_metrics = self._memory_cache.get_metrics()
            metrics["memory_cache"] = {
                "hits": memory_metrics.hits,
                "misses": memory_metrics.misses,
                "sets": memory_metrics.sets,
                "deletes": memory_metrics.deletes,
                "evictions": memory_metrics.evictions,
                "entry_count": memory_metrics.entry_count,
                "total_size_bytes": memory_metrics.total_size_bytes,
                "hit_rate": memory_metrics.hit_rate,
                "miss_rate": memory_metrics.miss_rate
            }
        
        # Redis cache metrics
        if self._redis_cache:
            redis_metrics = self._redis_cache.get_metrics()
            metrics["redis_cache"] = {
                "hits": redis_metrics.hits,
                "misses": redis_metrics.misses,
                "sets": redis_metrics.sets,
                "deletes": redis_metrics.deletes,
                "evictions": redis_metrics.evictions,
                "entry_count": redis_metrics.entry_count,
                "total_size_bytes": redis_metrics.total_size_bytes,
                "hit_rate": redis_metrics.hit_rate,
                "miss_rate": redis_metrics.miss_rate
            }
        
        return metrics
    
    async def warm_cache(self, key_value_pairs: Dict[str, Any], ttl: Optional[int] = None) -> None:
        """Warm cache with initial data"""
        for key, value in key_value_pairs.items():
            await self.set(key, value, ttl)
        
        logger.info(f"Cache warmed with {len(key_value_pairs)} entries")
    
    def generate_cache_key(self, prefix: str, *args, **kwargs) -> str:
        """Generate consistent cache key"""
        # Create key components
        components = [prefix]
        
        # Add positional arguments
        for arg in args:
            components.append(str(arg))
        
        # Add keyword arguments (sorted for consistency)
        for key, value in sorted(kwargs.items()):
            components.append(f"{key}:{value}")
        
        # Join components
        key = ":".join(components)
        
        # Hash if key is too long
        if len(key) > 250:
            key_hash = hashlib.md5(key.encode()).hexdigest()
            key = f"{prefix}:{key_hash}"
        
        return key


# Global cache service
cache_service = CacheService()


# Utility functions
async def start_cache_service():
    """Start the cache service"""
    await cache_service.start()


async def stop_cache_service():
    """Stop the cache service"""
    await cache_service.stop()


async def get_from_cache(key: str) -> Optional[Any]:
    """Get value from cache"""
    return await cache_service.get(key)


async def set_in_cache(key: str, value: Any, ttl: Optional[int] = None) -> None:
    """Set value in cache"""
    await cache_service.set(key, value, ttl)


async def delete_from_cache(key: str) -> bool:
    """Delete key from cache"""
    return await cache_service.delete(key)


async def clear_cache() -> None:
    """Clear all cache entries"""
    await cache_service.clear()


async def invalidate_cache_pattern(pattern: str) -> int:
    """Invalidate cache entries matching pattern"""
    return await cache_service.invalidate_pattern(pattern)


def get_cache_metrics() -> Dict[str, Any]:
    """Get cache metrics"""
    return cache_service.get_metrics()


def generate_cache_key(prefix: str, *args, **kwargs) -> str:
    """Generate consistent cache key"""
    return cache_service.generate_cache_key(prefix, *args, **kwargs)


# Cache decorators
def cache_result(ttl: int = 3600, key_prefix: str = "cache"):
    """Cache function result"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            # Generate cache key
            cache_key = generate_cache_key(key_prefix, func.__name__, *args, **kwargs)
            
            # Try to get from cache
            cached_result = await get_from_cache(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            # Cache result
            await set_in_cache(cache_key, result, ttl)
            
            return result
        
        return wrapper
    return decorator


def cache_invalidate(pattern: str):
    """Invalidate cache after function execution"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            # Execute function
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            # Invalidate cache
            await invalidate_cache_pattern(pattern)
            
            return result
        
        return wrapper
    return decorator




