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
import time
import json
import pickle
import hashlib
import functools
from typing import Dict, List, Any, Optional, Union, Callable, Awaitable, TypeVar, Generic
from datetime import datetime, timezone, timedelta
from contextlib import asynccontextmanager
import structlog
from dataclasses import dataclass, asdict
from enum import Enum
import redis.asyncio as redis
from collections import OrderedDict
import weakref
import threading
import gc
from typing import Any, List, Dict, Optional
import logging
#!/usr/bin/env python3
"""
Cache Manager for HeyGen AI API
Comprehensive caching system for static and frequently accessed data.
"""


logger = structlog.get_logger()

# =============================================================================
# Cache Types
# =============================================================================

class CacheType(Enum):
    """Cache type enumeration."""
    MEMORY = "memory"
    REDIS = "redis"
    HYBRID = "hybrid"
    TIERED = "tiered"

class CacheStrategy(Enum):
    """Cache strategy enumeration."""
    LRU = "lru"
    LFU = "lfu"
    FIFO = "fifo"
    TTL = "ttl"

class CachePriority(Enum):
    """Cache priority enumeration."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class CacheConfig:
    """Cache configuration."""
    cache_type: CacheType = CacheType.HYBRID
    strategy: CacheStrategy = CacheStrategy.LRU
    max_size: int = 1000
    default_ttl: int = 300
    compression_enabled: bool = True
    serialization_format: str = "json"  # json, pickle, msgpack
    enable_stats: bool = True
    enable_eviction: bool = True
    eviction_policy: str = "lru"
    memory_limit_mb: int = 100
    redis_url: Optional[str] = None
    redis_connection_pool_size: int = 10
    redis_socket_timeout: int = 5
    redis_socket_connect_timeout: int = 5
    redis_retry_on_timeout: bool = True

@dataclass
class CacheStats:
    """Cache statistics."""
    hits: int = 0
    misses: int = 0
    sets: int = 0
    deletes: int = 0
    evictions: int = 0
    memory_usage_bytes: int = 0
    redis_operations: int = 0
    last_accessed: Optional[datetime] = None
    created_at: datetime = None
    
    def __post_init__(self) -> Any:
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)
    
    @property
    def hit_rate(self) -> float:
        """Calculate hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    @property
    def memory_usage_mb(self) -> float:
        """Get memory usage in MB."""
        return self.memory_usage_bytes / (1024 * 1024)

# =============================================================================
# Memory Cache Implementation
# =============================================================================

class MemoryCache:
    """In-memory cache implementation with LRU/LFU/FIFO strategies."""
    
    def __init__(
        self,
        max_size: int = 1000,
        strategy: CacheStrategy = CacheStrategy.LRU,
        ttl_enabled: bool = True
    ):
        
    """__init__ function."""
self.max_size = max_size
        self.strategy = strategy
        self.ttl_enabled = ttl_enabled
        
        # Cache storage
        self._cache: Dict[str, Any] = {}
        self._access_order: List[str] = []
        self._access_count: Dict[str, int] = {}
        self._expiry_times: Dict[str, float] = {}
        
        # Statistics
        self.stats = CacheStats()
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Start cleanup task
        self._cleanup_task = None
        self._start_cleanup_task()
    
    def _start_cleanup_task(self) -> Any:
        """Start periodic cleanup task."""
        def cleanup():
            
    """cleanup function."""
while True:
                try:
                    time.sleep(60)  # Cleanup every minute
                    self._cleanup_expired()
                except Exception as e:
                    logger.error(f"Cache cleanup error: {e}")
        
        self._cleanup_task = threading.Thread(target=cleanup, daemon=True)
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        self._cleanup_task.start()
    
    def _cleanup_expired(self) -> Any:
        """Remove expired items from cache."""
        with self._lock:
            current_time = time.time()
            expired_keys = [
                key for key, expiry in self._expiry_times.items()
                if expiry < current_time
            ]
            
            for key in expired_keys:
                self._remove_item(key)
                self.stats.evictions += 1
    
    def _remove_item(self, key: str):
        """Remove item from cache."""
        if key in self._cache:
            del self._cache[key]
        
        if key in self._access_order:
            self._access_order.remove(key)
        
        if key in self._access_count:
            del self._access_count[key]
        
        if key in self._expiry_times:
            del self._expiry_times[key]
    
    def _update_access(self, key: str):
        """Update access tracking based on strategy."""
        if self.strategy == CacheStrategy.LRU:
            # Move to end (most recently used)
            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)
        
        elif self.strategy == CacheStrategy.LFU:
            # Increment access count
            self._access_count[key] = self._access_count.get(key, 0) + 1
        
        elif self.strategy == CacheStrategy.FIFO:
            # No change needed for FIFO
            pass
    
    def _evict_if_needed(self) -> Any:
        """Evict items if cache is full."""
        if len(self._cache) < self.max_size:
            return
        
        if self.strategy == CacheStrategy.LRU:
            # Remove least recently used
            if self._access_order:
                key_to_evict = self._access_order[0]
                self._remove_item(key_to_evict)
                self.stats.evictions += 1
        
        elif self.strategy == CacheStrategy.LFU:
            # Remove least frequently used
            if self._access_count:
                key_to_evict = min(self._access_count.items(), key=lambda x: x[1])[0]
                self._remove_item(key_to_evict)
                self.stats.evictions += 1
        
        elif self.strategy == CacheStrategy.FIFO:
            # Remove first in (oldest)
            if self._access_order:
                key_to_evict = self._access_order[0]
                self._remove_item(key_to_evict)
                self.stats.evictions += 1
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self._lock:
            # Check if expired
            if self.ttl_enabled and key in self._expiry_times:
                if time.time() > self._expiry_times[key]:
                    self._remove_item(key)
                    self.stats.misses += 1
                    return None
            
            if key in self._cache:
                self._update_access(key)
                self.stats.hits += 1
                self.stats.last_accessed = datetime.now(timezone.utc)
                return self._cache[key]
            else:
                self.stats.misses += 1
                return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set value in cache."""
        with self._lock:
            # Add to cache
            self._cache[key] = value
            
            # Set expiry time
            if ttl and self.ttl_enabled:
                self._expiry_times[key] = time.time() + ttl
            
            # Update access tracking
            self._update_access(key)
            
            # Evict if needed
            self._evict_if_needed()
            
            # Update statistics
            self.stats.sets += 1
            self.stats.last_accessed = datetime.now(timezone.utc)
    
    def delete(self, key: str) -> bool:
        """Delete value from cache."""
        with self._lock:
            if key in self._cache:
                self._remove_item(key)
                self.stats.deletes += 1
                return True
            return False
    
    def clear(self) -> Any:
        """Clear all cache."""
        with self._lock:
            self._cache.clear()
            self._access_order.clear()
            self._access_count.clear()
            self._expiry_times.clear()
    
    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        with self._lock:
            if key in self._cache:
                if self.ttl_enabled and key in self._expiry_times:
                    if time.time() > self._expiry_times[key]:
                        self._remove_item(key)
                        return False
                return True
            return False
    
    def keys(self) -> List[str]:
        """Get all cache keys."""
        with self._lock:
            # Clean expired items first
            self._cleanup_expired()
            return list(self._cache.keys())
    
    def size(self) -> int:
        """Get cache size."""
        with self._lock:
            self._cleanup_expired()
            return len(self._cache)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            return {
                "hits": self.stats.hits,
                "misses": self.stats.misses,
                "sets": self.stats.sets,
                "deletes": self.stats.deletes,
                "evictions": self.stats.evictions,
                "hit_rate": self.stats.hit_rate,
                "size": self.size(),
                "max_size": self.max_size,
                "strategy": self.strategy.value,
                "last_accessed": self.stats.last_accessed.isoformat() if self.stats.last_accessed else None,
                "created_at": self.stats.created_at.isoformat()
            }

# =============================================================================
# Redis Cache Implementation
# =============================================================================

class RedisCache:
    """Redis cache implementation."""
    
    def __init__(
        self,
        redis_url: str,
        connection_pool_size: int = 10,
        socket_timeout: int = 5,
        socket_connect_timeout: int = 5,
        retry_on_timeout: bool = True,
        compression_enabled: bool = True
    ):
        
    """__init__ function."""
self.redis_url = redis_url
        self.compression_enabled = compression_enabled
        
        # Create Redis connection pool
        self.redis_pool = redis.ConnectionPool.from_url(
            redis_url,
            max_connections=connection_pool_size,
            socket_timeout=socket_timeout,
            socket_connect_timeout=socket_connect_timeout,
            retry_on_timeout=retry_on_timeout,
            health_check_interval=30
        )
        
        # Statistics
        self.stats = CacheStats()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from Redis cache."""
        try:
            client = redis.Redis(connection_pool=self.redis_pool)
            data = await client.get(key)
            
            if data:
                self.stats.hits += 1
                self.stats.redis_operations += 1
                self.stats.last_accessed = datetime.now(timezone.utc)
                return self._deserialize(data)
            else:
                self.stats.misses += 1
                return None
                
        except Exception as e:
            logger.error(f"Redis get error: {e}")
            self.stats.misses += 1
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set value in Redis cache."""
        try:
            client = redis.Redis(connection_pool=self.redis_pool)
            data = self._serialize(value)
            
            if ttl:
                await client.setex(key, ttl, data)
            else:
                await client.set(key, data)
            
            self.stats.sets += 1
            self.stats.redis_operations += 1
            self.stats.last_accessed = datetime.now(timezone.utc)
            
        except Exception as e:
            logger.error(f"Redis set error: {e}")
    
    async def delete(self, key: str) -> bool:
        """Delete value from Redis cache."""
        try:
            client = redis.Redis(connection_pool=self.redis_pool)
            result = await client.delete(key)
            
            self.stats.deletes += 1
            self.stats.redis_operations += 1
            return result > 0
            
        except Exception as e:
            logger.error(f"Redis delete error: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in Redis cache."""
        try:
            client = redis.Redis(connection_pool=self.redis_pool)
            result = await client.exists(key)
            
            self.stats.redis_operations += 1
            return result > 0
            
        except Exception as e:
            logger.error(f"Redis exists error: {e}")
            return False
    
    async def keys(self, pattern: str = "*") -> List[str]:
        """Get keys matching pattern."""
        try:
            client = redis.Redis(connection_pool=self.redis_pool)
            keys = await client.keys(pattern)
            
            self.stats.redis_operations += 1
            return [key.decode() if isinstance(key, bytes) else key for key in keys]
            
        except Exception as e:
            logger.error(f"Redis keys error: {e}")
            return []
    
    async def clear(self, pattern: str = "*"):
        """Clear cache by pattern."""
        try:
            client = redis.Redis(connection_pool=self.redis_pool)
            keys = await client.keys(pattern)
            
            if keys:
                await client.delete(*keys)
            
            self.stats.redis_operations += 1
            
        except Exception as e:
            logger.error(f"Redis clear error: {e}")
    
    def _serialize(self, value: Any) -> bytes:
        """Serialize value for storage."""
        if self.compression_enabled:
            return pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            return json.dumps(value, default=str).encode()
    
    def _deserialize(self, data: bytes) -> Any:
        """Deserialize value from storage."""
        try:
            if self.compression_enabled:
                return pickle.loads(data)
            else:
                return json.loads(data.decode())
        except Exception as e:
            logger.warning(f"Deserialization error: {e}")
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "hits": self.stats.hits,
            "misses": self.stats.misses,
            "sets": self.stats.sets,
            "deletes": self.stats.deletes,
            "hit_rate": self.stats.hit_rate,
            "redis_operations": self.stats.redis_operations,
            "last_accessed": self.stats.last_accessed.isoformat() if self.stats.last_accessed else None,
            "created_at": self.stats.created_at.isoformat()
        }

# =============================================================================
# Hybrid Cache Implementation
# =============================================================================

class HybridCache:
    """Hybrid cache combining memory and Redis."""
    
    def __init__(
        self,
        memory_cache: MemoryCache,
        redis_cache: RedisCache,
        memory_first: bool = True
    ):
        
    """__init__ function."""
self.memory_cache = memory_cache
        self.redis_cache = redis_cache
        self.memory_first = memory_first
        
        # Statistics
        self.stats = CacheStats()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from hybrid cache."""
        # Try memory cache first if configured
        if self.memory_first:
            value = self.memory_cache.get(key)
            if value is not None:
                self.stats.hits += 1
                return value
            
            # Try Redis cache
            value = await self.redis_cache.get(key)
            if value is not None:
                # Store in memory cache for future access
                self.memory_cache.set(key, value)
                self.stats.hits += 1
                return value
        else:
            # Try Redis cache first
            value = await self.redis_cache.get(key)
            if value is not None:
                self.stats.hits += 1
                return value
            
            # Try memory cache
            value = self.memory_cache.get(key)
            if value is not None:
                self.stats.hits += 1
                return value
        
        self.stats.misses += 1
        return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set value in hybrid cache."""
        # Set in both caches
        self.memory_cache.set(key, value, ttl)
        await self.redis_cache.set(key, value, ttl)
        
        self.stats.sets += 1
        self.stats.last_accessed = datetime.now(timezone.utc)
    
    async def delete(self, key: str) -> bool:
        """Delete value from hybrid cache."""
        memory_result = self.memory_cache.delete(key)
        redis_result = await self.redis_cache.delete(key)
        
        self.stats.deletes += 1
        return memory_result or redis_result
    
    async def clear(self) -> Any:
        """Clear both caches."""
        self.memory_cache.clear()
        await self.redis_cache.clear()
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in either cache."""
        return self.memory_cache.exists(key) or await self.redis_cache.exists(key)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get combined cache statistics."""
        memory_stats = self.memory_cache.get_stats()
        redis_stats = self.redis_cache.get_stats()
        
        return {
            "hits": self.stats.hits,
            "misses": self.stats.misses,
            "sets": self.stats.sets,
            "deletes": self.stats.deletes,
            "hit_rate": self.stats.hit_rate,
            "memory_stats": memory_stats,
            "redis_stats": redis_stats,
            "last_accessed": self.stats.last_accessed.isoformat() if self.stats.last_accessed else None,
            "created_at": self.stats.created_at.isoformat()
        }

# =============================================================================
# Tiered Cache Implementation
# =============================================================================

class TieredCache:
    """Tiered cache with multiple levels."""
    
    def __init__(self, caches: List[Any]):
        
    """__init__ function."""
self.caches = caches
        
        # Statistics
        self.stats = CacheStats()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from tiered cache."""
        for i, cache in enumerate(self.caches):
            if hasattr(cache, 'get'):
                if asyncio.iscoroutinefunction(cache.get):
                    value = await cache.get(key)
                else:
                    value = cache.get(key)
                
                if value is not None:
                    # Promote to higher tiers
                    await self._promote_to_higher_tiers(key, value, i)
                    self.stats.hits += 1
                    return value
        
        self.stats.misses += 1
        return None
    
    async def _promote_to_higher_tiers(self, key: str, value: Any, current_tier: int):
        """Promote value to higher cache tiers."""
        for i in range(current_tier):
            cache = self.caches[i]
            if hasattr(cache, 'set'):
                if asyncio.iscoroutinefunction(cache.set):
                    await cache.set(key, value)
                else:
                    cache.set(key, value)
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set value in all cache tiers."""
        for cache in self.caches:
            if hasattr(cache, 'set'):
                if asyncio.iscoroutinefunction(cache.set):
                    await cache.set(key, value, ttl)
                else:
                    cache.set(key, value, ttl)
        
        self.stats.sets += 1
        self.stats.last_accessed = datetime.now(timezone.utc)
    
    async def delete(self, key: str) -> bool:
        """Delete value from all cache tiers."""
        results = []
        for cache in self.caches:
            if hasattr(cache, 'delete'):
                if asyncio.iscoroutinefunction(cache.delete):
                    result = await cache.delete(key)
                else:
                    result = cache.delete(key)
                results.append(result)
        
        self.stats.deletes += 1
        return any(results)
    
    async def clear(self) -> Any:
        """Clear all cache tiers."""
        for cache in self.caches:
            if hasattr(cache, 'clear'):
                if asyncio.iscoroutinefunction(cache.clear):
                    await cache.clear()
                else:
                    cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get tiered cache statistics."""
        tier_stats = []
        for i, cache in enumerate(self.caches):
            if hasattr(cache, 'get_stats'):
                stats = cache.get_stats()
                tier_stats.append({
                    "tier": i,
                    "type": type(cache).__name__,
                    "stats": stats
                })
        
        return {
            "hits": self.stats.hits,
            "misses": self.stats.misses,
            "sets": self.stats.sets,
            "deletes": self.stats.deletes,
            "hit_rate": self.stats.hit_rate,
            "tiers": tier_stats,
            "last_accessed": self.stats.last_accessed.isoformat() if self.stats.last_accessed else None,
            "created_at": self.stats.created_at.isoformat()
        }

# =============================================================================
# Cache Manager
# =============================================================================

class CacheManager:
    """Main cache manager for the application."""
    
    def __init__(self, config: CacheConfig):
        
    """__init__ function."""
self.config = config
        self.caches: Dict[str, Any] = {}
        
        # Initialize caches based on configuration
        self._initialize_caches()
        
        # Statistics
        self.stats = CacheStats()
    
    def _initialize_caches(self) -> Any:
        """Initialize caches based on configuration."""
        if self.config.cache_type == CacheType.MEMORY:
            self.caches["memory"] = MemoryCache(
                max_size=self.config.max_size,
                strategy=self.config.strategy,
                ttl_enabled=True
            )
        
        elif self.config.cache_type == CacheType.REDIS:
            if not self.config.redis_url:
                raise ValueError("Redis URL required for Redis cache type")
            
            self.caches["redis"] = RedisCache(
                redis_url=self.config.redis_url,
                connection_pool_size=self.config.redis_connection_pool_size,
                socket_timeout=self.config.redis_socket_timeout,
                socket_connect_timeout=self.config.redis_socket_connect_timeout,
                retry_on_timeout=self.config.redis_retry_on_timeout,
                compression_enabled=self.config.compression_enabled
            )
        
        elif self.config.cache_type == CacheType.HYBRID:
            if not self.config.redis_url:
                raise ValueError("Redis URL required for hybrid cache type")
            
            memory_cache = MemoryCache(
                max_size=self.config.max_size,
                strategy=self.config.strategy,
                ttl_enabled=True
            )
            
            redis_cache = RedisCache(
                redis_url=self.config.redis_url,
                connection_pool_size=self.config.redis_connection_pool_size,
                socket_timeout=self.config.redis_socket_timeout,
                socket_connect_timeout=self.config.redis_socket_connect_timeout,
                retry_on_timeout=self.config.redis_retry_on_timeout,
                compression_enabled=self.config.compression_enabled
            )
            
            self.caches["hybrid"] = HybridCache(memory_cache, redis_cache)
        
        elif self.config.cache_type == CacheType.TIERED:
            # Create tiered cache with memory and Redis
            memory_cache = MemoryCache(
                max_size=self.config.max_size // 2,
                strategy=self.config.strategy,
                ttl_enabled=True
            )
            
            redis_cache = RedisCache(
                redis_url=self.config.redis_url,
                connection_pool_size=self.config.redis_connection_pool_size,
                socket_timeout=self.config.redis_socket_timeout,
                socket_connect_timeout=self.config.redis_socket_connect_timeout,
                retry_on_timeout=self.config.redis_retry_on_timeout,
                compression_enabled=self.config.compression_enabled
            )
            
            self.caches["tiered"] = TieredCache([memory_cache, redis_cache])
    
    async def get(self, key: str, cache_name: str = "default") -> Optional[Any]:
        """Get value from cache."""
        cache = self.caches.get(cache_name)
        if not cache:
            raise ValueError(f"Cache '{cache_name}' not found")
        
        value = await cache.get(key) if asyncio.iscoroutinefunction(cache.get) else cache.get(key)
        
        if value is not None:
            self.stats.hits += 1
        else:
            self.stats.misses += 1
        
        self.stats.last_accessed = datetime.now(timezone.utc)
        return value
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None, cache_name: str = "default"):
        """Set value in cache."""
        cache = self.caches.get(cache_name)
        if not cache:
            raise ValueError(f"Cache '{cache_name}' not found")
        
        ttl = ttl or self.config.default_ttl
        
        if asyncio.iscoroutinefunction(cache.set):
            await cache.set(key, value, ttl)
        else:
            cache.set(key, value, ttl)
        
        self.stats.sets += 1
        self.stats.last_accessed = datetime.now(timezone.utc)
    
    async def delete(self, key: str, cache_name: str = "default") -> bool:
        """Delete value from cache."""
        cache = self.caches.get(cache_name)
        if not cache:
            raise ValueError(f"Cache '{cache_name}' not found")
        
        if asyncio.iscoroutinefunction(cache.delete):
            result = await cache.delete(key)
        else:
            result = cache.delete(key)
        
        self.stats.deletes += 1
        return result
    
    async def clear(self, cache_name: str = "default"):
        """Clear cache."""
        cache = self.caches.get(cache_name)
        if not cache:
            raise ValueError(f"Cache '{cache_name}' not found")
        
        if asyncio.iscoroutinefunction(cache.clear):
            await cache.clear()
        else:
            cache.clear()
    
    async def exists(self, key: str, cache_name: str = "default") -> bool:
        """Check if key exists in cache."""
        cache = self.caches.get(cache_name)
        if not cache:
            raise ValueError(f"Cache '{cache_name}' not found")
        
        if asyncio.iscoroutinefunction(cache.exists):
            return await cache.exists(key)
        else:
            return cache.exists(key)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        cache_stats = {}
        for name, cache in self.caches.items():
            if hasattr(cache, 'get_stats'):
                cache_stats[name] = cache.get_stats()
        
        return {
            "hits": self.stats.hits,
            "misses": self.stats.misses,
            "sets": self.stats.sets,
            "deletes": self.stats.deletes,
            "hit_rate": self.stats.hit_rate,
            "caches": cache_stats,
            "config": {
                "cache_type": self.config.cache_type.value,
                "strategy": self.config.strategy.value,
                "max_size": self.config.max_size,
                "default_ttl": self.config.default_ttl
            },
            "last_accessed": self.stats.last_accessed.isoformat() if self.stats.last_accessed else None,
            "created_at": self.stats.created_at.isoformat()
        }

# =============================================================================
# Cache Decorators
# =============================================================================

def cache_result(
    ttl: int = 300,
    cache_name: str = "default",
    key_generator: Optional[Callable] = None
):
    """Decorator to cache function results."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            # Generate cache key
            if key_generator:
                cache_key = key_generator(*args, **kwargs)
            else:
                # Default key generation
                key_data = {
                    "func": func.__name__,
                    "args": str(args),
                    "kwargs": str(sorted(kwargs.items()))
                }
                cache_key = hashlib.md5(json.dumps(key_data, sort_keys=True).encode()).hexdigest()
            
            # Get cache manager from function context
            cache_manager = getattr(wrapper, '_cache_manager', None)
            if not cache_manager:
                # Try to get from global context
                cache_manager = globals().get('cache_manager')
            
            if cache_manager:
                # Try to get from cache
                cached_result = await cache_manager.get(cache_key, cache_name)
                if cached_result is not None:
                    return cached_result
                
                # Execute function
                result = await func(*args, **kwargs)
                
                # Cache result
                await cache_manager.set(cache_key, result, ttl, cache_name)
                
                return result
            else:
                # No cache manager available, just execute function
                return await func(*args, **kwargs)
        
        return wrapper
    return decorator

def invalidate_cache(pattern: str, cache_name: str = "default"):
    """Decorator to invalidate cache after function execution."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            # Execute function
            result = await func(*args, **kwargs)
            
            # Get cache manager
            cache_manager = getattr(wrapper, '_cache_manager', None)
            if not cache_manager:
                cache_manager = globals().get('cache_manager')
            
            if cache_manager:
                # Invalidate cache
                await cache_manager.clear(cache_name)
            
            return result
        
        return wrapper
    return decorator

# =============================================================================
# FastAPI Integration
# =============================================================================

def get_cache_manager() -> CacheManager:
    """Dependency to get cache manager instance."""
    # This would be configured in your FastAPI app
    return CacheManager(CacheConfig())

# =============================================================================
# Export
# =============================================================================

__all__ = [
    "CacheType",
    "CacheStrategy",
    "CachePriority",
    "CacheConfig",
    "CacheStats",
    "MemoryCache",
    "RedisCache",
    "HybridCache",
    "TieredCache",
    "CacheManager",
    "cache_result",
    "invalidate_cache",
    "get_cache_manager",
] 