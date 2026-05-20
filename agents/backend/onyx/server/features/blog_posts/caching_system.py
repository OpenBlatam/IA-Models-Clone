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
import time
import pickle
import gzip
from typing import (
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from functools import wraps, lru_cache
from enum import Enum
import logging
    import redis.asyncio as redis
    from cachetools import TTLCache, LRUCache, LFUCache
from pydantic import BaseModel, Field
import structlog
from typing import Any, List, Dict, Optional
"""
🚀 COMPREHENSIVE CACHING SYSTEM
==============================

Production-ready caching system with:
- Redis caching for distributed environments
- In-memory caching for high-performance scenarios
- Multiple caching strategies (TTL, LRU, LFU)
- Cache invalidation patterns
- Cache warming and preloading
- Cache monitoring and metrics
- Integration with FastAPI middleware

Features:
- Multi-tier caching (L1: Memory, L2: Redis)
- Automatic cache key generation
- Cache invalidation strategies
- Background cache warming
- Cache hit/miss monitoring
- Distributed cache coordination
- Cache compression and serialization
"""

    Any, Optional, Dict, List, Union, Callable, Awaitable,
    Tuple, Set, TypeVar, Generic
)

try:
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    CACHETOOLS_AVAILABLE = True
except ImportError:
    CACHETOOLS_AVAILABLE = False


# ============================================================================
# CONFIGURATION MODELS
# ============================================================================

class CacheStrategy(str, Enum):
    """Cache eviction strategies."""
    TTL = "ttl"           # Time-based expiration
    LRU = "lru"           # Least Recently Used
    LFU = "lfu"           # Least Frequently Used
    FIFO = "fifo"         # First In, First Out

class CacheTier(str, Enum):
    """Cache tiers for multi-level caching."""
    L1 = "l1"             # In-memory cache (fastest)
    L2 = "l2"             # Redis cache (distributed)
    BOTH = "both"         # Both tiers

class CacheConfig(BaseModel):
    """Configuration for caching system."""
    
    # Redis configuration
    redis_url: Optional[str] = Field(default=None, description="Redis connection URL")
    redis_ttl: int = Field(default=3600, description="Default Redis TTL in seconds")
    redis_max_connections: int = Field(default=20, description="Redis connection pool size")
    redis_retry_on_timeout: bool = Field(default=True, description="Retry on Redis timeout")
    
    # In-memory cache configuration
    memory_cache_size: int = Field(default=1000, description="In-memory cache size")
    memory_cache_ttl: int = Field(default=300, description="In-memory cache TTL in seconds")
    memory_cache_strategy: CacheStrategy = Field(default=CacheStrategy.TTL, description="Memory cache strategy")
    
    # Multi-tier configuration
    enable_multi_tier: bool = Field(default=True, description="Enable multi-tier caching")
    cache_tier: CacheTier = Field(default=CacheTier.BOTH, description="Cache tier to use")
    
    # Cache key configuration
    key_prefix: str = Field(default="cache", description="Cache key prefix")
    key_separator: str = Field(default=":", description="Cache key separator")
    
    # Serialization configuration
    enable_compression: bool = Field(default=True, description="Enable cache compression")
    compression_threshold: int = Field(default=1024, description="Minimum size for compression")
    
    # Monitoring configuration
    enable_monitoring: bool = Field(default=True, description="Enable cache monitoring")
    monitor_interval: int = Field(default=60, description="Monitoring interval in seconds")
    
    # Cache warming configuration
    enable_cache_warming: bool = Field(default=False, description="Enable cache warming")
    warming_batch_size: int = Field(default=100, description="Cache warming batch size")
    
    class Config:
        validate_assignment = True

# ============================================================================
# CACHE KEY GENERATION
# ============================================================================

class CacheKeyGenerator:
    """Generate consistent cache keys."""
    
    def __init__(self, prefix: str = "cache", separator: str = ":"):
        
    """__init__ function."""
self.prefix = prefix
        self.separator = separator
    
    def generate_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments."""
        key_parts = [self.prefix]
        
        # Add positional arguments
        for arg in args:
            key_parts.append(str(arg))
        
        # Add keyword arguments (sorted for consistency)
        for key, value in sorted(kwargs.items()):
            key_parts.append(f"{key}={value}")
        
        return self.separator.join(key_parts)
    
    def generate_hash_key(self, *args, **kwargs) -> str:
        """Generate hash-based cache key."""
        key_string = self.generate_key(*args, **kwargs)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def generate_pattern_key(self, pattern: str, *args, **kwargs) -> str:
        """Generate pattern-based cache key."""
        key_parts = [self.prefix, pattern]
        
        # Add arguments
        for arg in args:
            key_parts.append(str(arg))
        
        # Add keyword arguments
        for key, value in sorted(kwargs.items()):
            key_parts.append(f"{key}={value}")
        
        return self.separator.join(key_parts)

# ============================================================================
# SERIALIZATION UTILITIES
# ============================================================================

class CacheSerializer:
    """Serialize and deserialize cache data."""
    
    def __init__(self, enable_compression: bool = True, compression_threshold: int = 1024):
        
    """__init__ function."""
self.enable_compression = enable_compression
        self.compression_threshold = compression_threshold
    
    def serialize(self, data: Any) -> bytes:
        """Serialize data for caching."""
        # Convert to JSON first
        json_data = json.dumps(data, default=str)
        json_bytes = json_data.encode('utf-8')
        
        # Compress if enabled and data is large enough
        if self.enable_compression and len(json_bytes) > self.compression_threshold:
            compressed = gzip.compress(json_bytes)
            # Add compression marker
            return b"gzip:" + compressed
        
        return json_bytes
    
    def deserialize(self, data: bytes) -> Any:
        """Deserialize cached data."""
        if not data:
            return None
        
        # Check for compression marker
        if data.startswith(b"gzip:"):
            compressed_data = data[5:]  # Remove "gzip:" prefix
            json_bytes = gzip.decompress(compressed_data)
        else:
            json_bytes = data
        
        # Parse JSON
        json_data = json_bytes.decode('utf-8')
        return json.loads(json_data)

# ============================================================================
# IN-MEMORY CACHE
# ============================================================================

class InMemoryCache:
    """In-memory cache implementation."""
    
    def __init__(self, config: CacheConfig):
        
    """__init__ function."""
self.config = config
        self.serializer = CacheSerializer(
            enable_compression=config.enable_compression,
            compression_threshold=config.compression_threshold
        )
        
        # Initialize cache based on strategy
        if config.memory_cache_strategy == CacheStrategy.TTL:
            self.cache = TTLCache(
                maxsize=config.memory_cache_size,
                ttl=config.memory_cache_ttl
            )
        elif config.memory_cache_strategy == CacheStrategy.LRU:
            self.cache = LRUCache(maxsize=config.memory_cache_size)
        elif config.memory_cache_strategy == CacheStrategy.LFU:
            self.cache = LFUCache(maxsize=config.memory_cache_size)
        else:
            # Default to TTL
            self.cache = TTLCache(
                maxsize=config.memory_cache_size,
                ttl=config.memory_cache_ttl
            )
        
        self.logger = structlog.get_logger("memory_cache")
        self.stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "deletes": 0
        }
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        try:
            value = self.cache.get(key)
            if value is not None:
                self.stats["hits"] += 1
                return self.serializer.deserialize(value)
            else:
                self.stats["misses"] += 1
                return None
        except Exception as e:
            self.logger.error("Error getting from memory cache", key=key, error=str(e))
            self.stats["misses"] += 1
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache."""
        try:
            serialized_value = self.serializer.serialize(value)
            self.cache[key] = serialized_value
            self.stats["sets"] += 1
            return True
        except Exception as e:
            self.logger.error("Error setting in memory cache", key=key, error=str(e))
            return False
    
    def delete(self, key: str) -> bool:
        """Delete value from cache."""
        try:
            if key in self.cache:
                del self.cache[key]
                self.stats["deletes"] += 1
                return True
            return False
        except Exception as e:
            self.logger.error("Error deleting from memory cache", key=key, error=str(e))
            return False
    
    def clear(self) -> bool:
        """Clear all cache entries."""
        try:
            self.cache.clear()
            return True
        except Exception as e:
            self.logger.error("Error clearing memory cache", error=str(e))
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.stats["hits"] + self.stats["misses"]
        hit_rate = self.stats["hits"] / total_requests if total_requests > 0 else 0
        
        return {
            "hits": self.stats["hits"],
            "misses": self.stats["misses"],
            "sets": self.stats["sets"],
            "deletes": self.stats["deletes"],
            "hit_rate": hit_rate,
            "size": len(self.cache),
            "max_size": self.config.memory_cache_size
        }

# ============================================================================
# REDIS CACHE
# ============================================================================

class RedisCache:
    """Redis cache implementation."""
    
    def __init__(self, config: CacheConfig):
        
    """__init__ function."""
self.config = config
        self.serializer = CacheSerializer(
            enable_compression=config.enable_compression,
            compression_threshold=config.compression_threshold
        )
        self.redis_client: Optional[redis.Redis] = None
        self.logger = structlog.get_logger("redis_cache")
        self.stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "deletes": 0,
            "errors": 0
        }
        
        if config.redis_url and REDIS_AVAILABLE:
            self._init_redis()
    
    def _init_redis(self) -> Any:
        """Initialize Redis connection."""
        try:
            self.redis_client = redis.from_url(
                self.config.redis_url,
                max_connections=self.config.redis_max_connections,
                retry_on_timeout=self.config.redis_retry_on_timeout,
                decode_responses=False  # We handle serialization ourselves
            )
            self.logger.info("Redis cache initialized", url=self.config.redis_url)
        except Exception as e:
            self.logger.error("Failed to initialize Redis cache", error=str(e))
            self.redis_client = None
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from Redis cache."""
        if not self.redis_client:
            self.stats["misses"] += 1
            return None
        
        try:
            value = await self.redis_client.get(key)
            if value is not None:
                self.stats["hits"] += 1
                return self.serializer.deserialize(value)
            else:
                self.stats["misses"] += 1
                return None
        except Exception as e:
            self.logger.error("Error getting from Redis cache", key=key, error=str(e))
            self.stats["errors"] += 1
            self.stats["misses"] += 1
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in Redis cache."""
        if not self.redis_client:
            return False
        
        try:
            serialized_value = self.serializer.serialize(value)
            ttl = ttl or self.config.redis_ttl
            
            await self.redis_client.setex(key, ttl, serialized_value)
            self.stats["sets"] += 1
            return True
        except Exception as e:
            self.logger.error("Error setting in Redis cache", key=key, error=str(e))
            self.stats["errors"] += 1
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete value from Redis cache."""
        if not self.redis_client:
            return False
        
        try:
            result = await self.redis_client.delete(key)
            if result > 0:
                self.stats["deletes"] += 1
                return True
            return False
        except Exception as e:
            self.logger.error("Error deleting from Redis cache", key=key, error=str(e))
            self.stats["errors"] += 1
            return False
    
    async def clear_pattern(self, pattern: str) -> int:
        """Clear cache entries matching pattern."""
        if not self.redis_client:
            return 0
        
        try:
            keys = await self.redis_client.keys(pattern)
            if keys:
                deleted = await self.redis_client.delete(*keys)
                self.stats["deletes"] += deleted
                return deleted
            return 0
        except Exception as e:
            self.logger.error("Error clearing pattern from Redis cache", pattern=pattern, error=str(e))
            self.stats["errors"] += 1
            return 0
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.stats["hits"] + self.stats["misses"]
        hit_rate = self.stats["hits"] / total_requests if total_requests > 0 else 0
        
        return {
            "hits": self.stats["hits"],
            "misses": self.stats["misses"],
            "sets": self.stats["sets"],
            "deletes": self.stats["deletes"],
            "errors": self.stats["errors"],
            "hit_rate": hit_rate,
            "connected": self.redis_client is not None
        }

# ============================================================================
# MULTI-TIER CACHE
# ============================================================================

class MultiTierCache:
    """Multi-tier caching system (L1: Memory, L2: Redis)."""
    
    def __init__(self, config: CacheConfig):
        
    """__init__ function."""
self.config = config
        self.key_generator = CacheKeyGenerator(
            prefix=config.key_prefix,
            separator=config.key_separator
        )
        
        # Initialize cache tiers
        self.l1_cache = InMemoryCache(config)
        self.l2_cache = RedisCache(config) if config.enable_multi_tier else None
        
        self.logger = structlog.get_logger("multi_tier_cache")
    
    async def get(self, key: str, tier: CacheTier = CacheTier.BOTH) -> Optional[Any]:
        """Get value from cache tiers."""
        # Try L1 cache first
        if tier in [CacheTier.L1, CacheTier.BOTH]:
            value = self.l1_cache.get(key)
            if value is not None:
                self.logger.debug("L1 cache hit", key=key)
                return value
        
        # Try L2 cache if available
        if self.l2_cache and tier in [CacheTier.L2, CacheTier.BOTH]:
            value = await self.l2_cache.get(key)
            if value is not None:
                # Populate L1 cache with L2 result
                if tier == CacheTier.BOTH:
                    self.l1_cache.set(key, value)
                self.logger.debug("L2 cache hit", key=key)
                return value
        
        self.logger.debug("Cache miss", key=key)
        return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None, tier: CacheTier = CacheTier.BOTH) -> bool:
        """Set value in cache tiers."""
        success = True
        
        # Set in L1 cache
        if tier in [CacheTier.L1, CacheTier.BOTH]:
            success &= self.l1_cache.set(key, value, ttl)
        
        # Set in L2 cache
        if self.l2_cache and tier in [CacheTier.L2, CacheTier.BOTH]:
            success &= await self.l2_cache.set(key, value, ttl)
        
        if success:
            self.logger.debug("Cache set", key=key, tier=tier.value)
        
        return success
    
    async def delete(self, key: str, tier: CacheTier = CacheTier.BOTH) -> bool:
        """Delete value from cache tiers."""
        success = True
        
        # Delete from L1 cache
        if tier in [CacheTier.L1, CacheTier.BOTH]:
            success &= self.l1_cache.delete(key)
        
        # Delete from L2 cache
        if self.l2_cache and tier in [CacheTier.L2, CacheTier.BOTH]:
            success &= await self.l2_cache.delete(key)
        
        if success:
            self.logger.debug("Cache deleted", key=key, tier=tier.value)
        
        return success
    
    async def clear_pattern(self, pattern: str) -> int:
        """Clear cache entries matching pattern."""
        deleted = 0
        
        # Clear L1 cache (simple clear for pattern matching)
        if self.l1_cache.clear():
            deleted += 1
        
        # Clear L2 cache pattern
        if self.l2_cache:
            deleted += await self.l2_cache.clear_pattern(pattern)
        
        self.logger.info("Cache pattern cleared", pattern=pattern, deleted=deleted)
        return deleted
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get combined cache statistics."""
        l1_stats = self.l1_cache.get_stats()
        l2_stats = await self.l2_cache.get_stats() if self.l2_cache else {}
        
        return {
            "l1_cache": l1_stats,
            "l2_cache": l2_stats,
            "total_hits": l1_stats.get("hits", 0) + l2_stats.get("hits", 0),
            "total_misses": l1_stats.get("misses", 0) + l2_stats.get("misses", 0),
            "total_sets": l1_stats.get("sets", 0) + l2_stats.get("sets", 0)
        }

# ============================================================================
# CACHE DECORATORS
# ============================================================================

def cached(
    ttl: Optional[int] = None,
    key_prefix: Optional[str] = None,
    tier: CacheTier = CacheTier.BOTH,
    cache_instance: Optional[MultiTierCache] = None
):
    """Decorator for caching function results."""
    def decorator(func: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            if cache_instance is None:
                return await func(*args, **kwargs)
            
            # Generate cache key
            key_parts = [key_prefix or func.__name__]
            key_parts.extend(str(arg) for arg in args)
            key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
            cache_key = ":".join(key_parts)
            
            # Try to get from cache
            cached_value = await cache_instance.get(cache_key, tier)
            if cached_value is not None:
                return cached_value
            
            # Execute function and cache result
            result = await func(*args, **kwargs)
            await cache_instance.set(cache_key, result, ttl, tier)
            
            return result
        
        return wrapper
    return decorator

def cache_invalidate(
    pattern: str,
    cache_instance: Optional[MultiTierCache] = None
):
    """Decorator for invalidating cache after function execution."""
    def decorator(func: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            result = await func(*args, **kwargs)
            
            if cache_instance:
                await cache_instance.clear_pattern(pattern)
            
            return result
        
        return wrapper
    return decorator

# ============================================================================
# CACHE WARMING
# ============================================================================

class CacheWarmer:
    """Cache warming utility."""
    
    def __init__(self, cache_instance: MultiTierCache):
        
    """__init__ function."""
self.cache = cache_instance
        self.logger = structlog.get_logger("cache_warmer")
    
    async def warm_cache(
        self,
        data_source: Callable[[], Awaitable[List[Tuple[str, Any]]]],
        batch_size: int = 100
    ):
        """Warm cache with data from source."""
        try:
            data = await data_source()
            
            for i in range(0, len(data), batch_size):
                batch = data[i:i + batch_size]
                
                # Set cache entries in batch
                for key, value in batch:
                    await self.cache.set(key, value)
                
                self.logger.info(
                    "Cache warming batch completed",
                    batch_start=i,
                    batch_end=min(i + batch_size, len(data)),
                    total=len(data)
                )
                
                # Small delay to prevent overwhelming the cache
                await asyncio.sleep(0.1)
            
            self.logger.info("Cache warming completed", total_entries=len(data))
            
        except Exception as e:
            self.logger.error("Cache warming failed", error=str(e))

# ============================================================================
# CACHE MONITORING
# ============================================================================

class CacheMonitor:
    """Cache monitoring and metrics collection."""
    
    def __init__(self, cache_instance: MultiTierCache, interval: int = 60):
        
    """__init__ function."""
self.cache = cache_instance
        self.interval = interval
        self.logger = structlog.get_logger("cache_monitor")
        self.monitoring_task: Optional[asyncio.Task] = None
    
    async def start_monitoring(self) -> Any:
        """Start cache monitoring."""
        if self.monitoring_task:
            return
        
        self.monitoring_task = asyncio.create_task(self._monitor_loop())
        self.logger.info("Cache monitoring started", interval=self.interval)
    
    async def stop_monitoring(self) -> Any:
        """Stop cache monitoring."""
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
            self.monitoring_task = None
            self.logger.info("Cache monitoring stopped")
    
    async def _monitor_loop(self) -> Any:
        """Monitoring loop."""
        while True:
            try:
                stats = await self.cache.get_stats()
                
                # Log cache statistics
                self.logger.info(
                    "Cache statistics",
                    l1_hit_rate=stats["l1_cache"]["hit_rate"],
                    l2_hit_rate=stats["l2_cache"].get("hit_rate", 0),
                    total_hits=stats["total_hits"],
                    total_misses=stats["total_misses"],
                    l1_size=stats["l1_cache"]["size"],
                    l2_connected=stats["l2_cache"].get("connected", False)
                )
                
                # Check for low hit rates
                if stats["l1_cache"]["hit_rate"] < 0.5:
                    self.logger.warning("Low L1 cache hit rate", hit_rate=stats["l1_cache"]["hit_rate"])
                
                if stats["l2_cache"].get("hit_rate", 0) < 0.3:
                    self.logger.warning("Low L2 cache hit rate", hit_rate=stats["l2_cache"]["hit_rate"])
                
                await asyncio.sleep(self.interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Cache monitoring error", error=str(e))
                await asyncio.sleep(self.interval)

# ============================================================================
# CACHE MANAGER
# ============================================================================

class CacheManager:
    """Main cache manager for the application."""
    
    def __init__(self, config: CacheConfig):
        
    """__init__ function."""
self.config = config
        self.cache = MultiTierCache(config)
        self.key_generator = self.cache.key_generator
        self.warmer = CacheWarmer(self.cache) if config.enable_cache_warming else None
        self.monitor = CacheMonitor(self.cache, config.monitor_interval) if config.enable_monitoring else None
        self.logger = structlog.get_logger("cache_manager")
    
    async def start(self) -> Any:
        """Start cache manager services."""
        if self.monitor:
            await self.monitor.start_monitoring()
        
        self.logger.info("Cache manager started", config=self.config.dict())
    
    async def stop(self) -> Any:
        """Stop cache manager services."""
        if self.monitor:
            await self.monitor.stop_monitoring()
        
        self.logger.info("Cache manager stopped")
    
    async def get(self, key: str, tier: CacheTier = CacheTier.BOTH) -> Optional[Any]:
        """Get value from cache."""
        return await self.cache.get(key, tier)
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None, tier: CacheTier = CacheTier.BOTH) -> bool:
        """Set value in cache."""
        return await self.cache.set(key, value, ttl, tier)
    
    async def delete(self, key: str, tier: CacheTier = CacheTier.BOTH) -> bool:
        """Delete value from cache."""
        return await self.cache.delete(key, tier)
    
    async def clear_pattern(self, pattern: str) -> int:
        """Clear cache entries matching pattern."""
        return await self.cache.clear_pattern(pattern)
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return await self.cache.get_stats()
    
    def cached(self, ttl: Optional[int] = None, key_prefix: Optional[str] = None, tier: CacheTier = CacheTier.BOTH):
        """Get cached decorator for this cache manager."""
        return cached(ttl, key_prefix, tier, self.cache)
    
    def cache_invalidate(self, pattern: str):
        """Get cache invalidation decorator for this cache manager."""
        return cache_invalidate(pattern, self.cache)
    
    async def warm_cache(self, data_source: Callable[[], Awaitable[List[Tuple[str, Any]]]]):
        """Warm cache with data."""
        if self.warmer:
            await self.warmer.warm_cache(data_source, self.config.warming_batch_size)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_cache_config(**kwargs) -> CacheConfig:
    """Create cache configuration with defaults."""
    return CacheConfig(**kwargs)

def create_cache_manager(config: Optional[CacheConfig] = None) -> CacheManager:
    """Create cache manager with configuration."""
    if config is None:
        config = CacheConfig()
    
    return CacheManager(config)

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

async def example_usage():
    """Example of how to use the caching system."""
    
    # Create cache configuration
    config = CacheConfig(
        redis_url="redis://localhost:6379",
        memory_cache_size=1000,
        memory_cache_ttl=300,
        enable_multi_tier=True,
        enable_monitoring=True
    )
    
    # Create cache manager
    cache_manager = create_cache_manager(config)
    await cache_manager.start()
    
    # Use cached decorator
    @cache_manager.cached(ttl=3600, key_prefix="user")
    async def get_user(user_id: int):
        
    """get_user function."""
# Simulate database call
        await asyncio.sleep(0.1)
        return {"id": user_id, "name": f"User {user_id}"}
    
    # Use cache invalidation decorator
    @cache_manager.cache_invalidate("user:*")
    async def update_user(user_id: int, data: dict):
        
    """update_user function."""
# Simulate database update
        await asyncio.sleep(0.1)
        return {"id": user_id, **data}
    
    # Example usage
    user1 = await get_user(1)  # Will be cached
    user1_again = await get_user(1)  # Will be served from cache
    
    await update_user(1, {"name": "Updated User"})  # Will invalidate user cache
    
    # Get cache statistics
    stats = await cache_manager.get_stats()
    print(f"Cache stats: {stats}")
    
    await cache_manager.stop()

match __name__:
    case "__main__":
    asyncio.run(example_usage()) 