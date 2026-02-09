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
import logging
import hashlib
import pickle
import gzip
import json
from typing import Any, Optional, Dict, List, Callable, Awaitable, Union, Tuple, Set
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from collections import defaultdict, deque, OrderedDict
from enum import Enum
import weakref
import os
import pathlib
import aioredis
import orjson
from cachetools import TTLCache, LRUCache, LFUCache
from pydantic import BaseModel, Field
import structlog
from typing import Any, List, Dict, Optional
"""
💾 Advanced Caching System
==========================

Comprehensive caching system for static and frequently accessed data:
- Multi-level caching (L1: Memory, L2: Redis, L3: Disk)
- Intelligent cache invalidation
- Cache warming and preloading
- Cache compression and serialization
- Cache statistics and monitoring
- Cache patterns and strategies
- Distributed caching support
"""



logger = structlog.get_logger(__name__)

class CacheLevel(Enum):
    """Cache levels"""
    L1_MEMORY = "l1_memory"      # Fastest, limited size
    L2_REDIS = "l2_redis"        # Distributed, persistent
    L3_DISK = "l3_disk"          # Slowest, unlimited size

class CacheStrategy(Enum):
    """Cache strategies"""
    LRU = "lru"                  # Least Recently Used
    LFU = "lfu"                  # Least Frequently Used
    TTL = "ttl"                  # Time To Live
    HYBRID = "hybrid"            # Combination of strategies

class CacheType(Enum):
    """Cache types"""
    STATIC = "static"            # Static data (configs, templates)
    DYNAMIC = "dynamic"          # Dynamic data (user data, sessions)
    COMPUTED = "computed"        # Computed data (calculations, aggregations)
    TEMPORARY = "temporary"      # Temporary data (locks, counters)

@dataclass
class CacheConfig:
    """Cache configuration"""
    # Memory cache settings
    memory_max_size: int = 10000
    memory_ttl: int = 3600
    memory_strategy: CacheStrategy = CacheStrategy.LRU
    
    # Redis cache settings
    redis_url: str = "redis://localhost:6379"
    redis_max_connections: int = 20
    redis_ttl: int = 86400  # 24 hours
    redis_compression: bool = True
    
    # Disk cache settings
    disk_cache_dir: str = "./cache"
    disk_max_size_mb: int = 1024  # 1GB
    disk_compression: bool = True
    
    # General settings
    enable_compression: bool = True
    compression_threshold: int = 1024  # Compress if > 1KB
    enable_metrics: bool = True
    cache_warming: bool = True
    cache_preloading: bool = True

@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    key: str
    value: Any
    level: CacheLevel
    cache_type: CacheType
    created_at: float
    accessed_at: float
    access_count: int = 0
    size_bytes: int = 0
    compressed: bool = False
    ttl: Optional[int] = None

@dataclass
class CacheMetrics:
    """Cache performance metrics"""
    hits: int = 0
    misses: int = 0
    sets: int = 0
    deletes: int = 0
    evictions: int = 0
    total_size_bytes: int = 0
    compression_ratio: float = 1.0
    average_access_time: float = 0.0

class MemoryCache:
    """Advanced in-memory cache with multiple strategies."""
    
    def __init__(self, config: CacheConfig):
        
    """__init__ function."""
self.config = config
        self.metrics = CacheMetrics()
        
        # Initialize cache based on strategy
        if config.memory_strategy == CacheStrategy.LRU:
            self.cache = LRUCache(maxsize=config.memory_max_size)
        elif config.memory_strategy == CacheStrategy.LFU:
            self.cache = LFUCache(maxsize=config.memory_max_size)
        elif config.memory_strategy == CacheStrategy.TTL:
            self.cache = TTLCache(maxsize=config.memory_max_size, ttl=config.memory_ttl)
        else:  # HYBRID
            self.cache = LRUCache(maxsize=config.memory_max_size)
        
        self._access_times = {}
        self._access_counts = defaultdict(int)
        self._lock = asyncio.Lock()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from memory cache."""
        start_time = time.time()
        
        async with self._lock:
            if key in self.cache:
                value = self.cache[key]
                self._access_times[key] = time.time()
                self._access_counts[key] += 1
                
                access_time = time.time() - start_time
                self.metrics.hits += 1
                self.metrics.average_access_time = (
                    (self.metrics.average_access_time * (self.metrics.hits - 1) + access_time) / self.metrics.hits
                )
                
                return value
            
            self.metrics.misses += 1
            return None
    
    async def set(self, key: str, value: Any, ttl: int = None) -> None:
        """Set value in memory cache."""
        async with self._lock:
            # Calculate size
            size = len(pickle.dumps(value))
            
            # Check if we need to evict
            if len(self.cache) >= self.config.memory_max_size:
                self._evict_entries()
            
            self.cache[key] = value
            self._access_times[key] = time.time()
            self._access_counts[key] = 0
            
            self.metrics.sets += 1
            self.metrics.total_size_bytes += size
    
    async def delete(self, key: str) -> bool:
        """Delete value from memory cache."""
        async with self._lock:
            if key in self.cache:
                del self.cache[key]
                self._access_times.pop(key, None)
                self._access_counts.pop(key, None)
                self.metrics.deletes += 1
                return True
            return False
    
    async def clear(self) -> None:
        """Clear all cache entries."""
        async with self._lock:
            self.cache.clear()
            self._access_times.clear()
            self._access_counts.clear()
            self.metrics = CacheMetrics()
    
    def _evict_entries(self) -> None:
        """Evict entries based on strategy."""
        if self.config.memory_strategy == CacheStrategy.HYBRID:
            # Hybrid strategy: combine LRU and LFU
            current_time = time.time()
            scores = {}
            
            for key in self.cache.keys():
                access_time = self._access_times.get(key, 0)
                access_count = self._access_counts.get(key, 0)
                
                # Score based on recency and frequency
                time_score = current_time - access_time
                freq_score = 1.0 / (access_count + 1)
                scores[key] = time_score * 0.7 + freq_score * 0.3
            
            # Remove lowest scoring entry
            if scores:
                worst_key = min(scores.keys(), key=lambda k: scores[k])
                del self.cache[worst_key]
                self._access_times.pop(worst_key, None)
                self._access_counts.pop(worst_key, None)
                self.metrics.evictions += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory cache statistics."""
        return {
            "size": len(self.cache),
            "max_size": self.config.memory_max_size,
            "strategy": self.config.memory_strategy.value,
            "metrics": self.metrics.__dict__,
            "hit_rate": self.metrics.hits / (self.metrics.hits + self.metrics.misses) if (self.metrics.hits + self.metrics.misses) > 0 else 0
        }

class RedisCache:
    """Redis-based distributed cache."""
    
    def __init__(self, config: CacheConfig):
        
    """__init__ function."""
self.config = config
        self.redis_client = None
        self.metrics = CacheMetrics()
        self._lock = asyncio.Lock()
    
    async def initialize(self) -> Any:
        """Initialize Redis connection."""
        self.redis_client = aioredis.from_url(
            self.config.redis_url,
            max_connections=self.config.redis_max_connections,
            decode_responses=False
        )
        
        # Test connection
        await self.redis_client.ping()
        logger.info("Redis cache initialized")
    
    async def shutdown(self) -> Any:
        """Shutdown Redis connection."""
        if self.redis_client:
            await self.redis_client.close()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from Redis cache."""
        start_time = time.time()
        
        try:
            value = await self.redis_client.get(key)
            if value:
                # Decompress if needed
                if self.config.redis_compression and value.startswith(b'gzip:'):
                    compressed_data = value[5:]  # Remove 'gzip:' prefix
                    value = gzip.decompress(compressed_data)
                
                # Deserialize
                deserialized = pickle.loads(value)
                
                access_time = time.time() - start_time
                self.metrics.hits += 1
                self.metrics.average_access_time = (
                    (self.metrics.average_access_time * (self.metrics.hits - 1) + access_time) / self.metrics.hits
                )
                
                return deserialized
            
            self.metrics.misses += 1
            return None
            
        except Exception as e:
            logger.error(f"Redis get error: {e}")
            self.metrics.misses += 1
            return None
    
    async def set(self, key: str, value: Any, ttl: int = None) -> None:
        """Set value in Redis cache."""
        try:
            # Serialize
            serialized = pickle.dumps(value)
            
            # Compress if needed
            if self.config.redis_compression and len(serialized) > self.config.compression_threshold:
                compressed = gzip.compress(serialized)
                if len(compressed) < len(serialized):
                    serialized = b'gzip:' + compressed
            
            # Set with TTL
            ttl = ttl or self.config.redis_ttl
            await self.redis_client.setex(key, ttl, serialized)
            
            self.metrics.sets += 1
            self.metrics.total_size_bytes += len(serialized)
            
        except Exception as e:
            logger.error(f"Redis set error: {e}")
    
    async def delete(self, key: str) -> bool:
        """Delete value from Redis cache."""
        try:
            result = await self.redis_client.delete(key)
            if result:
                self.metrics.deletes += 1
            return bool(result)
        except Exception as e:
            logger.error(f"Redis delete error: {e}")
            return False
    
    async def clear(self) -> None:
        """Clear all cache entries."""
        try:
            await self.redis_client.flushdb()
            self.metrics = CacheMetrics()
        except Exception as e:
            logger.error(f"Redis clear error: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get Redis cache statistics."""
        return {
            "url": self.config.redis_url,
            "compression": self.config.redis_compression,
            "metrics": self.metrics.__dict__,
            "hit_rate": self.metrics.hits / (self.metrics.hits + self.metrics.misses) if (self.metrics.hits + self.metrics.misses) > 0 else 0
        }

class DiskCache:
    """Disk-based persistent cache."""
    
    def __init__(self, config: CacheConfig):
        
    """__init__ function."""
self.config = config
        self.cache_dir = pathlib.Path(config.disk_cache_dir)
        self.metrics = CacheMetrics()
        self._lock = asyncio.Lock()
        self._ensure_cache_dir()
    
    def _ensure_cache_dir(self) -> Any:
        """Ensure cache directory exists."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_cache_path(self, key: str) -> pathlib.Path:
        """Get cache file path for key."""
        # Create hash-based filename
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.cache"
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from disk cache."""
        start_time = time.time()
        cache_path = self._get_cache_path(key)
        
        try:
            if cache_path.exists():
                with open(cache_path, 'rb') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                    data = f.read()
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                
                # Check if compressed
                if data.startswith(b'gzip:'):
                    compressed_data = data[5:]
                    data = gzip.decompress(compressed_data)
                
                # Deserialize
                value = pickle.loads(data)
                
                access_time = time.time() - start_time
                self.metrics.hits += 1
                self.metrics.average_access_time = (
                    (self.metrics.average_access_time * (self.metrics.hits - 1) + access_time) / self.metrics.hits
                )
                
                return value
            
            self.metrics.misses += 1
            return None
            
        except Exception as e:
            logger.error(f"Disk cache get error: {e}")
            self.metrics.misses += 1
            return None
    
    async def set(self, key: str, value: Any, ttl: int = None) -> None:
        """Set value in disk cache."""
        try:
            # Serialize
            serialized = pickle.dumps(value)
            
            # Compress if needed
            if self.config.disk_compression and len(serialized) > self.config.compression_threshold:
                compressed = gzip.compress(serialized)
                if len(compressed) < len(serialized):
                    serialized = b'gzip:' + compressed
            
            # Write to disk
            cache_path = self._get_cache_path(key)
            with open(cache_path, 'wb') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                f.write(serialized)
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            
            self.metrics.sets += 1
            self.metrics.total_size_bytes += len(serialized)
            
        except Exception as e:
            logger.error(f"Disk cache set error: {e}")
    
    async def delete(self, key: str) -> bool:
        """Delete value from disk cache."""
        try:
            cache_path = self._get_cache_path(key)
            if cache_path.exists():
                cache_path.unlink()
                self.metrics.deletes += 1
                return True
            return False
        except Exception as e:
            logger.error(f"Disk cache delete error: {e}")
            return False
    
    async def clear(self) -> None:
        """Clear all cache entries."""
        try:
            for cache_file in self.cache_dir.glob("*.cache"):
                cache_file.unlink()
            self.metrics = CacheMetrics()
        except Exception as e:
            logger.error(f"Disk cache clear error: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get disk cache statistics."""
        cache_files = list(self.cache_dir.glob("*.cache"))
        total_size = sum(f.stat().st_size for f in cache_files)
        
        return {
            "cache_dir": str(self.cache_dir),
            "file_count": len(cache_files),
            "total_size_bytes": total_size,
            "max_size_mb": self.config.disk_max_size_mb,
            "compression": self.config.disk_compression,
            "metrics": self.metrics.__dict__,
            "hit_rate": self.metrics.hits / (self.metrics.hits + self.metrics.misses) if (self.metrics.hits + self.metrics.misses) > 0 else 0
        }

class AdvancedCachingSystem:
    """
    Multi-level caching system with intelligent cache management.
    """
    
    def __init__(self, config: CacheConfig):
        
    """__init__ function."""
self.config = config
        self.memory_cache = MemoryCache(config)
        self.redis_cache = RedisCache(config)
        self.disk_cache = DiskCache(config)
        
        # Cache warming and preloading
        self.warmup_queue = deque()
        self.preload_patterns = {}
        
        # Statistics
        self.total_requests = 0
        self.total_hits = 0
        self.cache_patterns = defaultdict(int)
        
    async def initialize(self) -> Any:
        """Initialize all cache levels."""
        await self.redis_cache.initialize()
        
        if self.config.cache_warming:
            await self._warmup_cache()
        
        logger.info("Advanced caching system initialized")
    
    async def shutdown(self) -> Any:
        """Shutdown all cache levels."""
        await self.redis_cache.shutdown()
        logger.info("Advanced caching system shutdown complete")
    
    async def get(self, key: str, cache_type: CacheType = CacheType.DYNAMIC) -> Optional[Any]:
        """Get value from multi-level cache."""
        self.total_requests += 1
        self.cache_patterns[cache_type.value] += 1
        
        # Try L1 (Memory) first
        value = await self.memory_cache.get(key)
        if value is not None:
            self.total_hits += 1
            return value
        
        # Try L2 (Redis)
        value = await self.redis_cache.get(key)
        if value is not None:
            # Store in L1 for future access
            await self.memory_cache.set(key, value)
            self.total_hits += 1
            return value
        
        # Try L3 (Disk) for static data
        if cache_type == CacheType.STATIC:
            value = await self.disk_cache.get(key)
            if value is not None:
                # Store in L1 and L2 for future access
                await self.memory_cache.set(key, value)
                await self.redis_cache.set(key, value)
                self.total_hits += 1
                return value
        
        return None
    
    async def set(self, key: str, value: Any, cache_type: CacheType = CacheType.DYNAMIC, 
                  ttl: int = None, levels: List[CacheLevel] = None) -> None:
        """Set value in specified cache levels."""
        if levels is None:
            # Default levels based on cache type
            if cache_type == CacheType.STATIC:
                levels = [CacheLevel.L1_MEMORY, CacheLevel.L2_REDIS, CacheLevel.L3_DISK]
            elif cache_type == CacheType.DYNAMIC:
                levels = [CacheLevel.L1_MEMORY, CacheLevel.L2_REDIS]
            else:
                levels = [CacheLevel.L1_MEMORY]
        
        # Set in specified levels
        for level in levels:
            if level == CacheLevel.L1_MEMORY:
                await self.memory_cache.set(key, value, ttl)
            elif level == CacheLevel.L2_REDIS:
                await self.redis_cache.set(key, value, ttl)
            elif level == CacheLevel.L3_DISK:
                await self.disk_cache.set(key, value, ttl)
    
    async def delete(self, key: str, levels: List[CacheLevel] = None) -> bool:
        """Delete value from specified cache levels."""
        if levels is None:
            levels = [CacheLevel.L1_MEMORY, CacheLevel.L2_REDIS, CacheLevel.L3_DISK]
        
        deleted = False
        for level in levels:
            if level == CacheLevel.L1_MEMORY:
                if await self.memory_cache.delete(key):
                    deleted = True
            elif level == CacheLevel.L2_REDIS:
                if await self.redis_cache.delete(key):
                    deleted = True
            elif level == CacheLevel.L3_DISK:
                if await self.disk_cache.delete(key):
                    deleted = True
        
        return deleted
    
    async def clear(self, levels: List[CacheLevel] = None) -> None:
        """Clear specified cache levels."""
        if levels is None:
            levels = [CacheLevel.L1_MEMORY, CacheLevel.L2_REDIS, CacheLevel.L3_DISK]
        
        for level in levels:
            if level == CacheLevel.L1_MEMORY:
                await self.memory_cache.clear()
            elif level == CacheLevel.L2_REDIS:
                await self.redis_cache.clear()
            elif level == CacheLevel.L3_DISK:
                await self.disk_cache.clear()
    
    async def warmup_cache(self, data: Dict[str, Any], cache_type: CacheType = CacheType.STATIC):
        """Warm up cache with predefined data."""
        for key, value in data.items():
            await self.set(key, value, cache_type)
        
        logger.info(f"Cache warmed up with {len(data)} entries")
    
    async def preload_pattern(self, pattern: str, loader_func: Callable, 
                             cache_type: CacheType = CacheType.STATIC):
        """Preload cache based on pattern."""
        self.preload_patterns[pattern] = {
            "loader": loader_func,
            "cache_type": cache_type
        }
    
    async def _warmup_cache(self) -> Any:
        """Internal cache warming."""
        # Warm up with common static data
        static_data = {
            "config:app": {"version": "1.0.0", "environment": "production"},
            "config:features": {"feature_a": True, "feature_b": False},
            "templates:email": {"subject": "Welcome", "body": "Hello {{name}}"},
            "templates:sms": {"message": "Your code is {{code}}"}
        }
        
        await self.warmup_cache(static_data, CacheType.STATIC)
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        memory_stats = self.memory_cache.get_stats()
        redis_stats = self.redis_cache.get_stats()
        disk_stats = self.disk_cache.get_stats()
        
        overall_hit_rate = self.total_hits / self.total_requests if self.total_requests > 0 else 0
        
        return {
            "overall": {
                "total_requests": self.total_requests,
                "total_hits": self.total_hits,
                "hit_rate": overall_hit_rate,
                "cache_patterns": dict(self.cache_patterns)
            },
            "memory": memory_stats,
            "redis": redis_stats,
            "disk": disk_stats,
            "config": {
                "memory_max_size": self.config.memory_max_size,
                "redis_url": self.config.redis_url,
                "disk_cache_dir": self.config.disk_cache_dir,
                "compression_enabled": self.config.enable_compression
            }
        }

# Cache decorators
def cache_result(ttl: int = 3600, cache_type: CacheType = CacheType.DYNAMIC, 
                levels: List[CacheLevel] = None, key_generator: Callable = None):
    """Decorator for caching function results."""
    def decorator(func: Callable) -> Callable:
        cache_system = None
        
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            nonlocal cache_system
            
            # Initialize cache system if needed
            if cache_system is None:
                config = CacheConfig()
                cache_system = AdvancedCachingSystem(config)
                await cache_system.initialize()
            
            # Generate cache key
            if key_generator:
                cache_key = key_generator(*args, **kwargs)
            else:
                key_data = {
                    'func': func.__name__,
                    'args': args,
                    'kwargs': sorted(kwargs.items())
                }
                cache_key = hashlib.md5(orjson.dumps(key_data)).hexdigest()
            
            # Try to get from cache
            cached_result = await cache_system.get(cache_key, cache_type)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = await func(*args, **kwargs)
            await cache_system.set(cache_key, result, cache_type, ttl, levels)
            
            return result
        
        return wrapper
    return decorator

def static_cache(ttl: int = 86400):
    """Decorator for static data caching."""
    return cache_result(ttl=ttl, cache_type=CacheType.STATIC, 
                       levels=[CacheLevel.L1_MEMORY, CacheLevel.L2_REDIS, CacheLevel.L3_DISK])

def dynamic_cache(ttl: int = 3600):
    """Decorator for dynamic data caching."""
    return cache_result(ttl=ttl, cache_type=CacheType.DYNAMIC, 
                       levels=[CacheLevel.L1_MEMORY, CacheLevel.L2_REDIS])

# Example usage
async def example_caching_usage():
    """Example usage of advanced caching system."""
    
    # Create configuration
    config = CacheConfig(
        memory_max_size=10000,
        redis_url="redis://localhost:6379",
        disk_cache_dir="./cache",
        enable_compression=True,
        cache_warming=True
    )
    
    # Initialize caching system
    cache_system = AdvancedCachingSystem(config)
    await cache_system.initialize()
    
    try:
        # Cache static data
        await cache_system.set("config:app", {
            "version": "1.0.0",
            "environment": "production"
        }, CacheType.STATIC)
        
        # Cache dynamic data
        await cache_system.set("user:123", {
            "id": 123,
            "name": "John Doe",
            "email": "john@example.com"
        }, CacheType.DYNAMIC, ttl=1800)
        
        # Retrieve data
        app_config = await cache_system.get("config:app", CacheType.STATIC)
        user_data = await cache_system.get("user:123", CacheType.DYNAMIC)
        
        logger.info(f"App config: {app_config}")
        logger.info(f"User data: {user_data}")
        
        # Use decorators
        @static_cache(ttl=86400)
        async def get_app_config():
            
    """get_app_config function."""
# Expensive operation to get app config
            return {"version": "1.0.0", "features": ["a", "b", "c"]}
        
        @dynamic_cache(ttl=3600)
        async def get_user_profile(user_id: int):
            
    """get_user_profile function."""
# Expensive operation to get user profile
            return {"id": user_id, "name": f"User {user_id}"}
        
        # Get cached results
        config = await get_app_config()
        profile = await get_user_profile(123)
        
        # Get comprehensive statistics
        stats = cache_system.get_comprehensive_stats()
        logger.info(f"Cache statistics: {stats}")
        
    except Exception as e:
        logger.error(f"Caching error: {e}")
    
    finally:
        await cache_system.shutdown()

match __name__:
    case "__main__":
    asyncio.run(example_caching_usage()) 