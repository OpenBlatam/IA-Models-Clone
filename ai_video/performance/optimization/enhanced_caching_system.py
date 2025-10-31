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
import json
import hashlib
import pickle
import gzip
from typing import Any, Dict, List, Optional, Union, Callable, TypeVar, Generic, Set
from functools import lru_cache, wraps
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
import threading
from collections import defaultdict, deque, OrderedDict
import statistics
import weakref
from datetime import datetime, timedelta
import aioredis
import redis.asyncio as redis
from sqlalchemy.ext.asyncio import AsyncSession
import numpy as np
import torch
from typing import Any, List, Dict, Optional
"""
🚀 ENHANCED CACHING SYSTEM - STATIC & FREQUENTLY ACCESSED DATA
=============================================================

Advanced caching system for static and frequently accessed data with:
- Multi-tier caching (Memory + Redis)
- Predictive caching and cache warming
- Intelligent eviction policies
- Cache statistics and monitoring
- Static data management
- Frequently accessed data optimization
- Cache invalidation strategies
"""



logger = logging.getLogger(__name__)

# Type variables
T = TypeVar('T')
K = TypeVar('K')

# ============================================================================
# 1. CACHE CONFIGURATION AND TYPES
# ============================================================================

class CacheType(Enum):
    """Types of cache data."""
    STATIC = "static"           # Never changes (configs, models)
    FREQUENT = "frequent"       # Frequently accessed (user data, sessions)
    DYNAMIC = "dynamic"         # Changes occasionally (video metadata)
    TEMPORARY = "temporary"     # Short-lived (processing results)

class EvictionPolicy(Enum):
    """Cache eviction policies."""
    LRU = "lru"                 # Least Recently Used
    LFU = "lfu"                 # Least Frequently Used
    TTL = "ttl"                 # Time To Live
    ADAPTIVE = "adaptive"       # Adaptive based on access patterns

@dataclass
class CacheConfig:
    """Configuration for cache tiers."""
    name: str
    cache_type: CacheType
    ttl: int = 3600  # Time to live in seconds
    max_size: int = 1000  # Maximum number of items
    eviction_policy: EvictionPolicy = EvictionPolicy.LRU
    enable_compression: bool = True
    enable_stats: bool = True
    compression_threshold: int = 1024  # Compress if larger than this
    warm_cache: bool = False  # Pre-warm cache on startup

@dataclass
class CacheStats:
    """Cache performance statistics."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    size: int = 0
    total_size_bytes: int = 0
    compression_ratio: float = 1.0
    last_access: float = field(default_factory=time.time)
    access_count: int = 0
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    @property
    def avg_access_time(self) -> float:
        """Calculate average time between accesses."""
        if self.access_count <= 1:
            return 0.0
        return (time.time() - self.last_access) / self.access_count
    
    def reset(self) -> Any:
        """Reset statistics."""
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.access_count = 0

# ============================================================================
# 2. MEMORY CACHE IMPLEMENTATION
# ============================================================================

class MemoryCache:
    """High-performance in-memory cache with multiple eviction policies."""
    
    def __init__(self, config: CacheConfig):
        
    """__init__ function."""
self.config = config
        self.stats = CacheStats() if config.enable_stats else None
        self._lock = asyncio.Lock()
        
        # Initialize storage based on eviction policy
        if config.eviction_policy == EvictionPolicy.LRU:
            self._storage = OrderedDict()
        elif config.eviction_policy == EvictionPolicy.LFU:
            self._storage = {}
            self._access_counts = defaultdict(int)
        else:
            self._storage = {}
        
        # TTL tracking
        self._ttl_times = {}
        self._last_cleanup = time.time()
        self._cleanup_interval = 300  # 5 minutes
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from memory cache."""
        async with self._lock:
            # Check TTL first
            if key in self._ttl_times:
                if time.time() > self._ttl_times[key]:
                    await self._remove(key)
                    return None
            
            if key in self._storage:
                value = self._storage[key]
                
                # Update access patterns
                if self.config.eviction_policy == EvictionPolicy.LRU:
                    self._storage.move_to_end(key)
                elif self.config.eviction_policy == EvictionPolicy.LFU:
                    self._access_counts[key] += 1
                
                # Update statistics
                if self.stats:
                    self.stats.hits += 1
                    self.stats.last_access = time.time()
                    self.stats.access_count += 1
                
                return value
            
            if self.stats:
                self.stats.misses += 1
            
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in memory cache."""
        async with self._lock:
            # Check if we need to evict
            if len(self._storage) >= self.config.max_size:
                await self._evict()
            
            # Compress if needed
            if self.config.enable_compression and self._should_compress(value):
                value = await self._compress(value)
            
            # Store value
            self._storage[key] = value
            
            # Set TTL
            ttl = ttl or self.config.ttl
            if ttl > 0:
                self._ttl_times[key] = time.time() + ttl
            
            # Update statistics
            if self.stats:
                self.stats.size = len(self._storage)
                self.stats.total_size_bytes += self._get_size(value)
            
            return True
    
    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
        async with self._lock:
            return await self._remove(key)
    
    async def clear(self) -> Any:
        """Clear all cache entries."""
        async with self._lock:
            self._storage.clear()
            self._ttl_times.clear()
            if self.config.eviction_policy == EvictionPolicy.LFU:
                self._access_counts.clear()
            if self.stats:
                self.stats.size = 0
                self.stats.total_size_bytes = 0
    
    async def _evict(self) -> Any:
        """Evict items based on policy."""
        if not self._storage:
            return
        
        if self.config.eviction_policy == EvictionPolicy.LRU:
            # Remove oldest item
            oldest_key = next(iter(self._storage))
            await self._remove(oldest_key)
        
        elif self.config.eviction_policy == EvictionPolicy.LFU:
            # Remove least frequently used
            least_used_key = min(self._access_counts.keys(), 
                                key=lambda k: self._access_counts[k])
            await self._remove(least_used_key)
        
        elif self.config.eviction_policy == EvictionPolicy.TTL:
            # Remove expired items
            current_time = time.time()
            expired_keys = [
                key for key, expiry in self._ttl_times.items()
                if current_time > expiry
            ]
            for key in expired_keys:
                await self._remove(key)
    
    async def _remove(self, key: str) -> bool:
        """Remove key from cache."""
        if key in self._storage:
            value = self._storage.pop(key)
            self._ttl_times.pop(key, None)
            
            if self.config.eviction_policy == EvictionPolicy.LFU:
                self._access_counts.pop(key, None)
            
            if self.stats:
                self.stats.evictions += 1
                self.stats.size = len(self._storage)
                self.stats.total_size_bytes -= self._get_size(value)
            
            return True
        return False
    
    def _should_compress(self, value: Any) -> bool:
        """Check if value should be compressed."""
        size = self._get_size(value)
        return size > self.config.compression_threshold
    
    async def _compress(self, value: Any) -> bytes:
        """Compress value."""
        try:
            serialized = pickle.dumps(value)
            compressed = gzip.compress(serialized)
            return compressed
        except Exception as e:
            logger.error(f"Compression failed: {e}")
            return value
    
    async def _decompress(self, value: bytes) -> Any:
        """Decompress value."""
        try:
            if isinstance(value, bytes) and value.startswith(b'\x1f\x8b'):
                decompressed = gzip.decompress(value)
                return pickle.loads(decompressed)
            return value
        except Exception as e:
            logger.error(f"Decompression failed: {e}")
            return value
    
    def _get_size(self, value: Any) -> int:
        """Get approximate size of value in bytes."""
        try:
            return len(pickle.dumps(value))
        except:
            return 0
    
    async def cleanup_expired(self) -> Any:
        """Clean up expired entries."""
        current_time = time.time()
        if current_time - self._last_cleanup > self._cleanup_interval:
            async with self._lock:
                expired_keys = [
                    key for key, expiry in self._ttl_times.items()
                    if current_time > expiry
                ]
                for key in expired_keys:
                    await self._remove(key)
                self._last_cleanup = current_time
    
    def get_stats(self) -> Optional[Dict[str, Any]]:
        """Get cache statistics."""
        if not self.stats:
            return None
        
        return {
            "hits": self.stats.hits,
            "misses": self.stats.misses,
            "hit_rate": self.stats.hit_rate,
            "size": self.stats.size,
            "max_size": self.config.max_size,
            "total_size_bytes": self.stats.total_size_bytes,
            "evictions": self.stats.evictions,
            "avg_access_time": self.stats.avg_access_time
        }

# ============================================================================
# 3. REDIS CACHE IMPLEMENTATION
# ============================================================================

class RedisCache:
    """Redis-based cache with advanced features."""
    
    def __init__(self, redis_client: redis.Redis, config: CacheConfig):
        
    """__init__ function."""
self.redis_client = redis_client
        self.config = config
        self.stats = CacheStats() if config.enable_stats else None
        self._lock = asyncio.Lock()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from Redis cache."""
        try:
            value = await self.redis_client.get(key)
            if value:
                if self.stats:
                    self.stats.hits += 1
                    self.stats.last_access = time.time()
                    self.stats.access_count += 1
                
                # Decompress if needed
                if self.config.enable_compression:
                    return await self._decompress(value)
                return json.loads(value)
            
            if self.stats:
                self.stats.misses += 1
            
            return None
            
        except Exception as e:
            logger.error(f"Redis get failed for key {key}: {e}")
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in Redis cache."""
        try:
            ttl = ttl or self.config.ttl
            
            # Compress if needed
            if self.config.enable_compression and self._should_compress(value):
                serialized_value = await self._compress(value)
            else:
                serialized_value = json.dumps(value)
            
            if ttl > 0:
                await self.redis_client.setex(key, ttl, serialized_value)
            else:
                await self.redis_client.set(key, serialized_value)
            
            return True
            
        except Exception as e:
            logger.error(f"Redis set failed for key {key}: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete key from Redis cache."""
        try:
            result = await self.redis_client.delete(key)
            return result > 0
        except Exception as e:
            logger.error(f"Redis delete failed for key {key}: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in Redis."""
        try:
            return await self.redis_client.exists(key) > 0
        except Exception as e:
            logger.error(f"Redis exists failed for key {key}: {e}")
            return False
    
    async def get_ttl(self, key: str) -> int:
        """Get TTL for key."""
        try:
            return await self.redis_client.ttl(key)
        except Exception as e:
            logger.error(f"Redis TTL failed for key {key}: {e}")
            return -1
    
    def _should_compress(self, value: Any) -> bool:
        """Check if value should be compressed."""
        size = len(json.dumps(value))
        return size > self.config.compression_threshold
    
    async def _compress(self, value: Any) -> bytes:
        """Compress value."""
        try:
            serialized = json.dumps(value).encode('utf-8')
            compressed = gzip.compress(serialized)
            return compressed
        except Exception as e:
            logger.error(f"Compression failed: {e}")
            return json.dumps(value).encode('utf-8')
    
    async def _decompress(self, value: bytes) -> Any:
        """Decompress value."""
        try:
            if value.startswith(b'\x1f\x8b'):
                decompressed = gzip.decompress(value)
                return json.loads(decompressed.decode('utf-8'))
            return json.loads(value.decode('utf-8'))
        except Exception as e:
            logger.error(f"Decompression failed: {e}")
            return json.loads(value.decode('utf-8'))

# ============================================================================
# 4. PREDICTIVE CACHING SYSTEM
# ============================================================================

class PredictiveCache:
    """Predictive caching based on access patterns."""
    
    def __init__(self, max_patterns: int = 1000):
        
    """__init__ function."""
self.access_patterns = defaultdict(list)
        self.prediction_scores = defaultdict(float)
        self.max_patterns = max_patterns
        self._lock = asyncio.Lock()
    
    def record_access(self, key: str, context: Optional[str] = None):
        """Record access pattern."""
        async with self._lock:
            pattern_key = f"{context}:{key}" if context else key
            self.access_patterns[pattern_key].append(time.time())
            
            # Keep only recent accesses
            if len(self.access_patterns[pattern_key]) > 100:
                self.access_patterns[pattern_key] = self.access_patterns[pattern_key][-100:]
    
    def predict_next_access(self, key: str, context: Optional[str] = None) -> float:
        """Predict likelihood of next access."""
        pattern_key = f"{context}:{key}" if context else key
        
        if pattern_key not in self.access_patterns:
            return 0.0
        
        accesses = self.access_patterns[pattern_key]
        if len(accesses) < 2:
            return 0.5
        
        # Calculate time intervals between accesses
        intervals = [accesses[i] - accesses[i-1] for i in range(1, len(accesses))]
        
        if not intervals:
            return 0.0
        
        # Calculate average interval
        avg_interval = statistics.mean(intervals)
        
        # Calculate time since last access
        time_since_last = time.time() - accesses[-1]
        
        # Predict likelihood based on time since last access vs average interval
        if avg_interval > 0:
            likelihood = max(0.0, 1.0 - (time_since_last / avg_interval))
            return min(1.0, likelihood)
        
        return 0.0
    
    def get_related_keys(self, key: str, context: Optional[str] = None) -> List[str]:
        """Get keys that are frequently accessed together."""
        pattern_key = f"{context}:{key}" if context else key
        
        # Find keys that are accessed within a short time window
        related_keys = []
        if pattern_key in self.access_patterns:
            key_accesses = self.access_patterns[pattern_key]
            
            for other_key, other_accesses in self.access_patterns.items():
                if other_key != pattern_key:
                    # Check if accesses are correlated
                    correlation = self._calculate_correlation(key_accesses, other_accesses)
                    if correlation > 0.7:  # High correlation threshold
                        related_keys.append(other_key)
        
        return related_keys
    
    def _calculate_correlation(self, accesses1: List[float], accesses2: List[float]) -> float:
        """Calculate correlation between two access patterns."""
        if len(accesses1) < 2 or len(accesses2) < 2:
            return 0.0
        
        # Simple correlation based on timing
        try:
            return statistics.correlation(accesses1, accesses2)
        except:
            return 0.0

# ============================================================================
# 5. STATIC DATA MANAGER
# ============================================================================

class StaticDataManager:
    """Manages static data that rarely or never changes."""
    
    def __init__(self, memory_cache: MemoryCache, redis_cache: RedisCache):
        
    """__init__ function."""
self.memory_cache = memory_cache
        self.redis_cache = redis_cache
        self.static_keys: Set[str] = set()
        self._lock = asyncio.Lock()
    
    async def register_static_data(self, key: str, data: Any, ttl: Optional[int] = None):
        """Register static data that should be cached permanently."""
        async with self._lock:
            self.static_keys.add(key)
            
            # Store in both caches
            await self.memory_cache.set(key, data, ttl)
            await self.redis_cache.set(key, data, ttl)
            
            logger.info(f"Registered static data: {key}")
    
    async def get_static_data(self, key: str) -> Optional[Any]:
        """Get static data with fallback."""
        # Try memory cache first
        value = await self.memory_cache.get(key)
        if value is not None:
            return value
        
        # Try Redis cache
        value = await self.redis_cache.get(key)
        if value is not None:
            # Store in memory cache for faster access
            await self.memory_cache.set(key, value)
            return value
        
        return None
    
    async def update_static_data(self, key: str, data: Any, ttl: Optional[int] = None):
        """Update static data in both caches."""
        async with self._lock:
            await self.memory_cache.set(key, data, ttl)
            await self.redis_cache.set(key, data, ttl)
            
            logger.info(f"Updated static data: {key}")
    
    async def invalidate_static_data(self, key: str):
        """Invalidate static data from both caches."""
        async with self._lock:
            await self.memory_cache.delete(key)
            await self.redis_cache.delete(key)
            self.static_keys.discard(key)
            
            logger.info(f"Invalidated static data: {key}")
    
    def is_static_key(self, key: str) -> bool:
        """Check if key is registered as static data."""
        return key in self.static_keys

# ============================================================================
# 6. FREQUENTLY ACCESSED DATA MANAGER
# ============================================================================

class FrequentDataManager:
    """Manages frequently accessed data with predictive caching."""
    
    def __init__(self, memory_cache: MemoryCache, redis_cache: RedisCache, 
                 predictive_cache: PredictiveCache):
        
    """__init__ function."""
self.memory_cache = memory_cache
        self.redis_cache = redis_cache
        self.predictive_cache = predictive_cache
        self.frequent_keys: Set[str] = set()
        self._lock = asyncio.Lock()
    
    async def register_frequent_data(self, key: str, data: Any, ttl: int = 1800):
        """Register frequently accessed data."""
        async with self._lock:
            self.frequent_keys.add(key)
            
            # Store in memory cache with shorter TTL
            await self.memory_cache.set(key, data, ttl // 2)
            
            # Store in Redis cache with longer TTL
            await self.redis_cache.set(key, data, ttl)
            
            logger.info(f"Registered frequent data: {key}")
    
    async def get_frequent_data(self, key: str, context: Optional[str] = None) -> Optional[Any]:
        """Get frequently accessed data with predictive caching."""
        # Record access pattern
        self.predictive_cache.record_access(key, context)
        
        # Try memory cache first
        value = await self.memory_cache.get(key)
        if value is not None:
            return value
        
        # Try Redis cache
        value = await self.redis_cache.get(key)
        if value is not None:
            # Store in memory cache for faster access
            await self.memory_cache.set(key, value)
            
            # Predict and preload related data
            await self._predict_and_preload(key, context)
            
            return value
        
        return None
    
    async def _predict_and_preload(self, key: str, context: Optional[str] = None):
        """Predict and preload related data."""
        try:
            # Get related keys
            related_keys = self.predictive_cache.get_related_keys(key, context)
            
            # Preload high-probability keys
            for related_key in related_keys[:5]:  # Limit to 5 keys
                prediction = self.predictive_cache.predict_next_access(related_key, context)
                if prediction > 0.7:  # High probability threshold
                    await self._preload_key(related_key)
                    
        except Exception as e:
            logger.error(f"Predictive preloading failed: {e}")
    
    async def _preload_key(self, key: str):
        """Preload key into memory cache."""
        try:
            # Check if key exists in Redis but not in memory
            if await self.redis_cache.exists(key):
                value = await self.redis_cache.get(key)
                if value is not None:
                    await self.memory_cache.set(key, value)
                    logger.debug(f"Preloaded key: {key}")
        except Exception as e:
            logger.error(f"Preloading failed for key {key}: {e}")
    
    def is_frequent_key(self, key: str) -> bool:
        """Check if key is registered as frequent data."""
        return key in self.frequent_keys

# ============================================================================
# 7. CACHE WARMING SYSTEM
# ============================================================================

class CacheWarmer:
    """Warms up cache with frequently accessed data."""
    
    def __init__(self, static_manager: StaticDataManager, 
                 frequent_manager: FrequentDataManager):
        
    """__init__ function."""
self.static_manager = static_manager
        self.frequent_manager = frequent_manager
        self.warming_tasks: Set[str] = set()
        self._lock = asyncio.Lock()
    
    async def warm_cache(self, data_sources: Dict[str, Callable]):
        """Warm cache with data from multiple sources."""
        async with self._lock:
            tasks = []
            for key, data_source in data_sources.items():
                if key not in self.warming_tasks:
                    task = self._warm_data_source(key, data_source)
                    tasks.append(task)
                    self.warming_tasks.add(key)
            
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
                logger.info(f"Cache warming completed for {len(tasks)} data sources")
    
    async def _warm_data_source(self, key: str, data_source: Callable):
        """Warm cache with data from a single source."""
        try:
            logger.info(f"Warming cache for: {key}")
            
            # Get data from source
            if asyncio.iscoroutinefunction(data_source):
                data = await data_source()
            else:
                data = data_source()
            
            # Register as static or frequent data based on key pattern
            if key.startswith('static_'):
                await self.static_manager.register_static_data(key, data)
            else:
                await self.frequent_manager.register_frequent_data(key, data)
            
            logger.info(f"Successfully warmed cache for: {key}")
            
        except Exception as e:
            logger.error(f"Failed to warm cache for {key}: {e}")
        finally:
            self.warming_tasks.discard(key)
    
    async def warm_static_data(self, static_data: Dict[str, Any]):
        """Warm cache with static data."""
        for key, data in static_data.items():
            await self.static_manager.register_static_data(key, data)
    
    async def warm_frequent_data(self, frequent_data: Dict[str, Any]):
        """Warm cache with frequent data."""
        for key, data in frequent_data.items():
            await self.frequent_manager.register_frequent_data(key, data)

# ============================================================================
# 8. ENHANCED CACHING SYSTEM
# ============================================================================

class EnhancedCachingSystem:
    """Complete enhanced caching system for static and frequently accessed data."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        
    """__init__ function."""
self.redis_url = redis_url
        self.redis_client = None
        
        # Initialize cache tiers
        self.static_cache = MemoryCache(CacheConfig(
            name="static_memory",
            cache_type=CacheType.STATIC,
            ttl=86400,  # 24 hours
            max_size=500,
            eviction_policy=EvictionPolicy.LRU,
            warm_cache=True
        ))
        
        self.frequent_cache = MemoryCache(CacheConfig(
            name="frequent_memory",
            cache_type=CacheType.FREQUENT,
            ttl=1800,  # 30 minutes
            max_size=1000,
            eviction_policy=EvictionPolicy.LFU,
            warm_cache=True
        ))
        
        self.redis_cache = None
        self.predictive_cache = PredictiveCache()
        
        # Initialize managers
        self.static_manager = None
        self.frequent_manager = None
        self.cache_warmer = None
        
        # Performance monitoring
        self.monitoring_task = None
        self._initialized = False
    
    async def initialize(self) -> Any:
        """Initialize the caching system."""
        try:
            # Initialize Redis client
            self.redis_client = aioredis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=False
            )
            
            # Initialize Redis cache
            self.redis_cache = RedisCache(
                self.redis_client,
                CacheConfig(
                    name="redis_cache",
                    cache_type=CacheType.DYNAMIC,
                    ttl=3600,
                    max_size=10000,
                    eviction_policy=EvictionPolicy.TTL
                )
            )
            
            # Initialize managers
            self.static_manager = StaticDataManager(
                self.static_cache, self.redis_cache
            )
            
            self.frequent_manager = FrequentDataManager(
                self.frequent_cache, self.redis_cache, self.predictive_cache
            )
            
            # Initialize cache warmer
            self.cache_warmer = CacheWarmer(
                self.static_manager, self.frequent_manager
            )
            
            self._initialized = True
            logger.info("Enhanced caching system initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize caching system: {e}")
            raise
    
    async def get_static_data(self, key: str) -> Optional[Any]:
        """Get static data."""
        if not self._initialized:
            raise RuntimeError("Caching system not initialized")
        
        return await self.static_manager.get_static_data(key)
    
    async def get_frequent_data(self, key: str, context: Optional[str] = None) -> Optional[Any]:
        """Get frequently accessed data."""
        if not self._initialized:
            raise RuntimeError("Caching system not initialized")
        
        return await self.frequent_manager.get_frequent_data(key, context)
    
    async def set_static_data(self, key: str, data: Any, ttl: Optional[int] = None):
        """Set static data."""
        if not self._initialized:
            raise RuntimeError("Caching system not initialized")
        
        await self.static_manager.register_static_data(key, data, ttl)
    
    async def set_frequent_data(self, key: str, data: Any, ttl: int = 1800):
        """Set frequently accessed data."""
        if not self._initialized:
            raise RuntimeError("Caching system not initialized")
        
        await self.frequent_manager.register_frequent_data(key, data, ttl)
    
    async def warm_cache(self, data_sources: Dict[str, Callable]):
        """Warm cache with data sources."""
        if not self._initialized:
            raise RuntimeError("Caching system not initialized")
        
        await self.cache_warmer.warm_cache(data_sources)
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        if not self._initialized:
            return {}
        
        stats = {
            "static_cache": self.static_cache.get_stats(),
            "frequent_cache": self.frequent_cache.get_stats(),
            "redis_cache": self.redis_cache.stats.__dict__ if self.redis_cache.stats else None,
            "predictive_cache": {
                "patterns": len(self.predictive_cache.access_patterns),
                "predictions": len(self.predictive_cache.prediction_scores)
            }
        }
        
        return stats
    
    async def cleanup(self) -> Any:
        """Cleanup cache system."""
        if self.monitoring_task:
            self.monitoring_task.cancel()
        
        if self.redis_client:
            await self.redis_client.close()
        
        logger.info("Enhanced caching system cleaned up")

# ============================================================================
# 9. USAGE EXAMPLES
# ============================================================================

async def example_enhanced_caching():
    """Example usage of the enhanced caching system."""
    
    # Initialize caching system
    caching_system = EnhancedCachingSystem()
    await caching_system.initialize()
    
    # Define data sources for cache warming
    data_sources = {
        "static_config": lambda: {"api_version": "2.0", "features": ["video", "audio"]},
        "static_models": lambda: {"model1": "path/to/model1", "model2": "path/to/model2"},
        "frequent_user_sessions": lambda: {"session1": "user_data_1", "session2": "user_data_2"},
        "frequent_video_metadata": lambda: {"video1": {"title": "Video 1", "duration": 120}}
    }
    
    # Warm cache
    await caching_system.warm_cache(data_sources)
    
    # Use static data
    config = await caching_system.get_static_data("static_config")
    print(f"Static config: {config}")
    
    # Use frequent data
    session_data = await caching_system.get_frequent_data("frequent_user_sessions", "user_context")
    print(f"Session data: {session_data}")
    
    # Get cache statistics
    stats = await caching_system.get_cache_stats()
    print(f"Cache stats: {stats}")
    
    # Cleanup
    await caching_system.cleanup()

match __name__:
    case "__main__":
    asyncio.run(example_enhanced_caching()) 