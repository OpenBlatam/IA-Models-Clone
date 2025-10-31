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
import hashlib
import json
import gzip
import pickle
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import logging
import threading
from functools import lru_cache, wraps
import weakref
    import orjson
    import json
    import numba
    from numba import jit, njit
from typing import Any, List, Dict, Optional
"""
Smart Cache for Instagram Captions API v14.0

Advanced caching strategies:
- Multi-level caching (L1, L2, L3)
- Predictive prefetching
- Intelligent eviction policies
- Cache warming and preloading
- Compression and optimization
- Cache analytics and monitoring
"""


# Performance libraries
try:
    json_dumps = lambda obj: orjson.dumps(obj).decode()
    json_loads = orjson.loads
    ULTRA_JSON = True
except ImportError:
    json_dumps = lambda obj: json.dumps(obj)
    json_loads = json.loads
    ULTRA_JSON = False

try:
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def jit(*args, **kwargs) -> Any:
        def decorator(func) -> Any:
            return func
        return decorator
    njit = jit

logger = logging.getLogger(__name__)


class CacheLevel(Enum):
    """Cache levels for multi-level caching"""
    L1_HOT = "l1_hot"      # Fastest, smallest
    L2_WARM = "l2_warm"    # Medium speed, medium size
    L3_COLD = "l3_cold"    # Slower, largest


class EvictionPolicy(Enum):
    """Cache eviction policies"""
    LRU = "lru"                    # Least Recently Used
    LFU = "lfu"                    # Least Frequently Used
    TTL = "ttl"                    # Time To Live
    HYBRID = "hybrid"              # Combination of policies
    ADAPTIVE = "adaptive"          # Adaptive based on usage patterns


@dataclass
class CacheConfig:
    """Configuration for smart cache"""
    # L1 Cache (Hot)
    l1_size: int = 1000
    l1_ttl: int = 300  # 5 minutes
    
    # L2 Cache (Warm)
    l2_size: int = 10000
    l2_ttl: int = 3600  # 1 hour
    
    # L3 Cache (Cold)
    l3_size: int = 100000
    l3_ttl: int = 86400  # 24 hours
    
    # Eviction policies
    l1_policy: EvictionPolicy = EvictionPolicy.LRU
    l2_policy: EvictionPolicy = EvictionPolicy.HYBRID
    l3_policy: EvictionPolicy = EvictionPolicy.ADAPTIVE
    
    # Compression
    enable_compression: bool = True
    compression_threshold: int = 1024  # Compress if larger than 1KB
    
    # Predictive features
    enable_prefetching: bool = True
    prefetch_threshold: float = 0.8  # Prefetch if access probability > 80%
    
    # Analytics
    enable_analytics: bool = True
    analytics_interval: int = 60  # Analytics every 60 seconds


@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    key: str
    value: Any
    created_at: float
    accessed_at: float
    access_count: int = 0
    size: int = 0
    compressed: bool = False
    level: CacheLevel = CacheLevel.L1_HOT
    
    def update_access(self) -> Any:
        """Update access metadata"""
        self.accessed_at = time.time()
        self.access_count += 1
    
    def is_expired(self, ttl: int) -> bool:
        """Check if entry is expired"""
        return time.time() - self.created_at > ttl


class SmartCache:
    """Advanced multi-level cache with intelligent strategies"""
    
    def __init__(self, config: CacheConfig):
        
    """__init__ function."""
self.config = config
        self.l1_cache: Dict[str, CacheEntry] = {}
        self.l2_cache: Dict[str, CacheEntry] = {}
        self.l3_cache: Dict[str, CacheEntry] = {}
        
        # Access tracking for analytics
        self.access_patterns: Dict[str, List[float]] = defaultdict(list)
        self.prefetch_queue: deque = deque(maxlen=1000)
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Analytics
        self.stats = {
            "total_requests": 0,
            "l1_hits": 0,
            "l2_hits": 0,
            "l3_hits": 0,
            "misses": 0,
            "evictions": 0,
            "compressions": 0,
            "prefetches": 0
        }
        
        # Start background tasks
        self._start_background_tasks()
    
    def _start_background_tasks(self) -> Any:
        """Start background maintenance tasks"""
        asyncio.create_task(self._cleanup_expired())
        asyncio.create_task(self._update_analytics())
        if self.config.enable_prefetching:
            asyncio.create_task(self._prefetch_predictive())
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from multi-level cache"""
        with self._lock:
            self.stats["total_requests"] += 1
            
            # Try L1 cache first (fastest)
            if key in self.l1_cache:
                entry = self.l1_cache[key]
                if not entry.is_expired(self.config.l1_ttl):
                    entry.update_access()
                    self.stats["l1_hits"] += 1
                    self._record_access_pattern(key)
                    return self._decompress_value(entry)
                else:
                    del self.l1_cache[key]
            
            # Try L2 cache
            if key in self.l2_cache:
                entry = self.l2_cache[key]
                if not entry.is_expired(self.config.l2_ttl):
                    entry.update_access()
                    self.stats["l2_hits"] += 1
                    self._record_access_pattern(key)
                    # Promote to L1
                    await self._promote_to_l1(key, entry)
                    return self._decompress_value(entry)
                else:
                    del self.l2_cache[key]
            
            # Try L3 cache
            if key in self.l3_cache:
                entry = self.l3_cache[key]
                if not entry.is_expired(self.config.l3_ttl):
                    entry.update_access()
                    self.stats["l3_hits"] += 1
                    self._record_access_pattern(key)
                    # Promote to L2
                    await self._promote_to_l2(key, entry)
                    return self._decompress_value(entry)
                else:
                    del self.l3_cache[key]
            
            # Cache miss
            self.stats["misses"] += 1
            return None
    
    async def set(self, key: str, value: Any, level: CacheLevel = CacheLevel.L1_HOT) -> None:
        """Set value in cache with intelligent placement"""
        with self._lock:
            # Compress if needed
            compressed_value, is_compressed = self._compress_value(value)
            
            entry = CacheEntry(
                key=key,
                value=compressed_value,
                created_at=time.time(),
                accessed_at=time.time(),
                size=len(str(compressed_value)),
                compressed=is_compressed,
                level=level
            )
            
            if is_compressed:
                self.stats["compressions"] += 1
            
            # Place in appropriate level
            if level == CacheLevel.L1_HOT:
                await self._set_in_l1(key, entry)
            elif level == CacheLevel.L2_WARM:
                await self._set_in_l2(key, entry)
            else:
                await self._set_in_l3(key, entry)
    
    async def _set_in_l1(self, key: str, entry: CacheEntry) -> None:
        """Set entry in L1 cache with eviction"""
        if len(self.l1_cache) >= self.config.l1_size:
            await self._evict_from_l1()
        
        self.l1_cache[key] = entry
    
    async def _set_in_l2(self, key: str, entry: CacheEntry) -> None:
        """Set entry in L2 cache with eviction"""
        if len(self.l2_cache) >= self.config.l2_size:
            await self._evict_from_l2()
        
        self.l2_cache[key] = entry
    
    async def _set_in_l3(self, key: str, entry: CacheEntry) -> None:
        """Set entry in L3 cache with eviction"""
        if len(self.l3_cache) >= self.config.l3_size:
            await self._evict_from_l3()
        
        self.l3_cache[key] = entry
    
    async def _promote_to_l1(self, key: str, entry: CacheEntry) -> None:
        """Promote entry to L1 cache"""
        if len(self.l1_cache) >= self.config.l1_size:
            await self._evict_from_l1()
        
        entry.level = CacheLevel.L1_HOT
        self.l1_cache[key] = entry
    
    async def _promote_to_l2(self, key: str, entry: CacheEntry) -> None:
        """Promote entry to L2 cache"""
        if len(self.l2_cache) >= self.config.l2_size:
            await self._evict_from_l2()
        
        entry.level = CacheLevel.L2_WARM
        self.l2_cache[key] = entry
    
    async def _evict_from_l1(self) -> None:
        """Evict entry from L1 cache based on policy"""
        if self.config.l1_policy == EvictionPolicy.LRU:
            oldest_key = min(self.l1_cache.keys(), 
                           key=lambda k: self.l1_cache[k].accessed_at)
        elif self.config.l1_policy == EvictionPolicy.LFU:
            oldest_key = min(self.l1_cache.keys(), 
                           key=lambda k: self.l1_cache[k].access_count)
        else:
            oldest_key = next(iter(self.l1_cache.keys()))
        
        evicted_entry = self.l1_cache.pop(oldest_key)
        self.stats["evictions"] += 1
        
        # Demote to L2 if still valid
        if not evicted_entry.is_expired(self.config.l2_ttl):
            await self._set_in_l2(oldest_key, evicted_entry)
    
    async def _evict_from_l2(self) -> None:
        """Evict entry from L2 cache based on policy"""
        if self.config.l2_policy == EvictionPolicy.HYBRID:
            # Hybrid: consider both access time and frequency
            scores = {}
            current_time = time.time()
            for key, entry in self.l2_cache.items():
                age = current_time - entry.accessed_at
                frequency = entry.access_count
                scores[key] = frequency / (age + 1)  # Avoid division by zero
            
            oldest_key = min(scores.keys(), key=lambda k: scores[k])
        else:
            oldest_key = min(self.l2_cache.keys(), 
                           key=lambda k: self.l2_cache[k].accessed_at)
        
        evicted_entry = self.l2_cache.pop(oldest_key)
        self.stats["evictions"] += 1
        
        # Demote to L3 if still valid
        if not evicted_entry.is_expired(self.config.l3_ttl):
            await self._set_in_l3(oldest_key, evicted_entry)
    
    async def _evict_from_l3(self) -> None:
        """Evict entry from L3 cache based on policy"""
        if self.config.l3_policy == EvictionPolicy.ADAPTIVE:
            # Adaptive: learn from access patterns
            scores = {}
            current_time = time.time()
            for key, entry in self.l3_cache.items():
                # Consider access pattern predictability
                pattern_score = self._calculate_pattern_score(key)
                age = current_time - entry.accessed_at
                frequency = entry.access_count
                scores[key] = (frequency * pattern_score) / (age + 1)
            
            oldest_key = min(scores.keys(), key=lambda k: scores[k])
        else:
            oldest_key = min(self.l3_cache.keys(), 
                           key=lambda k: self.l3_cache[k].accessed_at)
        
        del self.l3_cache[oldest_key]
        self.stats["evictions"] += 1
    
    def _compress_value(self, value: Any) -> Tuple[Any, bool]:
        """Compress value if beneficial"""
        if not self.config.enable_compression:
            return value, False
        
        value_str = json_dumps(value)
        if len(value_str) < self.config.compression_threshold:
            return value, False
        
        try:
            compressed = gzip.compress(value_str.encode())
            if len(compressed) < len(value_str):
                return compressed, True
        except Exception:
            pass
        
        return value, False
    
    def _decompress_value(self, entry: CacheEntry) -> Any:
        """Decompress value if needed"""
        if not entry.compressed:
            return entry.value
        
        try:
            decompressed = gzip.decompress(entry.value).decode()
            return json_loads(decompressed)
        except Exception:
            return entry.value
    
    def _record_access_pattern(self, key: str) -> None:
        """Record access pattern for predictive prefetching"""
        if not self.config.enable_prefetching:
            return
        
        current_time = time.time()
        self.access_patterns[key].append(current_time)
        
        # Keep only recent accesses
        if len(self.access_patterns[key]) > 100:
            self.access_patterns[key] = self.access_patterns[key][-50:]
    
    def _calculate_pattern_score(self, key: str) -> float:
        """Calculate pattern predictability score"""
        if key not in self.access_patterns:
            return 0.0
        
        accesses = self.access_patterns[key]
        if len(accesses) < 3:
            return 0.0
        
        # Calculate regularity of access intervals
        intervals = [accesses[i] - accesses[i-1] for i in range(1, len(accesses))]
        if not intervals:
            return 0.0
        
        # Lower variance = more predictable
        mean_interval = sum(intervals) / len(intervals)
        variance = sum((x - mean_interval) ** 2 for x in intervals) / len(intervals)
        
        # Convert to 0-1 score (lower variance = higher score)
        return max(0.0, 1.0 - (variance / (mean_interval ** 2 + 1)))
    
    async async def _prefetch_predictive(self) -> None:
        """Predictive prefetching based on access patterns"""
        if not self.config.enable_prefetching:
            return
        
        while True:
            await asyncio.sleep(30)  # Check every 30 seconds
            
            try:
                # Find keys with high predictability
                predictable_keys = []
                for key, accesses in self.access_patterns.items():
                    if len(accesses) >= 5:
                        pattern_score = self._calculate_pattern_score(key)
                        if pattern_score > self.config.prefetch_threshold:
                            predictable_keys.append((key, pattern_score))
                
                # Prefetch most predictable keys
                predictable_keys.sort(key=lambda x: x[1], reverse=True)
                for key, score in predictable_keys[:10]:  # Top 10
                    if key not in self.l1_cache and key not in self.l2_cache:
                        # This would trigger prefetching logic
                        self.stats["prefetches"] += 1
                        logger.debug(f"Prefetching key: {key} (score: {score:.2f})")
            
            except Exception as e:
                logger.error(f"Prefetch error: {e}")
    
    async def _cleanup_expired(self) -> None:
        """Cleanup expired entries periodically"""
        while True:
            await asyncio.sleep(60)  # Check every minute
            
            try:
                with self._lock:
                    # Clean L1
                    expired_keys = [
                        key for key, entry in self.l1_cache.items()
                        if entry.is_expired(self.config.l1_ttl)
                    ]
                    for key in expired_keys:
                        del self.l1_cache[key]
                    
                    # Clean L2
                    expired_keys = [
                        key for key, entry in self.l2_cache.items()
                        if entry.is_expired(self.config.l2_ttl)
                    ]
                    for key in expired_keys:
                        del self.l2_cache[key]
                    
                    # Clean L3
                    expired_keys = [
                        key for key, entry in self.l3_cache.items()
                        if entry.is_expired(self.config.l3_ttl)
                    ]
                    for key in expired_keys:
                        del self.l3_cache[key]
            
            except Exception as e:
                logger.error(f"Cleanup error: {e}")
    
    async def _update_analytics(self) -> None:
        """Update cache analytics periodically"""
        if not self.config.enable_analytics:
            return
        
        while True:
            await asyncio.sleep(self.config.analytics_interval)
            
            try:
                total_hits = self.stats["l1_hits"] + self.stats["l2_hits"] + self.stats["l3_hits"]
                total_requests = self.stats["total_requests"]
                
                if total_requests > 0:
                    hit_rate = total_hits / total_requests
                    logger.info(f"Cache Analytics - Hit Rate: {hit_rate:.2%}, "
                              f"L1: {self.stats['l1_hits']}, "
                              f"L2: {self.stats['l2_hits']}, "
                              f"L3: {self.stats['l3_hits']}, "
                              f"Misses: {self.stats['misses']}")
            
            except Exception as e:
                logger.error(f"Analytics error: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_hits = self.stats["l1_hits"] + self.stats["l2_hits"] + self.stats["l3_hits"]
        total_requests = self.stats["total_requests"]
        
        return {
            "total_requests": total_requests,
            "total_hits": total_hits,
            "hit_rate": total_hits / total_requests if total_requests > 0 else 0.0,
            "l1_hits": self.stats["l1_hits"],
            "l2_hits": self.stats["l2_hits"],
            "l3_hits": self.stats["l3_hits"],
            "misses": self.stats["misses"],
            "evictions": self.stats["evictions"],
            "compressions": self.stats["compressions"],
            "prefetches": self.stats["prefetches"],
            "l1_size": len(self.l1_cache),
            "l2_size": len(self.l2_cache),
            "l3_size": len(self.l3_cache),
            "total_size": len(self.l1_cache) + len(self.l2_cache) + len(self.l3_cache)
        }
    
    async def clear(self) -> None:
        """Clear all cache levels"""
        with self._lock:
            self.l1_cache.clear()
            self.l2_cache.clear()
            self.l3_cache.clear()
            self.access_patterns.clear()
            self.prefetch_queue.clear()


# Cache decorators
def smart_cache(ttl: int = 3600, level: CacheLevel = CacheLevel.L1_HOT):
    """Smart cache decorator with multi-level support"""
    def decorator(func) -> Any:
        cache = SmartCache(CacheConfig())
        
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            # Generate cache key
            cache_key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"
            
            # Try cache first
            cached_result = await cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Cache result
            await cache.set(cache_key, result, level)
            
            return result
        
        return wrapper
    return decorator


def predictive_cache(ttl: int = 3600, prefetch_threshold: float = 0.8):
    """Predictive cache decorator with prefetching"""
    def decorator(func) -> Any:
        config = CacheConfig(
            enable_prefetching=True,
            prefetch_threshold=prefetch_threshold
        )
        cache = SmartCache(config)
        
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            cache_key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"
            
            # Try cache first
            cached_result = await cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Cache result
            await cache.set(cache_key, result)
            
            return result
        
        return wrapper
    return decorator


# Global cache instance
smart_cache_instance = SmartCache(CacheConfig())


# Utility functions
@njit if NUMBA_AVAILABLE else lambda f: f
def fast_cache_key(data: str) -> int:
    """Fast cache key generation with JIT optimization"""
    return hash(data)


async def cache_warmup(warmup_funcs: List[Callable]) -> None:
    """Warm up cache with common operations"""
    logger.info("Starting cache warmup...")
    
    warmup_tasks = []
    for func in warmup_funcs:
        warmup_tasks.append(func())
    
    await asyncio.gather(*warmup_tasks, return_exceptions=True)
    logger.info("Cache warmup completed")


async def cache_analytics(cache: SmartCache) -> Dict[str, Any]:
    """Get detailed cache analytics"""
    stats = cache.get_stats()
    
    # Calculate additional metrics
    if stats["total_requests"] > 0:
        stats["l1_hit_rate"] = stats["l1_hits"] / stats["total_requests"]
        stats["l2_hit_rate"] = stats["l2_hits"] / stats["total_requests"]
        stats["l3_hit_rate"] = stats["l3_hits"] / stats["total_requests"]
        stats["miss_rate"] = stats["misses"] / stats["total_requests"]
    
    return stats 