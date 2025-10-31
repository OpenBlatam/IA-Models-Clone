"""
Ultra-Efficient Async Cache Layer
================================

High-performance async caching with LRU, TTL, and memory optimization.
Designed for maximum throughput and minimal memory overhead.

Key Features:
- Async-first design
- LRU eviction policy
- TTL expiration
- Memory-efficient storage
- Thread-safe operations
- Performance monitoring
- Cache statistics

Author: TruthGPT Development Team
Version: 1.0.0 - Ultra-Efficient
License: MIT
"""

import asyncio
import time
import hashlib
import weakref
import gc
from typing import Any, Dict, Optional, Union, Callable, Awaitable
from dataclasses import dataclass, field
from collections import OrderedDict
import logging
from functools import wraps
import threading

logger = logging.getLogger(__name__)

@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    value: Any
    created_at: float
    accessed_at: float
    access_count: int = 0
    ttl: Optional[float] = None

@dataclass
class CacheStats:
    """Cache statistics for monitoring."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    expired: int = 0
    total_size: int = 0
    memory_usage: int = 0

class AsyncLRUCache:
    """Async LRU cache with TTL support."""
    
    def __init__(
        self,
        max_size: int = 1024,
        default_ttl: Optional[float] = None,
        cleanup_interval: float = 60.0
    ):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cleanup_interval = cleanup_interval
        
        # Thread-safe storage
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = asyncio.Lock()
        self._stats = CacheStats()
        
        # Background cleanup task
        self._cleanup_task: Optional[asyncio.Task] = None
        self._start_cleanup_task()
    
    def _start_cleanup_task(self):
        """Start background cleanup task."""
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
    
    async def _cleanup_loop(self):
        """Background cleanup loop."""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval)
                await self._cleanup_expired()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")
    
    async def _cleanup_expired(self):
        """Clean up expired entries."""
        async with self._lock:
            current_time = time.time()
            expired_keys = []
            
            for key, entry in self._cache.items():
                if entry.ttl and current_time - entry.created_at > entry.ttl:
                    expired_keys.append(key)
            
            for key in expired_keys:
                self._cache.pop(key, None)
                self._stats.expired += 1
            
            # LRU eviction if still over limit
            while len(self._cache) > self.max_size:
                if self._cache:
                    oldest_key = next(iter(self._cache))
                    self._cache.pop(oldest_key, None)
                    self._stats.evictions += 1
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        async with self._lock:
            if key not in self._cache:
                self._stats.misses += 1
                return None
            
            entry = self._cache[key]
            
            # Check TTL
            if entry.ttl and time.time() - entry.created_at > entry.ttl:
                self._cache.pop(key, None)
                self._stats.expired += 1
                self._stats.misses += 1
                return None
            
            # Update access info
            entry.accessed_at = time.time()
            entry.access_count += 1
            
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            
            self._stats.hits += 1
            return entry.value
    
    async def set(
        self, 
        key: str, 
        value: Any, 
        ttl: Optional[float] = None
    ) -> None:
        """Set value in cache."""
        async with self._lock:
            current_time = time.time()
            ttl_to_use = ttl or self.default_ttl
            
            # Create entry
            entry = CacheEntry(
                value=value,
                created_at=current_time,
                accessed_at=current_time,
                ttl=ttl_to_use
            )
            
            # Remove existing entry if present
            if key in self._cache:
                self._cache.pop(key, None)
            
            # Add new entry
            self._cache[key] = entry
            
            # Evict if over limit
            while len(self._cache) > self.max_size:
                if self._cache:
                    oldest_key = next(iter(self._cache))
                    self._cache.pop(oldest_key, None)
                    self._stats.evictions += 1
    
    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
        async with self._lock:
            if key in self._cache:
                self._cache.pop(key, None)
                return True
            return False
    
    async def clear(self) -> None:
        """Clear all cache entries."""
        async with self._lock:
            self._cache.clear()
            self._stats = CacheStats()
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        async with self._lock:
            if key not in self._cache:
                return False
            
            entry = self._cache[key]
            
            # Check TTL
            if entry.ttl and time.time() - entry.created_at > entry.ttl:
                self._cache.pop(key, None)
                self._stats.expired += 1
                return False
            
            return True
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        async with self._lock:
            total_requests = self._stats.hits + self._stats.misses
            hit_ratio = self._stats.hits / max(total_requests, 1)
            
            return {
                "hits": self._stats.hits,
                "misses": self._stats.misses,
                "hit_ratio": hit_ratio,
                "evictions": self._stats.evictions,
                "expired": self._stats.expired,
                "size": len(self._cache),
                "max_size": self.max_size,
                "default_ttl": self.default_ttl
            }
    
    async def close(self):
        """Close cache and cleanup resources."""
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        await self.clear()

# Global cache instances
_text_cache = AsyncLRUCache(max_size=1024, default_ttl=300)  # 5 minutes
_preview_cache = AsyncLRUCache(max_size=512, default_ttl=600)  # 10 minutes
_topic_cache = AsyncLRUCache(max_size=512, default_ttl=300)  # 5 minutes
_general_cache = AsyncLRUCache(max_size=2048, default_ttl=180)  # 3 minutes

def generate_cache_key(data: bytes, operation: str, params: str = "") -> str:
    """Generate cache key from data and parameters."""
    data_hash = hashlib.md5(data).hexdigest()
    return f"{operation}:{data_hash}:{params}"

async def get_from_cache(
    cache: AsyncLRUCache, 
    key: str
) -> Optional[Any]:
    """Get value from specified cache."""
    return await cache.get(key)

async def set_in_cache(
    cache: AsyncLRUCache, 
    key: str, 
    value: Any, 
    ttl: Optional[float] = None
) -> None:
    """Set value in specified cache."""
    await cache.set(key, value, ttl)

async def delete_from_cache(cache: AsyncLRUCache, key: str) -> bool:
    """Delete key from specified cache."""
    return await cache.delete(key)

# Cache decorator for async functions
def cached(
    cache: AsyncLRUCache,
    key_func: Optional[Callable] = None,
    ttl: Optional[float] = None
):
    """Decorator for caching async function results."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                # Default key generation
                key_data = str(args) + str(sorted(kwargs.items()))
                cache_key = hashlib.md5(key_data.encode()).hexdigest()
            
            # Try to get from cache
            cached_result = await cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Cache result
            await cache.set(cache_key, result, ttl)
            
            return result
        
        return wrapper
    return decorator

# Specific cache decorators
def cached_text(ttl: Optional[float] = None):
    """Decorator for caching text extraction results."""
    return cached(_text_cache, ttl=ttl)

def cached_preview(ttl: Optional[float] = None):
    """Decorator for caching preview generation results."""
    return cached(_preview_cache, ttl=ttl)

def cached_topics(ttl: Optional[float] = None):
    """Decorator for caching topic extraction results."""
    return cached(_topic_cache, ttl=ttl)

def cached_general(ttl: Optional[float] = None):
    """Decorator for general caching."""
    return cached(_general_cache, ttl=ttl)

# Cache management functions
async def get_all_cache_stats() -> Dict[str, Any]:
    """Get statistics from all caches."""
    stats = {}
    
    stats["text_cache"] = await _text_cache.get_stats()
    stats["preview_cache"] = await _preview_cache.get_stats()
    stats["topic_cache"] = await _topic_cache.get_stats()
    stats["general_cache"] = await _general_cache.get_stats()
    
    # Aggregate stats
    total_hits = sum(s["hits"] for s in stats.values())
    total_misses = sum(s["misses"] for s in stats.values())
    total_size = sum(s["size"] for s in stats.values())
    
    stats["aggregate"] = {
        "total_hits": total_hits,
        "total_misses": total_misses,
        "total_hit_ratio": total_hits / max(total_hits + total_misses, 1),
        "total_size": total_size,
        "total_evictions": sum(s["evictions"] for s in stats.values()),
        "total_expired": sum(s["expired"] for s in stats.values())
    }
    
    return stats

async def clear_all_caches() -> Dict[str, int]:
    """Clear all caches and return counts."""
    counts = {}
    
    counts["text_cache"] = len(_text_cache._cache)
    counts["preview_cache"] = len(_preview_cache._cache)
    counts["topic_cache"] = len(_topic_cache._cache)
    counts["general_cache"] = len(_general_cache._cache)
    
    await asyncio.gather(
        _text_cache.clear(),
        _preview_cache.clear(),
        _topic_cache.clear(),
        _general_cache.clear()
    )
    
    return counts

async def cleanup_all_caches() -> int:
    """Cleanup expired entries from all caches."""
    await asyncio.gather(
        _text_cache._cleanup_expired(),
        _preview_cache._cleanup_expired(),
        _topic_cache._cleanup_expired(),
        _general_cache._cleanup_expired()
    )
    
    # Return total cleaned
    stats = await get_all_cache_stats()
    return stats["aggregate"]["total_expired"]

# Cache warming functions
async def warm_cache_with_data(
    data: bytes,
    operations: List[str] = None
) -> Dict[str, Any]:
    """Warm cache with common operations."""
    if operations is None:
        operations = ["text", "preview", "topics"]
    
    results = {}
    
    for operation in operations:
        key = generate_cache_key(data, operation)
        
        if operation == "text":
            # This would call actual text extraction
            results[operation] = f"Warmed text cache for {key}"
        elif operation == "preview":
            # This would call actual preview generation
            results[operation] = f"Warmed preview cache for {key}"
        elif operation == "topics":
            # This would call actual topic extraction
            results[operation] = f"Warmed topics cache for {key}"
    
    return results

# Performance monitoring
class CachePerformanceMonitor:
    """Monitor cache performance metrics."""
    
    def __init__(self):
        self.metrics: Dict[str, List[float]] = {
            "get_times": [],
            "set_times": [],
            "hit_ratios": []
        }
        self._lock = asyncio.Lock()
    
    async def record_get_time(self, duration: float):
        """Record get operation duration."""
        async with self._lock:
            self.metrics["get_times"].append(duration)
            if len(self.metrics["get_times"]) > 1000:
                self.metrics["get_times"] = self.metrics["get_times"][-500:]
    
    async def record_set_time(self, duration: float):
        """Record set operation duration."""
        async with self._lock:
            self.metrics["set_times"].append(duration)
            if len(self.metrics["set_times"]) > 1000:
                self.metrics["set_times"] = self.metrics["set_times"][-500:]
    
    async def record_hit_ratio(self, ratio: float):
        """Record hit ratio."""
        async with self._lock:
            self.metrics["hit_ratios"].append(ratio)
            if len(self.metrics["hit_ratios"]) > 100:
                self.metrics["hit_ratios"] = self.metrics["hit_ratios"][-50:]
    
    async def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        async with self._lock:
            stats = {}
            
            if self.metrics["get_times"]:
                stats["avg_get_time"] = sum(self.metrics["get_times"]) / len(self.metrics["get_times"])
                stats["max_get_time"] = max(self.metrics["get_times"])
            
            if self.metrics["set_times"]:
                stats["avg_set_time"] = sum(self.metrics["set_times"]) / len(self.metrics["set_times"])
                stats["max_set_time"] = max(self.metrics["set_times"])
            
            if self.metrics["hit_ratios"]:
                stats["avg_hit_ratio"] = sum(self.metrics["hit_ratios"]) / len(self.metrics["hit_ratios"])
            
            return stats

# Global performance monitor
_performance_monitor = CachePerformanceMonitor()

# Enhanced cache decorator with performance monitoring
def monitored_cached(
    cache: AsyncLRUCache,
    key_func: Optional[Callable] = None,
    ttl: Optional[float] = None
):
    """Decorator for caching with performance monitoring."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                key_data = str(args) + str(sorted(kwargs.items()))
                cache_key = hashlib.md5(key_data.encode()).hexdigest()
            
            # Try to get from cache with timing
            start_time = time.time()
            cached_result = await cache.get(cache_key)
            get_time = time.time() - start_time
            
            await _performance_monitor.record_get_time(get_time)
            
            if cached_result is not None:
                return cached_result
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Cache result with timing
            start_time = time.time()
            await cache.set(cache_key, result, ttl)
            set_time = time.time() - start_time
            
            await _performance_monitor.record_set_time(set_time)
            
            return result
        
        return wrapper
    return decorator

# Cleanup on shutdown
async def cleanup_caches():
    """Cleanup all caches on shutdown."""
    await asyncio.gather(
        _text_cache.close(),
        _preview_cache.close(),
        _topic_cache.close(),
        _general_cache.close()
    )
