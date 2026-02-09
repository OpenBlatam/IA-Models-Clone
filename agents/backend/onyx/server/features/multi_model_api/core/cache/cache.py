"""
Main cache implementation using modular components
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional, List, Callable, Awaitable

from .constants import (
    DEFAULT_L1_MAX_SIZE,
    DEFAULT_L1_TTL,
    DEFAULT_L2_TTL,
    DEFAULT_COMPRESSION_THRESHOLD,
    MAX_PROMOTION_TASKS
)
from .serialization import Serializer
from .compression import Compressor
from .stats import CacheStats
from .tags import TagManager
from .utils import generate_key
from .layers import L1Cache, L2Cache, L3Cache

logger = logging.getLogger(__name__)

CacheCallback = Callable[[str, Any], Awaitable[None]]


class EnhancedCache:
    """
    Enhanced multi-tier caching system with optimizations
    L1: In-memory TTL/LRU cache (fastest, using cachetools)
    L2: Redis cache with connection pooling (distributed, persistent)
    L3: Disk cache (optional, persistent across restarts)
    """
    
    def __init__(
        self,
        l1_max_size: int = DEFAULT_L1_MAX_SIZE,
        l1_ttl: int = DEFAULT_L1_TTL,
        l2_enabled: bool = True,
        l2_redis_url: str = "redis://localhost:6379/0",
        l2_ttl: int = DEFAULT_L2_TTL,
        enable_compression: bool = True,
        compression_threshold: int = DEFAULT_COMPRESSION_THRESHOLD
    ):
        """Initialize enhanced multi-tier cache
        
        Args:
            l1_max_size: Maximum number of entries in L1 cache
            l1_ttl: Time-to-live for L1 cache entries in seconds
            l2_enabled: Enable Redis L2 cache
            l2_redis_url: Redis connection URL
            l2_ttl: Time-to-live for L2 cache entries in seconds
            enable_compression: Enable compression for large values
            compression_threshold: Minimum size in bytes to compress
            
        Raises:
            ValueError: If invalid parameters provided
        """
        if l1_max_size <= 0:
            raise ValueError("l1_max_size must be positive")
        if l1_ttl <= 0:
            raise ValueError("l1_ttl must be positive")
        if l2_ttl <= 0:
            raise ValueError("l2_ttl must be positive")
        if compression_threshold < 0:
            raise ValueError("compression_threshold must be non-negative")
        if not isinstance(l2_redis_url, str) or not l2_redis_url:
            raise ValueError("l2_redis_url must be a non-empty string")
        
        # Initialize cache layers
        self.l1 = L1Cache(max_size=l1_max_size, ttl=l1_ttl)
        self.l2 = L2Cache(redis_url=l2_redis_url, ttl=l2_ttl) if l2_enabled else None
        self.l3 = L3Cache()
        
        # Initialize components
        self.serializer = Serializer()
        self.compressor = Compressor(enabled=enable_compression, threshold=compression_threshold)
        self.stats = CacheStats()
        self.tags = TagManager()
        
        # Task tracking for L1 promotion
        self._promotion_tasks: set = set()
        
        # Cache version
        self._cache_version: int = 1
        
        # Cache callbacks
        self._on_hit: Optional[CacheCallback] = None
        self._on_miss: Optional[CacheCallback] = None
        self._on_evict: Optional[CacheCallback] = None
        self._on_set: Optional[CacheCallback] = None
        
        # Cache strategy
        self._write_strategy: str = "write-through"
    
    def _generate_key(self, prefix: str, *args, **kwargs) -> str:
        """Generate cache key (delegates to utils)"""
        return generate_key(prefix, *args, **kwargs)
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache (L1 first, then L2, then L3)
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
            
        Raises:
            ValueError: If key is invalid
        """
        if not key or not isinstance(key, str):
            raise ValueError("key must be a non-empty string")
        
        start_time = time.time()
        
        # Try L1
        value = await self.l1.get(key)
        if value is not None:
            latency = (time.time() - start_time) * 1000
            await self.stats.increment_hit()
            await self.stats.record_latency(latency)
            if self._on_hit:
                asyncio.create_task(self._on_hit(key, value))
            return value
        
        # Try L2 with error resilience
        if self.l2 and self.l2.enabled:
            try:
                cached_bytes = await self.l2.get(key, retry=True)
                if cached_bytes:
                    cached_bytes = self.compressor.decompress(cached_bytes)
                    value = self.serializer.deserialize(cached_bytes)
                    await self._promote_to_l1(key, value)
                    latency = (time.time() - start_time) * 1000
                    await self.stats.increment_hit()
                    await self.stats.record_latency(latency)
                    if self._on_hit:
                        asyncio.create_task(self._on_hit(key, value))
                    return value
            except Exception as e:
                logger.debug(f"L2 cache get failed for key {key}, continuing to L3: {e}")
        
        # Try L3
        if self.l3.enabled:
            disk_value = self.l3.get(key)
            if disk_value is not None:
                await self._promote_to_l1(key, disk_value)
                if self.l2:
                    await self.set(key, disk_value, ttl=self.l1.ttl, level="l2")
                latency = (time.time() - start_time) * 1000
                await self.stats.increment_hit()
                await self.stats.record_latency(latency)
                if self._on_hit:
                    asyncio.create_task(self._on_hit(key, disk_value))
                return disk_value
        
        latency = (time.time() - start_time) * 1000
        await self.stats.increment_miss()
        await self.stats.record_latency(latency)
        if self._on_miss:
            asyncio.create_task(self._on_miss(key, None))
        return None
    
    async def _promote_to_l1(self, key: str, value: Any):
        """Promote value from L2/L3 to L1 with task tracking"""
        task_ref = {"task": None}
        
        async def promote_task():
            try:
                await self.l1.set(key, value, ttl=self.l1.ttl)
            except Exception as e:
                logger.debug(f"L1 promotion failed for {key}: {e}")
            finally:
                if task_ref["task"]:
                    self._promotion_tasks.discard(task_ref["task"])
        
        task_ref["task"] = asyncio.create_task(promote_task())
        self._promotion_tasks.add(task_ref["task"])
        
        if len(self._promotion_tasks) > MAX_PROMOTION_TASKS:
            completed = [t for t in self._promotion_tasks if t.done()]
            self._promotion_tasks.difference_update(completed)
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        level: str = "both",
        tags: Optional[List[str]] = None
    ) -> bool:
        """Set value in cache
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (defaults to l1_ttl)
            level: Cache level ("l1", "l2", or "both")
            tags: Optional list of tags for invalidation
            
        Returns:
            True if successful, False otherwise
            
        Raises:
            ValueError: If parameters are invalid
        """
        if not key or not isinstance(key, str):
            raise ValueError("key must be a non-empty string")
        if level not in ("l1", "l2", "both"):
            raise ValueError("level must be 'l1', 'l2', or 'both'")
        if ttl is not None and ttl <= 0:
            raise ValueError("ttl must be positive")
        if tags is not None and not isinstance(tags, list):
            raise ValueError("tags must be a list")
        
        ttl = ttl or self.l1.ttl
        
        # Handle tags
        if tags:
            await self.tags.add_tags(key, tags)
        
        # Set in L1
        if level in ("l1", "both"):
            # Check if we need to evict before setting
            if self.l1.size() >= self.l1.max_size and key not in self.l1.keys():
                evicted_key = await self.l1.evict_oldest()
                if evicted_key:
                    await self.stats.increment_eviction()
                    if self._on_evict:
                        evicted_value = await self.l1.get(evicted_key)
                        asyncio.create_task(self._on_evict(evicted_key, evicted_value))
            
            await self.l1.set(key, value, ttl=ttl)
        
        # Set in L2 with error resilience
        if level in ("l2", "both") and self.l2 and self.l2.enabled:
            try:
                serialized = self.serializer.serialize(value)
                compressed = self.compressor.compress(serialized)
                success = await self.l2.set(key, compressed, ttl=ttl, retry=True)
                if not success and level == "l2":
                    logger.warning(f"L2 cache set failed for key {key}, falling back to L3")
                    if self.l3.enabled:
                        self.l3.set(key, value, expire=ttl)
                    return False
                
                if self._on_set:
                    asyncio.create_task(self._on_set(key, value))
                
                # Also store in L3 if enabled
                if self.l3.enabled:
                    self.l3.set(key, value, expire=ttl)
            except Exception as e:
                logger.warning(f"L2 cache set error for key {key}: {e}, falling back to L3")
                if self.l3.enabled:
                    self.l3.set(key, value, expire=ttl)
                if level == "l2":
                    return False
        
        return True
    
    async def delete(self, key: str) -> bool:
        """Delete from cache
        
        Args:
            key: Cache key to delete
            
        Returns:
            True if deleted, False otherwise
            
        Raises:
            ValueError: If key is invalid
        """
        if not key or not isinstance(key, str):
            raise ValueError("key must be a non-empty string")
        
        deleted = False
        
        # Delete from L1
        if await self.l1.delete(key):
            deleted = True
        
        # Remove tags
        await self.tags.remove_tags(key)
        
        # Delete from L2
        if self.l2 and self.l2.enabled:
            if await self.l2.delete(key):
                deleted = True
        
        # Delete from L3
        if self.l3.enabled:
            if self.l3.delete(key):
                deleted = True
        
        return deleted
    
    async def clear(self, level: Optional[str] = None) -> bool:
        """Clear cache
        
        Args:
            level: Cache level to clear ("l1", "l2", "l3", or None for all)
            
        Returns:
            True if successful, False otherwise
        """
        if level is None or level == "l1":
            await self.l1.clear()
        
        if (level is None or level == "l2") and self.l2 and self.l2.enabled:
            if not await self.l2.clear():
                return False
        
        if (level is None or level == "l3") and self.l3.enabled:
            if not self.l3.clear():
                return False
        
        return True
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics with advanced metrics"""
        l1_size = self.l1.size()
        latency_stats = self.stats.get_latency_stats()
        
        stats = {
            "hits": self.stats.hits,
            "misses": self.stats.misses,
            "hit_rate": round(self.stats.hit_rate, 2),
            "evictions": self.stats.evictions,
            "total_requests": self.stats.total_requests,
            "l1_size": l1_size,
            "l1_max_size": self.l1.max_size,
            "l1_utilization": round((l1_size / self.l1.max_size) * 100, 2) if self.l1.max_size > 0 else 0.0,
            "l2_enabled": self.l2.enabled if self.l2 else False,
            "l3_enabled": self.l3.enabled,
            "compression_enabled": self.compressor.enabled,
            "serialization": self.serializer.get_available_formats(),
            "latency_ms": {
                "avg": round(latency_stats["avg"], 3),
                "min": round(latency_stats["min"], 3),
                "max": round(latency_stats["max"], 3),
                "p50": round(latency_stats["p50"], 3),
                "p95": round(latency_stats["p95"], 3),
                "p99": round(latency_stats["p99"], 3)
            },
            "tags_count": await self.tags.get_tag_count(),
            "cache_version": self._cache_version
        }
        
        if self.l2 and self.l2.enabled:
            l2_stats = await self.l2.get_stats()
            stats.update({
                "l2_memory_usage": l2_stats.get("memory_usage", "N/A"),
                "l2_keys": l2_stats.get("keys", 0)
            })
        
        return stats
    
    async def get_many(self, keys: List[str]) -> Dict[str, Any]:
        """Get multiple values from cache efficiently"""
        if not isinstance(keys, list):
            raise ValueError("keys must be a list")
        if not keys:
            return {}
        
        results = {}
        
        # Get from L1
        for key in keys:
            value = await self.l1.get(key)
            if value is not None:
                results[key] = value
                await self.stats.increment_hit()
        
        # Get remaining from L2
        remaining_keys = [k for k in keys if k not in results]
        if remaining_keys and self.l2 and self.l2.enabled:
            cached_values = await self.l2.mget(remaining_keys)
            for key, cached_bytes in zip(remaining_keys, cached_values):
                if cached_bytes:
                    cached_bytes = self.compressor.decompress(cached_bytes)
                    value = self.serializer.deserialize(cached_bytes)
                    results[key] = value
                    await self._promote_to_l1(key, value)
                    await self.stats.increment_hit()
                else:
                    await self.stats.increment_miss()
        else:
            for key in remaining_keys:
                await self.stats.increment_miss()
        
        return results
    
    async def set_many(self, items: Dict[str, Any], ttl: Optional[int] = None, level: str = "both") -> int:
        """Set multiple values in cache efficiently"""
        if not isinstance(items, dict):
            raise ValueError("items must be a dictionary")
        if not items:
            return 0
        if level not in ("l1", "l2", "both"):
            raise ValueError("level must be 'l1', 'l2', or 'both'")
        if ttl is not None and ttl <= 0:
            raise ValueError("ttl must be positive")
        
        ttl = ttl or self.l1.ttl
        success_count = 0
        
        # Set in L1
        if level in ("l1", "both"):
            for key, value in items.items():
                await self.l1.set(key, value, ttl=ttl)
                success_count += 1
        
        # Set in L2
        if level in ("l2", "both") and self.l2 and self.l2.enabled:
            l2_items = {}
            for key, value in items.items():
                serialized = self.serializer.serialize(value)
                compressed = self.compressor.compress(serialized)
                l2_items[key] = compressed
            
            if await self.l2.mset(l2_items, ttl=ttl):
                success_count = len(items)
        
        return success_count
    
    async def invalidate_by_tag(self, tag: str) -> int:
        """Invalidate all cache entries with a specific tag"""
        if not tag or not isinstance(tag, str):
            raise ValueError("tag must be a non-empty string")
        
        keys = await self.tags.get_keys_by_tag(tag)
        count = 0
        for key in keys:
            if await self.delete(key):
                count += 1
        return count
    
    async def invalidate_by_pattern(self, pattern: str) -> int:
        """Invalidate cache entries matching a pattern"""
        if not pattern or not isinstance(pattern, str):
            raise ValueError("pattern must be a non-empty string")
        
        all_keys = self.l1.keys()
        if self.l2 and self.l2.enabled:
            l2_keys = await self.l2.get_keys(pattern)
            all_keys.extend(k for k in l2_keys if k not in all_keys)
        
        matching_keys = await self.tags.get_keys_by_pattern(pattern, all_keys)
        count = 0
        for key in matching_keys:
            if await self.delete(key):
                count += 1
        return count
    
    async def get_by_tag(self, tag: str) -> Dict[str, Any]:
        """Get all cache entries with a specific tag"""
        if not tag or not isinstance(tag, str):
            raise ValueError("tag must be a non-empty string")
        
        results = {}
        keys = await self.tags.get_keys_by_tag(tag)
        for key in keys:
            value = await self.get(key)
            if value is not None:
                results[key] = value
        return results
    
    async def refresh_ttl(self, key: str, ttl: Optional[int] = None) -> bool:
        """Refresh TTL for an existing cache entry"""
        if not key or not isinstance(key, str):
            raise ValueError("key must be a non-empty string")
        if ttl is not None and ttl <= 0:
            raise ValueError("ttl must be positive")
        
        ttl = ttl or self.l1.ttl
        
        # Refresh in L1 (re-set with new TTL)
        value = await self.l1.get(key)
        if value is not None:
            await self.l1.set(key, value, ttl=ttl)
            return True
        
        # Refresh in L2
        if self.l2 and self.l2.enabled:
            if await self.l2.expire(key, ttl):
                return True
        
        return False
    
    async def warm_cache(self, items: Dict[str, Any], ttl: Optional[int] = None) -> int:
        """Warm cache with precomputed values"""
        return await self.set_many(items, ttl=ttl, level="both")
    
    async def get_keys(self, pattern: Optional[str] = None) -> List[str]:
        """Get all cache keys, optionally filtered by pattern"""
        if pattern is not None and (not isinstance(pattern, str) or not pattern):
            raise ValueError("pattern must be a non-empty string or None")
        
        keys = self.l1.keys()
        
        if self.l2 and self.l2.enabled:
            l2_keys = await self.l2.get_keys(pattern or "*")
            keys.extend(k for k in l2_keys if k not in keys)
        
        if pattern:
            matching_keys = await self.tags.get_keys_by_pattern(pattern, keys)
            return matching_keys
        
        return keys
    
    async def exists(self, key: str) -> bool:
        """Check if a key exists in cache"""
        if not key or not isinstance(key, str):
            raise ValueError("key must be a non-empty string")
        
        if await self.l1.get(key) is not None:
            return True
        
        if self.l2 and self.l2.enabled:
            if await self.l2.exists(key):
                return True
        
        return False
    
    def increment_version(self):
        """Increment cache version to invalidate all entries"""
        self._cache_version += 1
        logger.info(f"Cache version incremented to {self._cache_version}")
    
    async def get_diagnostics(self) -> Dict[str, Any]:
        """Get detailed diagnostics information"""
        diagnostics = {
            "cache_version": self._cache_version,
            "l1_config": {
                "max_size": self.l1.max_size,
                "current_size": self.l1.size(),
                "ttl": self.l1.ttl
            },
            "l2_config": await self.l2.get_stats() if self.l2 else {"enabled": False},
            "l3_config": {
                "enabled": self.l3.enabled
            },
            "compression": {
                "enabled": self.compressor.enabled,
                "threshold": self.compressor.threshold
            },
            "tags": {
                "total_tags": await self.tags.get_tag_count(),
                "tagged_keys": await self.tags.get_tagged_key_count()
            },
            "tasks": {
                "promotion_tasks": len(self._promotion_tasks),
                "completed_tasks": len([t for t in self._promotion_tasks if t.done()])
            },
            "stats": await self.get_stats()
        }
        
        return diagnostics
    
    def set_callbacks(
        self,
        on_hit: Optional[CacheCallback] = None,
        on_miss: Optional[CacheCallback] = None,
        on_evict: Optional[CacheCallback] = None,
        on_set: Optional[CacheCallback] = None
    ):
        """Set cache event callbacks"""
        self._on_hit = on_hit
        self._on_miss = on_miss
        self._on_evict = on_evict
        self._on_set = on_set
    
    def set_write_strategy(self, strategy: str):
        """Set cache write strategy: 'write-through' or 'write-back'"""
        if strategy not in ("write-through", "write-back"):
            raise ValueError("Strategy must be 'write-through' or 'write-back'")
        self._write_strategy = strategy
    
    @asynccontextmanager
    async def transaction(self):
        """Context manager for cache transactions"""
        operations = []
        
        class CacheTransaction:
            def __init__(self, cache, ops):
                self.cache = cache
                self.ops = ops
            
            async def get(self, key: str):
                result = await self.cache.get(key)
                self.ops.append(("get", key, result))
                return result
            
            async def set(self, key: str, value: Any, ttl: Optional[int] = None):
                self.ops.append(("set", key, value, ttl))
                return True
        
        transaction = CacheTransaction(self, operations)
        try:
            yield transaction
            for op in operations:
                if op[0] == "set":
                    await self.set(op[1], op[2], ttl=op[3] if len(op) > 3 else None)
        except Exception as e:
            logger.error(f"Cache transaction failed: {e}")
            raise
    
    async def auto_tune_size(self, target_hit_rate: float = 80.0, min_size: int = 100, max_size: int = 10000):
        """Auto-tune cache size based on hit rate"""
        current_hit_rate = self.stats.hit_rate
        current_size = self.l1.size()
        
        if current_hit_rate < target_hit_rate and current_size >= self.l1.max_size:
            new_size = min(int(current_size * 1.2), max_size)
            if new_size > self.l1.max_size:
                self.l1.max_size = new_size
                logger.info(f"Auto-tuned cache size to {new_size} (hit rate: {current_hit_rate:.2f}%)")
        elif current_hit_rate > target_hit_rate + 10 and current_size < self.l1.max_size * 0.5:
            new_size = max(int(current_size * 0.9), min_size)
            if new_size < self.l1.max_size:
                self.l1.max_size = new_size
                logger.info(f"Auto-tuned cache size to {new_size} (hit rate: {current_hit_rate:.2f}%)")
    
    async def close(self):
        """Close cache connections properly and cleanup resources"""
        if self._promotion_tasks:
            try:
                await asyncio.gather(*self._promotion_tasks, return_exceptions=True)
            except Exception as e:
                logger.warning(f"Error waiting for promotion tasks: {e}")
            finally:
                self._promotion_tasks.clear()
        
        if self.l2:
            await self.l2.close()
        
        if self.l3:
            self.l3.close()


# Global cache instance
_cache_instance: Optional[EnhancedCache] = None


def get_cache() -> EnhancedCache:
    """Get or create cache instance"""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = EnhancedCache()
    return _cache_instance


async def close_cache():
    """Close cache instance"""
    global _cache_instance
    if _cache_instance:
        await _cache_instance.close()
        _cache_instance = None

