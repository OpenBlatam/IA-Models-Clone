from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import asyncio
import time
from typing import Any, Optional, Dict, List, Tuple
    from .config_v5 import config
    from .metrics_v5 import metrics
    from config_v5 import config
    from metrics_v5 import metrics
from typing import Any, List, Dict, Optional
import logging
"""
Instagram Captions API v5.0 - Cache Module

Ultra-fast caching system with LRU eviction and intelligent cleanup.
"""

try:
except ImportError:


class UltraFastCache:
    """Ultra-fast LRU cache with automatic cleanup and performance optimization."""
    
    def __init__(self, max_size: int = None, ttl: int = None):
        
    """__init__ function."""
self.max_size = max_size or config.CACHE_MAX_SIZE
        self.ttl = ttl or config.CACHE_TTL
        
        # Storage
        self._cache: Dict[str, Any] = {}
        self._access_times: Dict[str, float] = {}
        self._access_counts: Dict[str, int] = {}
        self._creation_times: Dict[str, float] = {}
        
        # Thread safety
        self._lock = asyncio.Lock()
        
        # Statistics
        self._cleanup_count = 0
        self._eviction_count = 0
    
    async def get(self, key: str) -> Optional[Any]:
        """Get item from cache with LRU tracking."""
        async with self._lock:
            current_time = time.time()
            
            # Check if key exists
            if key not in self._cache:
                metrics.record_cache_miss()
                return None
            
            # Check if expired
            if self._is_expired(key, current_time):
                await self._remove_key(key)
                metrics.record_cache_miss()
                return None
            
            # Update access tracking
            self._access_times[key] = current_time
            self._access_counts[key] = self._access_counts.get(key, 0) + 1
            
            metrics.record_cache_hit()
            return self._cache[key]
    
    async def set(self, key: str, value: Any) -> None:
        """Set item in cache with automatic cleanup."""
        async with self._lock:
            current_time = time.time()
            
            # Store the item
            self._cache[key] = value
            self._access_times[key] = current_time
            self._creation_times[key] = current_time
            self._access_counts[key] = 1
            
            # Trigger cleanup if needed
            if len(self._cache) > self.max_size:
                await self._intelligent_cleanup()
    
    async def delete(self, key: str) -> bool:
        """Delete specific key from cache."""
        async with self._lock:
            if key in self._cache:
                await self._remove_key(key)
                return True
            return False
    
    async def clear(self) -> int:
        """Clear all cache entries."""
        async with self._lock:
            size = len(self._cache)
            
            self._cache.clear()
            self._access_times.clear()
            self._access_counts.clear()
            self._creation_times.clear()
            
            return size
    
    async def cleanup_expired(self) -> int:
        """Manually cleanup expired entries."""
        async with self._lock:
            return await self._cleanup_expired_items()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "ttl_seconds": self.ttl,
            "cleanup_count": self._cleanup_count,
            "eviction_count": self._eviction_count,
            "most_accessed": self._get_most_accessed_items(5),
            "cache_efficiency": self._calculate_efficiency()
        }
    
    async def get_detailed_stats(self) -> Dict[str, Any]:
        """Get detailed cache statistics."""
        async with self._lock:
            current_time = time.time()
            
            # Calculate age distribution
            ages = [current_time - create_time for create_time in self._creation_times.values()]
            
            return {
                **self.get_stats(),
                "avg_age_seconds": round(sum(ages) / len(ages), 2) if ages else 0,
                "oldest_item_age": round(max(ages), 2) if ages else 0,
                "newest_item_age": round(min(ages), 2) if ages else 0,
                "expired_items": await self._count_expired_items()
            }
    
    def _is_expired(self, key: str, current_time: float) -> bool:
        """Check if a cache item is expired."""
        creation_time = self._creation_times.get(key, 0)
        return current_time - creation_time > self.ttl
    
    async def _remove_key(self, key: str) -> None:
        """Remove a key and all its tracking data."""
        self._cache.pop(key, None)
        self._access_times.pop(key, None)
        self._access_counts.pop(key, None)
        self._creation_times.pop(key, None)
    
    async def _intelligent_cleanup(self) -> None:
        """Intelligent cleanup using multiple strategies."""
        self._cleanup_count += 1
        
        # First, remove expired items
        expired_removed = await self._cleanup_expired_items()
        
        # If still over capacity, use LRU eviction
        if len(self._cache) > self.max_size:
            lru_removed = await self._lru_eviction()
            self._eviction_count += lru_removed
    
    async def _cleanup_expired_items(self) -> int:
        """Remove all expired items."""
        current_time = time.time()
        expired_keys = [
            key for key in self._cache.keys()
            if self._is_expired(key, current_time)
        ]
        
        for key in expired_keys:
            await self._remove_key(key)
        
        return len(expired_keys)
    
    async def _lru_eviction(self) -> int:
        """Remove least recently used items."""
        if len(self._cache) <= self.max_size:
            return 0
        
        # Calculate removal count (remove 10% of excess)
        excess = len(self._cache) - self.max_size
        removal_count = max(1, excess + (self.max_size // 10))
        
        # Sort by LRU criteria (access time and count)
        items_by_lru = sorted(
            self._access_times.items(),
            key=lambda x: (self._access_counts.get(x[0], 0), x[1])
        )
        
        # Remove least recently/frequently used items
        keys_to_remove = [key for key, _ in items_by_lru[:removal_count]]
        
        for key in keys_to_remove:
            await self._remove_key(key)
        
        return len(keys_to_remove)
    
    def _get_most_accessed_items(self, count: int) -> List[Tuple[str, int]]:
        """Get most accessed items."""
        return sorted(
            self._access_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:count]
    
    def _calculate_efficiency(self) -> float:
        """Calculate cache efficiency based on access patterns."""
        if not self._access_counts:
            return 0.0
        
        # Calculate hit distribution
        total_accesses = sum(self._access_counts.values())
        unique_items = len(self._access_counts)
        
        # Efficiency = average access per item (higher is better)
        return round(total_accesses / unique_items, 2)
    
    async def _count_expired_items(self) -> int:
        """Count expired items without removing them."""
        current_time = time.time()
        return sum(
            1 for key in self._cache.keys()
            if self._is_expired(key, current_time)
        )


class CacheManager:
    """High-level cache management with multiple cache instances."""
    
    def __init__(self) -> Any:
        # Different caches for different purposes
        self.caption_cache = UltraFastCache(
            max_size=config.CACHE_MAX_SIZE,
            ttl=config.CACHE_TTL
        )
        self.batch_cache = UltraFastCache(
            max_size=config.CACHE_MAX_SIZE // 4,  # Smaller for batch results
            ttl=config.CACHE_TTL // 2  # Shorter TTL for batch results
        )
        self.health_cache = UltraFastCache(
            max_size=100,  # Small cache for health checks
            ttl=60  # 1 minute TTL
        )
    
    async def get_caption(self, key: str) -> Optional[Any]:
        """Get from caption cache."""
        return await self.caption_cache.get(key)
    
    async def set_caption(self, key: str, value: Any) -> None:
        """Set in caption cache."""
        await self.caption_cache.set(key, value)
    
    async def get_batch(self, key: str) -> Optional[Any]:
        """Get from batch cache."""
        return await self.batch_cache.get(key)
    
    async def set_batch(self, key: str, value: Any) -> None:
        """Set in batch cache."""
        await self.batch_cache.set(key, value)
    
    async def get_health(self, key: str) -> Optional[Any]:
        """Get from health cache."""
        return await self.health_cache.get(key)
    
    async def set_health(self, key: str, value: Any) -> None:
        """Set in health cache."""
        await self.health_cache.set(key, value)
    
    async def clear_all(self) -> Dict[str, int]:
        """Clear all caches."""
        return {
            "caption_cache": await self.caption_cache.clear(),
            "batch_cache": await self.batch_cache.clear(),
            "health_cache": await self.health_cache.clear()
        }
    
    async def cleanup_all(self) -> Dict[str, int]:
        """Cleanup expired items in all caches."""
        return {
            "caption_cache": await self.caption_cache.cleanup_expired(),
            "batch_cache": await self.batch_cache.cleanup_expired(),
            "health_cache": await self.health_cache.cleanup_expired()
        }
    
    async def get_all_stats(self) -> Dict[str, Any]:
        """Get statistics for all caches."""
        return {
            "caption_cache": await self.caption_cache.get_detailed_stats(),
            "batch_cache": await self.batch_cache.get_detailed_stats(),
            "health_cache": await self.health_cache.get_detailed_stats()
        }


# Global cache manager instance
cache_manager = CacheManager()


# Export public interface
__all__ = [
    'UltraFastCache',
    'CacheManager',
    'cache_manager'
] 