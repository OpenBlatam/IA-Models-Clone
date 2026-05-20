"""
Advanced Caching System
LRU cache with TTL, size limits, and statistics
"""

from typing import Dict, Any, Optional, Callable
import logging
import time
from collections import OrderedDict
from threading import Lock

logger = logging.getLogger(__name__)


class LRUCache:
    """
    LRU Cache with TTL and size limits
    """
    
    def __init__(
        self,
        capacity: int = 1000,
        ttl: Optional[float] = None,  # Time to live in seconds
        eviction_policy: str = "lru"  # "lru", "fifo", "lfu"
    ):
        self.capacity = capacity
        self.ttl = ttl
        self.eviction_policy = eviction_policy
        
        self.cache: OrderedDict = OrderedDict()
        self.access_times: Dict[str, float] = {}
        self.access_counts: Dict[str, int] = {}
        self.lock = Lock()
        
        self.hits = 0
        self.misses = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        with self.lock:
            if key not in self.cache:
                self.misses += 1
                return None
            
            # Check TTL
            if self.ttl:
                if time.time() - self.access_times.get(key, 0) > self.ttl:
                    del self.cache[key]
                    if key in self.access_times:
                        del self.access_times[key]
                    if key in self.access_counts:
                        del self.access_counts[key]
                    self.misses += 1
                    return None
            
            # Move to end (most recently used)
            value = self.cache.pop(key)
            self.cache[key] = value
            
            # Update access info
            self.access_times[key] = time.time()
            self.access_counts[key] = self.access_counts.get(key, 0) + 1
            
            self.hits += 1
            return value
    
    def set(self, key: str, value: Any):
        """Set value in cache"""
        with self.lock:
            # Remove if exists
            if key in self.cache:
                del self.cache[key]
            
            # Evict if at capacity
            if len(self.cache) >= self.capacity:
                self._evict()
            
            # Add to cache
            self.cache[key] = value
            self.access_times[key] = time.time()
            self.access_counts[key] = 1
    
    def _evict(self):
        """Evict item based on policy"""
        if self.eviction_policy == "lru":
            # Remove least recently used (first item)
            self.cache.popitem(last=False)
        elif self.eviction_policy == "fifo":
            # Remove first item (FIFO)
            self.cache.popitem(last=False)
        elif self.eviction_policy == "lfu":
            # Remove least frequently used
            if self.access_counts:
                lfu_key = min(self.access_counts, key=self.access_counts.get)
                if lfu_key in self.cache:
                    del self.cache[lfu_key]
                    del self.access_times[lfu_key]
                    del self.access_counts[lfu_key]
    
    def clear(self):
        """Clear cache"""
        with self.lock:
            self.cache.clear()
            self.access_times.clear()
            self.access_counts.clear()
            self.hits = 0
            self.misses = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        
        return {
            "size": len(self.cache),
            "capacity": self.capacity,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "eviction_policy": self.eviction_policy,
            "ttl": self.ttl
        }


class CacheManager:
    """
    Manager for multiple caches
    """
    
    def __init__(self):
        self.caches: Dict[str, LRUCache] = {}
    
    def create_cache(
        self,
        name: str,
        capacity: int = 1000,
        ttl: Optional[float] = None,
        eviction_policy: str = "lru"
    ) -> LRUCache:
        """Create a new cache"""
        cache = LRUCache(capacity=capacity, ttl=ttl, eviction_policy=eviction_policy)
        self.caches[name] = cache
        return cache
    
    def get_cache(self, name: str) -> Optional[LRUCache]:
        """Get cache by name"""
        return self.caches.get(name)
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all caches"""
        return {
            name: cache.get_stats()
            for name, cache in self.caches.items()
        }
    
    def clear_all(self):
        """Clear all caches"""
        for cache in self.caches.values():
            cache.clear()

