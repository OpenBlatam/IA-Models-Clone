"""
Smart Caching
Intelligent caching with TTL and size limits
"""

import time
from typing import Any, Dict, Optional, Callable
from collections import OrderedDict
import logging

logger = logging.getLogger(__name__)


class SmartCache:
    """
    Smart cache with TTL and size limits
    """
    
    def __init__(
        self,
        max_size: int = 1000,
        ttl_seconds: Optional[float] = None,
        eviction_policy: str = "lru"
    ):
        """
        Initialize smart cache
        
        Args:
            max_size: Maximum cache size
            ttl_seconds: Time to live in seconds (None for no expiration)
            eviction_policy: Eviction policy (lru, lfu, fifo)
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.eviction_policy = eviction_policy
        
        if eviction_policy == "lru":
            self.cache: OrderedDict = OrderedDict()
        else:
            self.cache: Dict[str, Dict[str, Any]] = {}
        
        self.access_times: Dict[str, float] = {}
        self.access_counts: Dict[str, int] = {}
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None
        """
        if key not in self.cache:
            return None
        
        # Check TTL
        if self.ttl_seconds:
            if time.time() - self.access_times.get(key, 0) > self.ttl_seconds:
                self.delete(key)
                return None
        
        # Update access
        self.access_times[key] = time.time()
        self.access_counts[key] = self.access_counts.get(key, 0) + 1
        
        # Update LRU order
        if self.eviction_policy == "lru" and isinstance(self.cache, OrderedDict):
            self.cache.move_to_end(key)
        
        return self.cache[key]["value"]
    
    def put(self, key: str, value: Any):
        """
        Put value in cache
        
        Args:
            key: Cache key
            value: Value to cache
        """
        # Evict if needed
        if len(self.cache) >= self.max_size and key not in self.cache:
            self._evict()
        
        # Store value
        if isinstance(self.cache, OrderedDict):
            self.cache[key] = {"value": value, "timestamp": time.time()}
            self.cache.move_to_end(key)
        else:
            self.cache[key] = {"value": value, "timestamp": time.time()}
        
        self.access_times[key] = time.time()
        self.access_counts[key] = self.access_counts.get(key, 0) + 1
    
    def delete(self, key: str):
        """Delete key from cache"""
        if key in self.cache:
            del self.cache[key]
        if key in self.access_times:
            del self.access_times[key]
        if key in self.access_counts:
            del self.access_counts[key]
    
    def _evict(self):
        """Evict item based on policy"""
        if not self.cache:
            return
        
        if self.eviction_policy == "lru" and isinstance(self.cache, OrderedDict):
            # Remove least recently used
            key = next(iter(self.cache))
            self.delete(key)
        elif self.eviction_policy == "lfu":
            # Remove least frequently used
            key = min(self.access_counts.items(), key=lambda x: x[1])[0]
            self.delete(key)
        elif self.eviction_policy == "fifo":
            # Remove first in
            key = next(iter(self.cache))
            self.delete(key)
    
    def clear(self):
        """Clear cache"""
        self.cache.clear()
        self.access_times.clear()
        self.access_counts.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hit_rate": self._calculate_hit_rate(),
            "eviction_policy": self.eviction_policy,
            "ttl_seconds": self.ttl_seconds
        }
    
    def _calculate_hit_rate(self) -> float:
        """Calculate hit rate (simplified)"""
        total_accesses = sum(self.access_counts.values())
        if total_accesses == 0:
            return 0.0
        # Simplified: assume hits = current size
        return len(self.cache) / max(total_accesses, 1)


def create_smart_cache(
    max_size: int = 1000,
    ttl_seconds: Optional[float] = None,
    eviction_policy: str = "lru"
) -> SmartCache:
    """Factory for smart cache"""
    return SmartCache(max_size, ttl_seconds, eviction_policy)








