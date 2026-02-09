"""
Caching Strategy for Color Grading AI
======================================

Advanced caching strategies and patterns.
"""

import logging
from typing import Dict, Any, Optional, Callable
from enum import Enum
from functools import wraps
import hashlib
import json

logger = logging.getLogger(__name__)


class CacheStrategy(Enum):
    """Cache strategies."""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    FIFO = "fifo"  # First In First Out
    TTL = "ttl"  # Time To Live
    ADAPTIVE = "adaptive"  # Adaptive based on usage


class CachingStrategy:
    """
    Advanced caching strategies.
    
    Features:
    - Multiple cache strategies
    - Cache warming
    - Cache invalidation patterns
    - Cache statistics
    """
    
    def __init__(self, strategy: CacheStrategy = CacheStrategy.LRU, max_size: int = 1000):
        """
        Initialize caching strategy.
        
        Args:
            strategy: Cache strategy to use
            max_size: Maximum cache size
        """
        self.strategy = strategy
        self.max_size = max_size
        self._cache: Dict[str, Any] = {}
        self._access_times: Dict[str, float] = {}
        self._access_counts: Dict[str, int] = {}
        self._insertion_order: list = []
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if key not in self._cache:
            return None
        
        # Update access tracking
        import time
        self._access_times[key] = time.time()
        self._access_counts[key] = self._access_counts.get(key, 0) + 1
        
        return self._cache[key]
    
    def set(self, key: str, value: Any):
        """Set value in cache."""
        # Evict if needed
        if len(self._cache) >= self.max_size:
            self._evict()
        
        self._cache[key] = value
        import time
        self._access_times[key] = time.time()
        self._access_counts[key] = 1
        self._insertion_order.append(key)
    
    def _evict(self):
        """Evict entry based on strategy."""
        if not self._cache:
            return
        
        if self.strategy == CacheStrategy.LRU:
            # Remove least recently used
            lru_key = min(self._access_times.items(), key=lambda x: x[1])[0]
            self._remove_key(lru_key)
        
        elif self.strategy == CacheStrategy.LFU:
            # Remove least frequently used
            lfu_key = min(self._access_counts.items(), key=lambda x: x[1])[0]
            self._remove_key(lfu_key)
        
        elif self.strategy == CacheStrategy.FIFO:
            # Remove first inserted
            if self._insertion_order:
                fifo_key = self._insertion_order.pop(0)
                self._remove_key(fifo_key)
        
        elif self.strategy == CacheStrategy.ADAPTIVE:
            # Adaptive: combine LRU and LFU
            scores = {}
            for key in self._cache.keys():
                recency = self._access_times.get(key, 0)
                frequency = self._access_counts.get(key, 0)
                scores[key] = frequency / (1 + recency)  # Lower score = evict
            evict_key = min(scores.items(), key=lambda x: x[1])[0]
            self._remove_key(evict_key)
    
    def _remove_key(self, key: str):
        """Remove key from cache and tracking."""
        if key in self._cache:
            del self._cache[key]
        if key in self._access_times:
            del self._access_times[key]
        if key in self._access_counts:
            del self._access_counts[key]
        if key in self._insertion_order:
            self._insertion_order.remove(key)
    
    def clear(self):
        """Clear cache."""
        self._cache.clear()
        self._access_times.clear()
        self._access_counts.clear()
        self._insertion_order.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "strategy": self.strategy.value,
            "hit_rate": self._calculate_hit_rate(),
        }
    
    def _calculate_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total_accesses = sum(self._access_counts.values())
        if total_accesses == 0:
            return 0.0
        return len(self._cache) / total_accesses if total_accesses > 0 else 0.0


def cache_result(
    cache: CachingStrategy,
    key_func: Optional[Callable] = None,
    ttl: Optional[float] = None
):
    """
    Decorator for caching function results.
    
    Args:
        cache: Caching strategy instance
        key_func: Optional function to generate cache key
        ttl: Optional time to live in seconds
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                key_data = json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True, default=str)
                cache_key = hashlib.md5(key_data.encode()).hexdigest()
            
            # Check cache
            cached = cache.get(cache_key)
            if cached is not None:
                return cached
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Store in cache
            cache.set(cache_key, result)
            
            return result
        
        return wrapper
    return decorator




