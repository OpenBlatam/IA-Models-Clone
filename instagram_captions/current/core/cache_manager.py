"""
Cache Manager for Instagram Captions API v10.0

Optimized caching functionality.
"""

import time
from typing import Any, Optional, Dict
from collections import OrderedDict

class CacheManager:
    """Simple and efficient cache manager."""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 300):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: OrderedDict = OrderedDict()
        self.timestamps: Dict[str, float] = {}
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if key not in self.cache:
            return None
        
        # Check TTL
        if time.time() - self.timestamps[key] > self.default_ttl:
            self.delete(key)
            return None
        
        # Move to end (LRU)
        self.cache.move_to_end(key)
        return self.cache[key]
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache."""
        if ttl is None:
            ttl = self.default_ttl
        
        # Remove if exists
        if key in self.cache:
            self.delete(key)
        
        # Check size limit
        if len(self.cache) >= self.max_size:
            # Remove oldest item
            oldest_key = next(iter(self.cache))
            self.delete(oldest_key)
        
        # Add new item
        self.cache[key] = value
        self.timestamps[key] = time.time()
    
    def delete(self, key: str) -> None:
        """Delete key from cache."""
        if key in self.cache:
            del self.cache[key]
            del self.timestamps[key]
    
    def clear(self) -> None:
        """Clear all cache."""
        self.cache.clear()
        self.timestamps.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "usage_percent": (len(self.cache) / self.max_size) * 100
        }






