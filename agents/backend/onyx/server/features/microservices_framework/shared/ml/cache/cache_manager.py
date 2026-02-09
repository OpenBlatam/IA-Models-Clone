"""
Cache Manager
Advanced caching with TTL, LRU, and distributed cache support.
"""

import time
from typing import Any, Optional, Dict, Callable
from collections import OrderedDict
import hashlib
import json
import logging

logger = logging.getLogger(__name__)


class CacheEntry:
    """Cache entry with metadata."""
    
    def __init__(self, value: Any, ttl: Optional[float] = None):
        self.value = value
        self.created_at = time.time()
        self.ttl = ttl
        self.access_count = 0
        self.last_accessed = self.created_at
    
    def is_expired(self) -> bool:
        """Check if entry is expired."""
        if self.ttl is None:
            return False
        return (time.time() - self.created_at) > self.ttl
    
    def access(self):
        """Record access."""
        self.access_count += 1
        self.last_accessed = time.time()


class LRUCache:
    """LRU cache implementation."""
    
    def __init__(self, max_size: int = 100, default_ttl: Optional[float] = None):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if key not in self.cache:
            return None
        
        entry = self.cache[key]
        
        if entry.is_expired():
            del self.cache[key]
            return None
        
        entry.access()
        # Move to end (most recently used)
        self.cache.move_to_end(key)
        return entry.value
    
    def set(self, key: str, value: Any, ttl: Optional[float] = None):
        """Set value in cache."""
        ttl = ttl or self.default_ttl
        entry = CacheEntry(value, ttl)
        
        if key in self.cache:
            self.cache.move_to_end(key)
        
        self.cache[key] = entry
        
        # Evict if over size limit
        if len(self.cache) > self.max_size:
            # Remove least recently used
            self.cache.popitem(last=False)
    
    def delete(self, key: str) -> bool:
        """Delete key from cache."""
        if key in self.cache:
            del self.cache[key]
            return True
        return False
    
    def clear(self):
        """Clear all cache."""
        self.cache.clear()
    
    def size(self) -> int:
        """Get cache size."""
        return len(self.cache)
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_accesses = sum(entry.access_count for entry in self.cache.values())
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "total_accesses": total_accesses,
            "entries": [
                {
                    "key": key,
                    "access_count": entry.access_count,
                    "age": time.time() - entry.created_at,
                }
                for key, entry in self.cache.items()
            ],
        }


class CacheManager:
    """Advanced cache manager with multiple strategies."""
    
    def __init__(
        self,
        cache_type: str = "lru",
        max_size: int = 100,
        default_ttl: Optional[float] = None,
    ):
        self.cache_type = cache_type
        self.max_size = max_size
        self.default_ttl = default_ttl
        
        if cache_type == "lru":
            self.cache = LRUCache(max_size=max_size, default_ttl=default_ttl)
        else:
            raise ValueError(f"Unknown cache type: {cache_type}")
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        return self.cache.get(key)
    
    def set(self, key: str, value: Any, ttl: Optional[float] = None):
        """Set value in cache."""
        self.cache.set(key, value, ttl)
    
    def delete(self, key: str) -> bool:
        """Delete key from cache."""
        return self.cache.delete(key)
    
    def clear(self):
        """Clear cache."""
        self.cache.clear()
    
    def get_or_set(
        self,
        key: str,
        factory: Callable[[], Any],
        ttl: Optional[float] = None,
    ) -> Any:
        """Get value or set using factory function."""
        value = self.get(key)
        if value is None:
            value = factory()
            self.set(key, value, ttl)
        return value
    
    def make_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments."""
        key_data = {
            "args": args,
            "kwargs": kwargs,
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def cached(self, ttl: Optional[float] = None):
        """Decorator for caching function results."""
        def decorator(func: Callable) -> Callable:
            def wrapper(*args, **kwargs):
                key = self.make_key(func.__name__, *args, **kwargs)
                return self.get_or_set(key, lambda: func(*args, **kwargs), ttl)
            return wrapper
        return decorator



