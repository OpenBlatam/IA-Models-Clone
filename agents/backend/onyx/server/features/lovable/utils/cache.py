"""
Cache utilities for in-memory caching.
"""

from typing import Optional, Any, Dict
from collections import OrderedDict
from datetime import datetime, timedelta
import time
import logging

logger = logging.getLogger(__name__)

# Global cache instance
_cache_instance: Optional['LRUCache'] = None


class LRUCache:
    """LRU cache with TTL support."""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        """
        Initialize LRU cache.
        
        Args:
            max_size: Maximum number of items
            default_ttl: Default TTL in seconds
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: OrderedDict = OrderedDict()
        self.expiry_times: Dict[str, float] = {}
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found or expired
        """
        if key not in self.cache:
            return None
        
        # Check expiry
        if key in self.expiry_times:
            if time.time() > self.expiry_times[key]:
                # Expired, remove it
                del self.cache[key]
                del self.expiry_times[key]
                return None
        
        # Move to end (most recently used)
        value = self.cache.pop(key)
        self.cache[key] = value
        
        return value
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (uses default if None)
        """
        # Remove if exists
        if key in self.cache:
            del self.cache[key]
        
        # Check size limit
        if len(self.cache) >= self.max_size:
            # Remove least recently used
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
            if oldest_key in self.expiry_times:
                del self.expiry_times[oldest_key]
        
        # Add to cache
        self.cache[key] = value
        
        # Set expiry
        ttl = ttl or self.default_ttl
        self.expiry_times[key] = time.time() + ttl
    
    def delete(self, key: str) -> bool:
        """
        Delete key from cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if deleted, False if not found
        """
        if key in self.cache:
            del self.cache[key]
            if key in self.expiry_times:
                del self.expiry_times[key]
            return True
        return False
    
    def clear(self) -> None:
        """Clear all cache entries."""
        self.cache.clear()
        self.expiry_times.clear()
    
    def size(self) -> int:
        """
        Get current cache size.
        
        Returns:
            Number of items in cache
        """
        return len(self.cache)
    
    def cleanup_expired(self) -> int:
        """
        Clean up expired entries.
        
        Returns:
            Number of entries removed
        """
        current_time = time.time()
        expired_keys = [
            key for key, expiry in self.expiry_times.items()
            if current_time > expiry
        ]
        
        for key in expired_keys:
            if key in self.cache:
                del self.cache[key]
            del self.expiry_times[key]
        
        return len(expired_keys)


def get_cache(max_size: int = 1000, default_ttl: int = 3600) -> LRUCache:
    """
    Get or create global cache instance.
    
    Args:
        max_size: Maximum number of items
        default_ttl: Default TTL in seconds
        
    Returns:
        Cache instance
    """
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = LRUCache(max_size=max_size, default_ttl=default_ttl)
        logger.info(f"Cache initialized: max_size={max_size}, default_ttl={default_ttl}")
    return _cache_instance




