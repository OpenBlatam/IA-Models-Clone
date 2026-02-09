"""
Caching utilities for professional documents module.

Simple in-memory cache for frequently accessed data.
"""

from functools import lru_cache
from typing import TypeVar, Callable, Optional, Any
import time

T = TypeVar('T')


class TimedCache:
    """Simple time-based cache with TTL."""
    
    def __init__(self, ttl_seconds: float = 300.0):
        """
        Initialize timed cache.
        
        Args:
            ttl_seconds: Time to live in seconds (default: 5 minutes)
        """
        self._cache: dict[str, tuple[Any, float]] = {}
        self.ttl_seconds = ttl_seconds
    
    def get(self, key: str) -> Optional[T]:
        """
        Get value from cache if not expired.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if expired/not found
        """
        if key not in self._cache:
            return None
        
        value, timestamp = self._cache[key]
        
        if time.time() - timestamp > self.ttl_seconds:
            del self._cache[key]
            return None
        
        return value
    
    def set(self, key: str, value: T) -> None:
        """
        Set value in cache with current timestamp.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        self._cache[key] = (value, time.time())
    
    def clear(self) -> None:
        """Clear all cached values."""
        self._cache.clear()
    
    def invalidate(self, key: str) -> None:
        """
        Invalidate a specific cache key.
        
        Args:
            key: Cache key to invalidate
        """
        self._cache.pop(key, None)
    
    def size(self) -> int:
        """Get number of cached items."""
        return len(self._cache)


# Global cache instance for templates
template_cache = TimedCache(ttl_seconds=600.0)  # 10 minutes


def cached_template(key_func: Callable[..., str]):
    """
    Decorator to cache template lookups.
    
    Args:
        key_func: Function to generate cache key from arguments
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        def wrapper(*args, **kwargs) -> T:
            cache_key = key_func(*args, **kwargs)
            cached_value = template_cache.get(cache_key)
            
            if cached_value is not None:
                return cached_value
            
            result = func(*args, **kwargs)
            template_cache.set(cache_key, result)
            return result
        
        return wrapper
    return decorator

