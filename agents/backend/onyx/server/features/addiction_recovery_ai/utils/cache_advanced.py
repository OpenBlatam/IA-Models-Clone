"""
Advanced caching utilities
Advanced caching functions with TTL and invalidation
"""

from typing import Any, Optional, Callable, Dict
from datetime import datetime, timedelta
from functools import wraps
import hashlib
import json


class CacheEntry:
    """
    Cache entry with TTL
    """
    
    def __init__(self, value: Any, ttl: Optional[float] = None):
        """
        Initialize cache entry
        
        Args:
            value: Cached value
            ttl: Time to live in seconds
        """
        self.value = value
        self.created_at = datetime.now()
        self.ttl = ttl
    
    def is_expired(self) -> bool:
        """
        Check if entry is expired
        
        Returns:
            True if expired
        """
        if self.ttl is None:
            return False
        
        elapsed = (datetime.now() - self.created_at).total_seconds()
        return elapsed > self.ttl
    
    def get_remaining_ttl(self) -> Optional[float]:
        """
        Get remaining TTL
        
        Returns:
            Remaining TTL in seconds or None
        """
        if self.ttl is None:
            return None
        
        elapsed = (datetime.now() - self.created_at).total_seconds()
        remaining = self.ttl - elapsed
        return max(0, remaining)


class AdvancedCache:
    """
    Advanced cache with TTL support
    """
    
    def __init__(self, default_ttl: Optional[float] = None):
        """
        Initialize cache
        
        Args:
            default_ttl: Default TTL in seconds
        """
        self._cache: Dict[str, CacheEntry] = {}
        self.default_ttl = default_ttl
    
    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """
        Set cache value
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
        """
        ttl = ttl or self.default_ttl
        self._cache[key] = CacheEntry(value, ttl)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get cache value
        
        Args:
            key: Cache key
            default: Default value if not found or expired
        
        Returns:
            Cached value or default
        """
        if key not in self._cache:
            return default
        
        entry = self._cache[key]
        
        if entry.is_expired():
            del self._cache[key]
            return default
        
        return entry.value
    
    def has(self, key: str) -> bool:
        """
        Check if key exists and is not expired
        
        Args:
            key: Cache key
        
        Returns:
            True if key exists and not expired
        """
        if key not in self._cache:
            return False
        
        if self._cache[key].is_expired():
            del self._cache[key]
            return False
        
        return True
    
    def delete(self, key: str) -> bool:
        """
        Delete cache entry
        
        Args:
            key: Cache key
        
        Returns:
            True if deleted
        """
        if key in self._cache:
            del self._cache[key]
            return True
        return False
    
    def clear(self) -> None:
        """
        Clear all cache entries
        """
        self._cache.clear()
    
    def cleanup_expired(self) -> int:
        """
        Remove expired entries
        
        Returns:
            Number of entries removed
        """
        expired_keys = [
            key for key, entry in self._cache.items()
            if entry.is_expired()
        ]
        
        for key in expired_keys:
            del self._cache[key]
        
        return len(expired_keys)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics
        
        Returns:
            Cache statistics
        """
        self.cleanup_expired()
        
        return {
            "size": len(self._cache),
            "default_ttl": self.default_ttl,
        }


def cache_key(*args, **kwargs) -> str:
    """
    Generate cache key from arguments
    
    Args:
        *args: Positional arguments
        **kwargs: Keyword arguments
    
    Returns:
        Cache key string
    """
    key_data = {
        "args": args,
        "kwargs": sorted(kwargs.items())
    }
    
    key_str = json.dumps(key_data, sort_keys=True, default=str)
    return hashlib.md5(key_str.encode()).hexdigest()


def cached(ttl: Optional[float] = None, cache_instance: Optional[AdvancedCache] = None):
    """
    Decorator for caching function results
    
    Args:
        ttl: Time to live in seconds
        cache_instance: Optional cache instance
    
    Returns:
        Decorator function
    """
    if cache_instance is None:
        cache_instance = AdvancedCache(default_ttl=ttl)
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            key = f"{func.__name__}:{cache_key(*args, **kwargs)}"
            
            if cache_instance.has(key):
                return cache_instance.get(key)
            
            result = func(*args, **kwargs)
            cache_instance.set(key, result, ttl=ttl)
            
            return result
        
        return wrapper
    return decorator


def invalidate_cache(cache_instance: AdvancedCache, pattern: Optional[str] = None):
    """
    Invalidate cache entries
    
    Args:
        cache_instance: Cache instance
        pattern: Optional pattern to match keys
    """
    if pattern is None:
        cache_instance.clear()
        return
    
    keys_to_delete = [
        key for key in cache_instance._cache.keys()
        if pattern in key
    ]
    
    for key in keys_to_delete:
        cache_instance.delete(key)

