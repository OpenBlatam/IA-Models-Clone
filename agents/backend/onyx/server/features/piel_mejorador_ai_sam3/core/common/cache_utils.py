"""
Cache Utilities for Piel Mejorador AI SAM3
==========================================

Unified cache utilities and patterns.
"""

import hashlib
import json
import logging
from typing import Any, Optional, Callable, Dict, TypeVar, Union
from datetime import datetime, timedelta
from functools import wraps
from collections import OrderedDict

logger = logging.getLogger(__name__)

T = TypeVar('T')


class CacheUtils:
    """Unified cache utilities."""
    
    @staticmethod
    def generate_key(*args, **kwargs) -> str:
        """
        Generate cache key from arguments.
        
        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Cache key (hash)
        """
        key_data = {
            "args": args,
            "kwargs": sorted(kwargs.items())
        }
        key_string = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.sha256(key_string.encode()).hexdigest()
    
    @staticmethod
    def generate_key_from_dict(data: Dict[str, Any]) -> str:
        """
        Generate cache key from dictionary.
        
        Args:
            data: Dictionary data
            
        Returns:
            Cache key (hash)
        """
        key_string = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(key_string.encode()).hexdigest()
    
    @staticmethod
    def is_expired(
        created_at: datetime,
        ttl: timedelta,
        current_time: Optional[datetime] = None
    ) -> bool:
        """
        Check if cache entry is expired.
        
        Args:
            created_at: Creation time
            ttl: Time to live
            current_time: Optional current time (defaults to now)
            
        Returns:
            True if expired
        """
        if current_time is None:
            current_time = datetime.now()
        return current_time - created_at > ttl
    
    @staticmethod
    def calculate_ttl_seconds(ttl: Union[timedelta, int, float]) -> int:
        """
        Convert TTL to seconds.
        
        Args:
            ttl: TTL as timedelta, int (seconds), or float (seconds)
            
        Returns:
            TTL in seconds
        """
        if isinstance(ttl, timedelta):
            return int(ttl.total_seconds())
        return int(ttl)


class SimpleCache:
    """Simple in-memory cache with TTL and size limits."""
    
    def __init__(
        self,
        max_size: Optional[int] = None,
        default_ttl: Optional[timedelta] = None
    ):
        """
        Initialize simple cache.
        
        Args:
            max_size: Maximum cache size (None = unlimited)
            default_ttl: Default TTL for entries
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None
        """
        if key not in self._cache:
            return None
        
        entry = self._cache[key]
        
        # Check expiration
        if entry.get("ttl"):
            if CacheUtils.is_expired(
                entry["created_at"],
                entry["ttl"]
            ):
                del self._cache[key]
                return None
        
        # Move to end (LRU)
        self._cache.move_to_end(key)
        return entry["value"]
    
    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[timedelta] = None
    ):
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Optional TTL (uses default if not provided)
        """
        ttl = ttl or self.default_ttl
        
        entry = {
            "value": value,
            "created_at": datetime.now(),
            "ttl": ttl
        }
        
        # Remove if exists
        if key in self._cache:
            del self._cache[key]
        
        # Add new entry
        self._cache[key] = entry
        
        # Enforce max size
        if self.max_size and len(self._cache) > self.max_size:
            self._cache.popitem(last=False)  # Remove oldest
    
    def delete(self, key: str) -> bool:
        """
        Delete key from cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if deleted
        """
        if key in self._cache:
            del self._cache[key]
            return True
        return False
    
    def clear(self):
        """Clear all cache entries."""
        self._cache.clear()
    
    def size(self) -> int:
        """
        Get cache size.
        
        Returns:
            Number of entries
        """
        return len(self._cache)
    
    def cleanup_expired(self) -> int:
        """
        Remove expired entries.
        
        Returns:
            Number of entries removed
        """
        removed = 0
        keys_to_remove = []
        
        for key, entry in self._cache.items():
            if entry.get("ttl"):
                if CacheUtils.is_expired(entry["created_at"], entry["ttl"]):
                    keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self._cache[key]
            removed += 1
        
        return removed


def cached(
    ttl: Optional[timedelta] = None,
    max_size: Optional[int] = None,
    key_func: Optional[Callable] = None
):
    """
    Decorator for caching function results.
    
    Args:
        ttl: Time to live
        max_size: Maximum cache size
        key_func: Optional key function
        
    Returns:
        Decorator function
    """
    cache = SimpleCache(max_size=max_size, default_ttl=ttl)
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            # Generate key
            if key_func:
                key = key_func(*args, **kwargs)
            else:
                key = CacheUtils.generate_key(*args, **kwargs)
            
            # Check cache
            cached_value = cache.get(key)
            if cached_value is not None:
                return cached_value
            
            # Call function
            result = func(*args, **kwargs)
            
            # Cache result
            cache.set(key, result, ttl=ttl)
            
            return result
        
        # Add cache management methods
        wrapper.cache = cache
        wrapper.cache_clear = cache.clear
        wrapper.cache_size = cache.size
        wrapper.cache_cleanup = cache.cleanup_expired
        
        return wrapper
    
    return decorator


# Convenience functions
def generate_key(*args, **kwargs) -> str:
    """Generate cache key."""
    return CacheUtils.generate_key(*args, **kwargs)


def is_expired(created_at: datetime, ttl: timedelta, **kwargs) -> bool:
    """Check if expired."""
    return CacheUtils.is_expired(created_at, ttl, **kwargs)




