"""
Cache Manager

Manages caching for inspection results and model outputs.
"""

import logging
import hashlib
import json
from typing import Optional, Any, Callable
from datetime import datetime, timedelta
from functools import wraps

logger = logging.getLogger(__name__)


class CacheManager:
    """
    Cache manager for inspection results.
    
    Supports TTL-based expiration and size limits.
    """
    
    def __init__(
        self,
        max_size: int = 1000,
        default_ttl: int = 3600  # 1 hour
    ):
        """
        Initialize cache manager.
        
        Args:
            max_size: Maximum number of cached items
            default_ttl: Default TTL in seconds
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache = {}
        self._access_times = {}
    
    def _generate_key(self, *args, **kwargs) -> str:
        """
        Generate cache key from arguments.
        
        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments
        
        Returns:
            Cache key string
        """
        # Create hash from arguments
        key_data = {
            "args": args,
            "kwargs": sorted(kwargs.items())
        }
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()
    
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
        if datetime.utcnow() > entry['expires_at']:
            del self._cache[key]
            if key in self._access_times:
                del self._access_times[key]
            return None
        
        # Update access time
        self._access_times[key] = datetime.utcnow()
        
        return entry['value']
    
    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ):
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (uses default if None)
        """
        # Evict if at capacity
        if len(self._cache) >= self.max_size and key not in self._cache:
            self._evict_oldest()
        
        ttl = ttl or self.default_ttl
        expires_at = datetime.utcnow() + timedelta(seconds=ttl)
        
        self._cache[key] = {
            'value': value,
            'expires_at': expires_at,
            'created_at': datetime.utcnow(),
        }
        self._access_times[key] = datetime.utcnow()
    
    def _evict_oldest(self):
        """Evict oldest accessed item."""
        if not self._access_times:
            # If no access times, evict first item
            if self._cache:
                key = next(iter(self._cache))
                del self._cache[key]
            return
        
        # Find least recently accessed
        oldest_key = min(self._access_times.items(), key=lambda x: x[1])[0]
        del self._cache[oldest_key]
        del self._access_times[oldest_key]
    
    def clear(self):
        """Clear all cache."""
        self._cache.clear()
        self._access_times.clear()
    
    def get_stats(self) -> dict:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache stats
        """
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "default_ttl": self.default_ttl,
        }
    
    def cached(
        self,
        ttl: Optional[int] = None,
        key_func: Optional[Callable] = None
    ):
        """
        Decorator for caching function results.
        
        Args:
            ttl: Time to live in seconds
            key_func: Optional function to generate cache key
        
        Returns:
            Decorated function
        """
        def decorator(func: Callable):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Generate cache key
                if key_func:
                    cache_key = key_func(*args, **kwargs)
                else:
                    cache_key = self._generate_key(*args, **kwargs)
                
                # Try to get from cache
                cached_value = self.get(cache_key)
                if cached_value is not None:
                    logger.debug(f"Cache hit for {func.__name__}")
                    return cached_value
                
                # Execute function
                logger.debug(f"Cache miss for {func.__name__}")
                result = func(*args, **kwargs)
                
                # Store in cache
                self.set(cache_key, result, ttl)
                
                return result
            
            return wrapper
        return decorator


# Global cache manager instance
_cache_manager: Optional[CacheManager] = None


def get_cache_manager() -> CacheManager:
    """
    Get global cache manager instance.
    
    Returns:
        CacheManager instance
    """
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager



