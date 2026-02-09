"""
API Cache
========

Simple in-memory cache for API responses.
"""

from typing import Any, Optional, Dict
from datetime import datetime, timedelta
import threading
import hashlib
import json


class CacheEntry:
    """Cache entry with expiration."""
    
    def __init__(self, value: Any, ttl: int = 300):
        """
        Initialize cache entry.
        
        Args:
            value: Cached value
            ttl: Time to live in seconds (default: 5 minutes)
        """
        self.value = value
        self.created_at = datetime.now()
        self.ttl = ttl
    
    def is_expired(self) -> bool:
        """Check if cache entry is expired."""
        return datetime.now() - self.created_at > timedelta(seconds=self.ttl)
    
    def get_age(self) -> float:
        """Get age of cache entry in seconds."""
        return (datetime.now() - self.created_at).total_seconds()


class SimpleCache:
    """Simple in-memory cache with TTL."""
    
    def __init__(self, default_ttl: int = 300):
        """
        Initialize cache.
        
        Args:
            default_ttl: Default time to live in seconds
        """
        self._cache: Dict[str, CacheEntry] = {}
        self._lock = threading.Lock()
        self._default_ttl = default_ttl
    
    def _make_key(self, *args, **kwargs) -> str:
        """
        Create cache key from arguments.
        
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
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return None
            
            if entry.is_expired():
                del self._cache[key]
                return None
            
            return entry.value
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (uses default if None)
        """
        with self._lock:
            ttl = ttl or self._default_ttl
            self._cache[key] = CacheEntry(value, ttl)
    
    def delete(self, key: str):
        """
        Delete value from cache.
        
        Args:
            key: Cache key
        """
        with self._lock:
            self._cache.pop(key, None)
    
    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
    
    def cleanup_expired(self):
        """Remove expired entries from cache."""
        with self._lock:
            expired_keys = [
                key for key, entry in self._cache.items()
                if entry.is_expired()
            ]
            for key in expired_keys:
                del self._cache[key]
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        with self._lock:
            self.cleanup_expired()
            return {
                "size": len(self._cache),
                "default_ttl": self._default_ttl,
                "entries": [
                    {
                        "key": key,
                        "age": entry.get_age(),
                        "ttl": entry.ttl
                    }
                    for key, entry in self._cache.items()
                ]
            }


cache = SimpleCache(default_ttl=300)

