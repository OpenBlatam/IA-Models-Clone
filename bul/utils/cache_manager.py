"""
Cache Manager for BUL System
============================

Provides efficient caching for document generation and API responses.
"""

import asyncio
import hashlib
import json
import logging
from typing import Any, Dict, Optional, Union
from datetime import datetime, timedelta
import time

logger = logging.getLogger(__name__)

class CacheManager:
    """
    Simple in-memory cache manager with TTL support
    """
    
    def __init__(self, default_ttl: int = 3600, max_size: int = 1000):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.default_ttl = default_ttl
        self.max_size = max_size
        self.access_times: Dict[str, float] = {}
        
    def _generate_key(self, data: Union[str, Dict[str, Any]]) -> str:
        """Generate a cache key from data"""
        if isinstance(data, dict):
            data_str = json.dumps(data, sort_keys=True)
        else:
            data_str = str(data)
        
        return hashlib.md5(data_str.encode()).hexdigest()
    
    def _is_expired(self, key: str) -> bool:
        """Check if a cache entry is expired"""
        if key not in self.cache:
            return True
        
        entry = self.cache[key]
        if 'expires_at' not in entry:
            return True
        
        return datetime.now() > entry['expires_at']
    
    def _cleanup_expired(self):
        """Remove expired entries"""
        expired_keys = [key for key in self.cache.keys() if self._is_expired(key)]
        for key in expired_keys:
            del self.cache[key]
            if key in self.access_times:
                del self.access_times[key]
    
    def _evict_lru(self):
        """Evict least recently used entry if cache is full"""
        if len(self.cache) >= self.max_size:
            # Find least recently used key
            lru_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
            del self.cache[lru_key]
            del self.access_times[lru_key]
    
    async def get(self, key: Union[str, Dict[str, Any]]) -> Optional[Any]:
        """Get value from cache"""
        cache_key = self._generate_key(key) if isinstance(key, dict) else key
        
        if self._is_expired(cache_key):
            if cache_key in self.cache:
                del self.cache[cache_key]
                if cache_key in self.access_times:
                    del self.access_times[cache_key]
            return None
        
        # Update access time
        self.access_times[cache_key] = time.time()
        
        return self.cache[cache_key]['value']
    
    async def set(self, key: Union[str, Dict[str, Any]], value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache with TTL"""
        cache_key = self._generate_key(key) if isinstance(key, dict) else key
        
        # Cleanup expired entries
        self._cleanup_expired()
        
        # Evict LRU if needed
        self._evict_lru()
        
        # Set TTL
        if ttl is None:
            ttl = self.default_ttl
        
        expires_at = datetime.now() + timedelta(seconds=ttl)
        
        self.cache[cache_key] = {
            'value': value,
            'expires_at': expires_at,
            'created_at': datetime.now()
        }
        
        self.access_times[cache_key] = time.time()
    
    async def delete(self, key: Union[str, Dict[str, Any]]) -> bool:
        """Delete key from cache"""
        cache_key = self._generate_key(key) if isinstance(key, dict) else key
        
        if cache_key in self.cache:
            del self.cache[cache_key]
            if cache_key in self.access_times:
                del self.access_times[cache_key]
            return True
        return False
    
    async def clear(self) -> None:
        """Clear all cache entries"""
        self.cache.clear()
        self.access_times.clear()
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        self._cleanup_expired()
        
        return {
            'total_entries': len(self.cache),
            'max_size': self.max_size,
            'default_ttl': self.default_ttl,
            'hit_rate': 0.0,  # Would need to track hits/misses
            'memory_usage_estimate': len(str(self.cache))
        }

# Global cache instance
_cache_manager: Optional[CacheManager] = None

def get_cache_manager() -> CacheManager:
    """Get the global cache manager instance"""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager

# Cache decorator
def cached(ttl: int = 3600, key_func: Optional[callable] = None):
    """
    Decorator to cache function results
    
    Args:
        ttl: Time to live in seconds
        key_func: Function to generate cache key from arguments
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            cache = get_cache_manager()
            
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = {
                    'func': func.__name__,
                    'args': args,
                    'kwargs': kwargs
                }
            
            # Try to get from cache
            result = await cache.get(cache_key)
            if result is not None:
                logger.debug(f"Cache hit for {func.__name__}")
                return result
            
            # Execute function and cache result
            logger.debug(f"Cache miss for {func.__name__}")
            result = await func(*args, **kwargs)
            await cache.set(cache_key, result, ttl)
            
            return result
        
        return wrapper
    return decorator