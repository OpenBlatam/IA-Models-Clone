"""
Caching system for content analysis results
"""

import asyncio
import hashlib
import json
import time
from typing import Any, Dict, Optional, Union
from functools import wraps
import logging

logger = logging.getLogger(__name__)


class MemoryCache:
    """Simple in-memory cache with TTL support"""
    
    def __init__(self, default_ttl: int = 300):
        self._cache: Dict[str, Dict[str, Any]] = {}
        self.default_ttl = default_ttl
    
    def _is_expired(self, entry: Dict[str, Any]) -> bool:
        """Check if cache entry is expired"""
        return time.time() > entry.get('expires_at', 0)
    
    def _cleanup_expired(self) -> None:
        """Remove expired entries"""
        current_time = time.time()
        expired_keys = [
            key for key, entry in self._cache.items()
            if current_time > entry.get('expires_at', 0)
        ]
        for key in expired_keys:
            del self._cache[key]
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if key not in self._cache:
            return None
        
        entry = self._cache[key]
        if self._is_expired(entry):
            del self._cache[key]
            return None
        
        return entry['value']
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache with TTL"""
        ttl = ttl or self.default_ttl
        self._cache[key] = {
            'value': value,
            'expires_at': time.time() + ttl
        }
        self._cleanup_expired()
    
    def delete(self, key: str) -> bool:
        """Delete key from cache"""
        if key in self._cache:
            del self._cache[key]
            return True
        return False
    
    def clear(self) -> None:
        """Clear all cache entries"""
        self._cache.clear()
    
    def size(self) -> int:
        """Get cache size"""
        self._cleanup_expired()
        return len(self._cache)


# Global cache instance
cache = MemoryCache(default_ttl=300)


def generate_cache_key(*args, **kwargs) -> str:
    """Generate cache key from arguments"""
    key_data = {
        'args': args,
        'kwargs': sorted(kwargs.items()) if kwargs else {}
    }
    key_string = json.dumps(key_data, sort_keys=True)
    return hashlib.md5(key_string.encode()).hexdigest()


def cached(ttl: int = 300, key_prefix: str = ""):
    """Decorator for caching function results"""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            cache_key = f"{key_prefix}:{generate_cache_key(*args, **kwargs)}"
            
            # Try to get from cache
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                logger.debug(f"Cache hit for {func.__name__}")
                return cached_result
            
            # Execute function and cache result
            logger.debug(f"Cache miss for {func.__name__}")
            result = await func(*args, **kwargs)
            cache.set(cache_key, result, ttl)
            
            return result
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            cache_key = f"{key_prefix}:{generate_cache_key(*args, **kwargs)}"
            
            # Try to get from cache
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                logger.debug(f"Cache hit for {func.__name__}")
                return cached_result
            
            # Execute function and cache result
            logger.debug(f"Cache miss for {func.__name__}")
            result = func(*args, **kwargs)
            cache.set(cache_key, result, ttl)
            
            return result
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator


def cache_analysis_result(content: str, result: Dict[str, Any], ttl: int = 300) -> None:
    """Cache analysis result"""
    content_hash = hashlib.md5(content.encode()).hexdigest()
    cache_key = f"analysis:{content_hash}"
    cache.set(cache_key, result, ttl)


def get_cached_analysis_result(content: str) -> Optional[Dict[str, Any]]:
    """Get cached analysis result"""
    content_hash = hashlib.md5(content.encode()).hexdigest()
    cache_key = f"analysis:{content_hash}"
    return cache.get(cache_key)


def cache_similarity_result(text1: str, text2: str, threshold: float, result: Dict[str, Any], ttl: int = 300) -> None:
    """Cache similarity result"""
    # Create deterministic key for text pair
    text_pair = tuple(sorted([text1, text2]))
    cache_key = f"similarity:{hashlib.md5(str(text_pair).encode()).hexdigest()}:{threshold}"
    cache.set(cache_key, result, ttl)


def get_cached_similarity_result(text1: str, text2: str, threshold: float) -> Optional[Dict[str, Any]]:
    """Get cached similarity result"""
    text_pair = tuple(sorted([text1, text2]))
    cache_key = f"similarity:{hashlib.md5(str(text_pair).encode()).hexdigest()}:{threshold}"
    return cache.get(cache_key)


def cache_quality_result(content: str, result: Dict[str, Any], ttl: int = 300) -> None:
    """Cache quality result"""
    content_hash = hashlib.md5(content.encode()).hexdigest()
    cache_key = f"quality:{content_hash}"
    cache.set(cache_key, result, ttl)


def get_cached_quality_result(content: str) -> Optional[Dict[str, Any]]:
    """Get cached quality result"""
    content_hash = hashlib.md5(content.encode()).hexdigest()
    cache_key = f"quality:{content_hash}"
    return cache.get(cache_key)


def get_cache_stats() -> Dict[str, Any]:
    """Get cache statistics"""
    return {
        "size": cache.size(),
        "default_ttl": cache.default_ttl,
        "timestamp": time.time()
    }


def clear_cache() -> None:
    """Clear all cache entries"""
    cache.clear()
    logger.info("Cache cleared")
