"""
Cache Utilities
================

Advanced caching utilities.
"""

import hashlib
import json
import logging
import time
from typing import Any, Callable, Optional, Dict
from functools import wraps
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class CacheUtils:
    """Cache utility functions."""
    
    @staticmethod
    def generate_key(*args, **kwargs) -> str:
        """
        Generate cache key from arguments.
        
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
    
    @staticmethod
    def cache_result(ttl: int = 3600, key_prefix: str = ""):
        """
        Decorator to cache function results.
        
        Args:
            ttl: Time to live in seconds
            key_prefix: Key prefix
            
        Returns:
            Decorator function
        """
        def decorator(func: Callable) -> Callable:
            cache: Dict[str, tuple[Any, float]] = {}
            
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                cache_key = CacheUtils.generate_key(*args, **kwargs)
                if key_prefix:
                    cache_key = f"{key_prefix}:{cache_key}"
                
                # Check cache
                if cache_key in cache:
                    result, expiry = cache[cache_key]
                    if time.time() < expiry:
                        logger.debug(f"Cache hit for {func.__name__}")
                        return result
                    else:
                        # Expired
                        del cache[cache_key]
                
                # Execute function
                result = await func(*args, **kwargs)
                
                # Store in cache
                cache[cache_key] = (result, time.time() + ttl)
                logger.debug(f"Cached result for {func.__name__}")
                
                return result
            
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                cache_key = CacheUtils.generate_key(*args, **kwargs)
                if key_prefix:
                    cache_key = f"{key_prefix}:{cache_key}"
                
                # Check cache
                if cache_key in cache:
                    result, expiry = cache[cache_key]
                    if time.time() < expiry:
                        logger.debug(f"Cache hit for {func.__name__}")
                        return result
                    else:
                        # Expired
                        del cache[cache_key]
                
                # Execute function
                result = func(*args, **kwargs)
                
                # Store in cache
                cache[cache_key] = (result, time.time() + ttl)
                logger.debug(f"Cached result for {func.__name__}")
                
                return result
            
            # Return appropriate wrapper
            import inspect
            if inspect.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper
        
        return decorator
    
    @staticmethod
    def invalidate_cache(pattern: str, cache: Dict[str, Any]):
        """
        Invalidate cache entries matching pattern.
        
        Args:
            pattern: Pattern to match
            cache: Cache dictionary
        """
        keys_to_remove = [key for key in cache.keys() if pattern in key]
        for key in keys_to_remove:
            del cache[key]
        logger.info(f"Invalidated {len(keys_to_remove)} cache entries matching {pattern}")
    
    @staticmethod
    def clear_cache(cache: Dict[str, Any]):
        """Clear all cache entries."""
        cache.clear()
        logger.info("Cache cleared")
    
    @staticmethod
    def get_cache_stats(cache: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Args:
            cache: Cache dictionary
            
        Returns:
            Cache statistics
        """
        total_size = len(cache)
        expired = sum(
            1 for _, (_, expiry) in cache.items()
            if time.time() >= expiry
        )
        
        return {
            "total_entries": total_size,
            "expired_entries": expired,
            "active_entries": total_size - expired
        }




