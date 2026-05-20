"""
Caching utilities for performance optimization
"""

from functools import wraps
from typing import Callable, Any, Optional
import hashlib
import json
import logging

logger = logging.getLogger(__name__)

# Simple in-memory cache (replace with Redis in production)
_cache: dict[str, tuple[Any, float]] = {}


def get_cache_key(*args: Any, **kwargs: Any) -> str:
    """Generate cache key from function arguments"""
    key_data = {
        "args": args,
        "kwargs": sorted(kwargs.items())
    }
    key_string = json.dumps(key_data, sort_keys=True, default=str)
    return hashlib.md5(key_string.encode()).hexdigest()


def cache_result(
    ttl: int = 300,
    key_prefix: Optional[str] = None
) -> Callable:
    """
    Decorator to cache function results
    
    Args:
        ttl: Time to live in seconds (default: 5 minutes)
        key_prefix: Optional prefix for cache key
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            import time
            
            cache_key = get_cache_key(*args, **kwargs)
            if key_prefix:
                cache_key = f"{key_prefix}:{cache_key}"
            
            # Check cache
            if cache_key in _cache:
                result, expiry = _cache[cache_key]
                if time.time() < expiry:
                    logger.debug(f"Cache hit: {func.__name__}")
                    return result
                else:
                    # Expired, remove from cache
                    del _cache[cache_key]
            
            # Cache miss, execute function
            logger.debug(f"Cache miss: {func.__name__}")
            result = await func(*args, **kwargs)
            
            # Store in cache
            _cache[cache_key] = (result, time.time() + ttl)
            
            return result
        
        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            import time
            
            cache_key = get_cache_key(*args, **kwargs)
            if key_prefix:
                cache_key = f"{key_prefix}:{cache_key}"
            
            # Check cache
            if cache_key in _cache:
                result, expiry = _cache[cache_key]
                if time.time() < expiry:
                    logger.debug(f"Cache hit: {func.__name__}")
                    return result
                else:
                    del _cache[cache_key]
            
            # Cache miss, execute function
            logger.debug(f"Cache miss: {func.__name__}")
            result = func(*args, **kwargs)
            
            # Store in cache
            _cache[cache_key] = (result, time.time() + ttl)
            
            return result
        
        # Return appropriate wrapper based on function type
        import inspect
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


def clear_cache(key_prefix: Optional[str] = None) -> int:
    """
    Clear cache entries
    
    Args:
        key_prefix: Optional prefix to clear only matching keys
    
    Returns:
        Number of entries cleared
    """
    if key_prefix:
        keys_to_remove = [
            key for key in _cache.keys()
            if key.startswith(key_prefix)
        ]
        for key in keys_to_remove:
            del _cache[key]
        return len(keys_to_remove)
    
    count = len(_cache)
    _cache.clear()
    return count


def get_cache_stats() -> dict[str, Any]:
    """Get cache statistics"""
    import time
    
    active_entries = sum(
        1 for _, expiry in _cache.values()
        if time.time() < expiry
    )
    
    return {
        "total_entries": len(_cache),
        "active_entries": active_entries,
        "expired_entries": len(_cache) - active_entries
    }

