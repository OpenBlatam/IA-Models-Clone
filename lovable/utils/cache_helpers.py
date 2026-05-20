"""
Cache helper utilities.
"""

from typing import Optional, Callable, Any
import logging

logger = logging.getLogger(__name__)


def cache_key(prefix: str, *args, **kwargs) -> str:
    """
    Generate a cache key from prefix and arguments.
    
    Args:
        prefix: Key prefix
        *args: Positional arguments
        **kwargs: Keyword arguments
        
    Returns:
        Cache key string
    """
    key_parts = [prefix]
    
    if args:
        key_parts.extend(str(arg) for arg in args)
    
    if kwargs:
        sorted_kwargs = sorted(kwargs.items())
        key_parts.extend(f"{k}:{v}" for k, v in sorted_kwargs)
    
    return ":".join(key_parts)


def cached_result(
    cache_func: Callable,
    key: str,
    ttl: Optional[int] = None,
    fallback: Optional[Callable] = None
) -> Any:
    """
    Get cached result or compute and cache.
    
    Args:
        cache_func: Cache get function
        key: Cache key
        ttl: Time to live in seconds
        fallback: Function to call if cache miss
        
    Returns:
        Cached or computed result
    """
    try:
        result = cache_func(key)
        if result is not None:
            logger.debug(f"Cache hit for key: {key}")
            # Record cache hit
            try:
                from ..utils.performance_metrics import get_metrics
                get_metrics().record_cache_hit()
            except Exception:
                pass
            return result
    except Exception as e:
        logger.warning(f"Cache error for key {key}: {e}")
    
    if fallback:
        logger.debug(f"Cache miss for key: {key}, computing...")
        # Record cache miss
        try:
            from ..utils.performance_metrics import get_metrics
            get_metrics().record_cache_miss()
        except Exception:
            pass
        
        result = fallback()
        
        # Try to cache the result
        try:
            from ..utils.cache import get_cache
            cache = get_cache()
            if ttl:
                cache.set(key, result, ttl=ttl)
            else:
                cache.set(key, result)
        except Exception as e:
            logger.warning(f"Failed to cache result for key {key}: {e}")
        
        return result
    
    return None






