"""
Cache helper utilities
"""

from typing import Optional, Any, Callable
import hashlib
import json
import logging

logger = logging.getLogger(__name__)


def generate_cache_key(prefix: str, *args, **kwargs) -> str:
    """
    Generate a cache key from arguments
    
    Args:
        prefix: Cache key prefix
        *args: Positional arguments
        **kwargs: Keyword arguments
    
    Returns:
        Cache key string
    """
    key_data = {
        "args": args,
        "kwargs": kwargs
    }
    key_str = json.dumps(key_data, sort_keys=True, default=str)
    key_hash = hashlib.md5(key_str.encode()).hexdigest()
    return f"{prefix}:{key_hash}"


def cache_key_from_params(prefix: str, **params) -> str:
    """
    Generate cache key from parameters
    
    Args:
        prefix: Cache key prefix
        **params: Parameters to include in key
    
    Returns:
        Cache key string
    """
    return generate_cache_key(prefix, **params)


def cached_call(
    cache_manager,
    cache_key: str,
    func: Callable,
    *args,
    ttl: int = 300,
    **kwargs
) -> Any:
    """
    Call function with caching
    
    Args:
        cache_manager: Cache manager instance
        cache_key: Cache key
        func: Function to call
        *args: Function arguments
        ttl: Time to live in seconds
        **kwargs: Function keyword arguments
    
    Returns:
        Function result (from cache or execution)
    """
    cached = cache_manager.get("api", cache_key)
    if cached is not None:
        logger.debug(f"Cache hit: {cache_key}")
        return cached
    
    logger.debug(f"Cache miss: {cache_key}")
    result = func(*args, **kwargs)
    cache_manager.set("api", cache_key, result, ttl=ttl)
    return result

