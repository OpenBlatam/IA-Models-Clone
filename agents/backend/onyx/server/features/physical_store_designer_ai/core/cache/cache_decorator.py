"""
Cache Decorator

Decorator to cache function results.
"""

from typing import Callable, Optional, TypeVar, ParamSpec
from functools import wraps

from ..logging_config import get_logger
from .cache_manager import get_cache_manager
from .cache_utils import generate_cache_key

logger = get_logger(__name__)

P = ParamSpec('P')
T = TypeVar('T')


def cached(
    cache_name: str = "default",
    ttl: Optional[float] = None,
    key_prefix: Optional[str] = None
):
    """
    Decorator to cache function results
    
    Args:
        cache_name: Name of the cache to use
        ttl: Time to live in seconds (uses cache default if None)
        key_prefix: Optional prefix for cache keys
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        cache_manager = get_cache_manager()
        cache = cache_manager.get_cache(cache_name)
        
        @wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            # Generate cache key
            prefix = key_prefix or func.__name__
            cache_key = f"{prefix}:{generate_cache_key(*args, **kwargs)}"
            
            # Try to get from cache
            cached_value = cache.get(cache_key)
            if cached_value is not None:
                logger.debug(f"Cache hit: {cache_key}")
                return cached_value
            
            # Execute function
            logger.debug(f"Cache miss: {cache_key}")
            result = await func(*args, **kwargs)
            
            # Store in cache
            cache.set(cache_key, result, ttl)
            return result
        
        @wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            # Generate cache key
            prefix = key_prefix or func.__name__
            cache_key = f"{prefix}:{generate_cache_key(*args, **kwargs)}"
            
            # Try to get from cache
            cached_value = cache.get(cache_key)
            if cached_value is not None:
                logger.debug(f"Cache hit: {cache_key}")
                return cached_value
            
            # Execute function
            logger.debug(f"Cache miss: {cache_key}")
            result = func(*args, **kwargs)
            
            # Store in cache
            cache.set(cache_key, result, ttl)
            return result
        
        if hasattr(func, '__code__') and func.__code__.co_flags & 0x80:  # Check if async
            return async_wrapper
        return sync_wrapper
    
    return decorator




