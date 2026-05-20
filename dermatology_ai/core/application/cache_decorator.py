"""
Cache Decorator for Use Cases
Provides transparent caching for expensive operations
"""

import hashlib
import json
from typing import Callable, Any, Optional
from functools import wraps
import logging

logger = logging.getLogger(__name__)


def cache_result(
    cache_service,
    ttl: int = 3600,
    key_prefix: Optional[str] = None,
    serialize_args: bool = True
):
    """
    Decorator to cache function results
    
    Args:
        cache_service: Cache service instance
        ttl: Time to live in seconds
        key_prefix: Prefix for cache keys
        serialize_args: Whether to serialize arguments for cache key
    """
    def decorator(func: Callable) -> Callable:
        prefix = key_prefix or f"{func.__module__}.{func.__name__}"
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = _generate_cache_key(prefix, args, kwargs, serialize_args)
            
            # Try to get from cache
            try:
                cached_result = await cache_service.get(cache_key)
                if cached_result is not None:
                    logger.debug(f"Cache hit for {cache_key}")
                    return cached_result
            except Exception as e:
                logger.warning(f"Cache get failed: {e}")
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Store in cache
            try:
                await cache_service.set(cache_key, result, ttl=ttl)
                logger.debug(f"Cached result for {cache_key} (TTL: {ttl}s)")
            except Exception as e:
                logger.warning(f"Cache set failed: {e}")
            
            return result
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # For sync functions, we'd need to handle differently
            # For now, just execute without caching
            return func(*args, **kwargs)
        
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


def _generate_cache_key(
    prefix: str,
    args: tuple,
    kwargs: dict,
    serialize_args: bool
) -> str:
    """Generate cache key from function arguments"""
    if serialize_args:
        # Serialize arguments for cache key
        key_data = {
            "args": args,
            "kwargs": kwargs
        }
        try:
            key_str = json.dumps(key_data, sort_keys=True, default=str)
        except (TypeError, ValueError):
            # Fallback to string representation
            key_str = f"{args}{kwargs}"
    else:
        # Simple string representation
        key_str = f"{args}{kwargs}"
    
    # Create hash for shorter key
    key_hash = hashlib.md5(key_str.encode()).hexdigest()
    return f"{prefix}:{key_hash}"


def invalidate_cache(
    cache_service,
    key_pattern: str
):
    """
    Decorator to invalidate cache entries matching pattern
    Useful for write operations that should invalidate related caches
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Execute function
            result = await func(*args, **kwargs)
            
            # Invalidate cache (simplified - in production, use pattern matching)
            try:
                # This is a simplified version
                # In production, you'd want to track cache keys or use pattern matching
                logger.debug(f"Cache invalidation requested for pattern: {key_pattern}")
                # Actual invalidation would depend on cache implementation
            except Exception as e:
                logger.warning(f"Cache invalidation failed: {e}")
            
            return result
        
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return func
    
    return decorator















