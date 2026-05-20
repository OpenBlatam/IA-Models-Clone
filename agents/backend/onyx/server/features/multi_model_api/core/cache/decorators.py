"""
Cache decorators and context managers
"""

import asyncio
import functools
import inspect
from typing import Any, Dict, Optional, List, Callable, Awaitable

from .utils import generate_key


def cached(
    ttl: Optional[int] = None,
    key_prefix: str = "",
    tags: Optional[List[str]] = None,
    cache_instance: Optional[Any] = None
):
    """
    Decorator to cache async function results
    
    Usage:
        @cached(ttl=3600, key_prefix="api", tags=["api", "users"])
        async def get_user(user_id: int):
            return await fetch_user(user_id)
    
    Args:
        ttl: Time-to-live in seconds
        key_prefix: Prefix for cache keys
        tags: List of tags for invalidation
        cache_instance: Cache instance to use (defaults to global)
    """
    def decorator(func: Callable) -> Callable:
        cache = cache_instance
        
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            if cache is None:
                from .cache import get_cache
                cache_inst = get_cache()
            else:
                cache_inst = cache
            
            key_parts = [key_prefix or func.__name__]
            
            sig = inspect.signature(func)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            
            for param_name, param_value in bound.arguments.items():
                if param_name != 'self' and param_name != 'cls':
                    key_parts.append(f"{param_name}:{param_value}")
            
            cache_key = generate_key(*key_parts)
            
            cached_value = await cache_inst.get(cache_key)
            if cached_value is not None:
                return cached_value
            
            result = await func(*args, **kwargs)
            await cache_inst.set(cache_key, result, ttl=ttl, tags=tags)
            
            return result
        
        return wrapper
    return decorator


class CacheContext:
    """Context manager for cache operations with automatic cleanup"""
    
    def __init__(self, cache: Any, keys: Optional[List[str]] = None):
        """Initialize cache context
        
        Args:
            cache: Cache instance
            keys: Optional list of keys to track
        """
        self.cache = cache
        self.keys = keys or []
        self._original_keys = []
    
    async def __aenter__(self):
        """Enter context"""
        if self.keys:
            for key in self.keys:
                if await self.cache.exists(key):
                    self._original_keys.append(key)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit context with cleanup on error"""
        if exc_type is not None and self.keys:
            for key in self.keys:
                await self.cache.delete(key)
        return False
    
    async def get_or_set(self, key: str, factory: Callable[[], Awaitable[Any]], ttl: Optional[int] = None) -> Any:
        """Get from cache or set using factory function
        
        Args:
            key: Cache key
            factory: Async function to generate value if not cached
            ttl: Time-to-live in seconds
            
        Returns:
            Cached or generated value
        """
        value = await self.cache.get(key)
        if value is None:
            value = await factory()
            await self.cache.set(key, value, ttl=ttl)
            self.keys.append(key)
        return value

