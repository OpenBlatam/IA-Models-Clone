"""
Cache Strategies
Different caching strategies for different data types
"""

from typing import Any, Optional, Callable
from enum import Enum
import time
import logging

logger = logging.getLogger(__name__)


class CacheStrategy(Enum):
    """Cache strategy types"""
    NO_CACHE = "no_cache"
    SHORT_TERM = "short_term"  # 5 minutes
    MEDIUM_TERM = "medium_term"  # 1 hour
    LONG_TERM = "long_term"  # 24 hours
    PERMANENT = "permanent"  # Until manually invalidated


class CacheStrategyManager:
    """Manages different cache strategies"""
    
    # TTL mappings for strategies
    TTL_MAP = {
        CacheStrategy.NO_CACHE: 0,
        CacheStrategy.SHORT_TERM: 300,  # 5 minutes
        CacheStrategy.MEDIUM_TERM: 3600,  # 1 hour
        CacheStrategy.LONG_TERM: 86400,  # 24 hours
        CacheStrategy.PERMANENT: None,  # No expiration
    }
    
    def __init__(self, cache_service):
        self.cache_service = cache_service
    
    def get_ttl(self, strategy: CacheStrategy) -> Optional[int]:
        """Get TTL for strategy"""
        return self.TTL_MAP.get(strategy)
    
    async def get(
        self,
        key: str,
        strategy: CacheStrategy = CacheStrategy.MEDIUM_TERM
    ) -> Optional[Any]:
        """Get from cache"""
        if strategy == CacheStrategy.NO_CACHE:
            return None
        
        return await self.cache_service.get(key)
    
    async def set(
        self,
        key: str,
        value: Any,
        strategy: CacheStrategy = CacheStrategy.MEDIUM_TERM
    ) -> bool:
        """Set in cache with strategy"""
        if strategy == CacheStrategy.NO_CACHE:
            return False
        
        ttl = self.get_ttl(strategy)
        return await self.cache_service.set(key, value, ttl=ttl)
    
    async def get_or_set(
        self,
        key: str,
        factory: Callable,
        strategy: CacheStrategy = CacheStrategy.MEDIUM_TERM,
        *args,
        **kwargs
    ) -> Any:
        """
        Get from cache or set using factory function
        
        Args:
            key: Cache key
            factory: Function to generate value if not cached
            strategy: Cache strategy
            *args: Factory function arguments
            **kwargs: Factory function keyword arguments
        """
        # Try to get from cache
        cached = await self.get(key, strategy)
        if cached is not None:
            logger.debug(f"Cache hit: {key}")
            return cached
        
        # Generate value
        import asyncio
        if asyncio.iscoroutinefunction(factory):
            value = await factory(*args, **kwargs)
        else:
            value = factory(*args, **kwargs)
        
        # Cache it
        await self.set(key, value, strategy)
        logger.debug(f"Cached value: {key}")
        
        return value


def cache_with_strategy(strategy: CacheStrategy = CacheStrategy.MEDIUM_TERM):
    """
    Decorator to cache function results with specific strategy
    
    Args:
        strategy: Cache strategy to use
    """
    def decorator(func: Callable) -> Callable:
        import functools
        import asyncio
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Generate cache key
            import hashlib
            import json
            key_data = {"args": args, "kwargs": kwargs}
            key_str = json.dumps(key_data, sort_keys=True, default=str)
            cache_key = f"{func.__module__}.{func.__name__}:{hashlib.md5(key_str.encode()).hexdigest()}"
            
            # This would need cache_service injected
            # For now, just execute function
            return await func(*args, **kwargs)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return func
    
    return decorator















