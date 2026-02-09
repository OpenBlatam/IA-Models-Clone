"""
Caching Strategy
===============

Advanced caching strategies for optimal performance.
"""

import logging
import hashlib
import json
from enum import Enum
from typing import Any, Optional, Callable, Dict
from functools import wraps
import time

logger = logging.getLogger(__name__)


class CacheStrategy(Enum):
    """Cache strategy types."""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    FIFO = "fifo"  # First In First Out
    TTL = "ttl"  # Time To Live
    WRITE_THROUGH = "write_through"
    WRITE_BACK = "write_back"


class CachingStrategy:
    """Advanced caching strategy manager."""
    
    def __init__(self, cache_adapter, strategy: CacheStrategy = CacheStrategy.TTL):
        self.cache_adapter = cache_adapter
        self.strategy = strategy
        self._access_times: Dict[str, float] = {}
        self._access_counts: Dict[str, int] = {}
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value with strategy."""
        value = await self.cache_adapter.get(key)
        
        if value:
            # Update access tracking
            self._access_times[key] = time.time()
            self._access_counts[key] = self._access_counts.get(key, 0) + 1
        
        return value
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        strategy: Optional[CacheStrategy] = None
    ) -> bool:
        """Set value with strategy."""
        strategy = strategy or self.strategy
        
        if strategy == CacheStrategy.WRITE_THROUGH:
            # Write to cache and underlying storage
            result = await self.cache_adapter.set(key, value, ttl)
            # In production, also write to underlying storage
            return result
        
        elif strategy == CacheStrategy.WRITE_BACK:
            # Write to cache only, flush later
            return await self.cache_adapter.set(key, value, ttl)
        
        else:
            # Standard TTL caching
            return await self.cache_adapter.set(key, value, ttl)
    
    async def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate keys matching pattern."""
        return await self.cache_adapter.clear(pattern)
    
    def cache_result(
        self,
        ttl: Optional[int] = None,
        key_prefix: str = "",
        key_builder: Optional[Callable] = None
    ):
        """Decorator to cache function results."""
        def decorator(func: Callable):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                # Build cache key
                if key_builder:
                    cache_key = key_builder(*args, **kwargs)
                else:
                    key_data = {
                        "func": func.__name__,
                        "args": str(args),
                        "kwargs": str(sorted(kwargs.items()))
                    }
                    key_str = json.dumps(key_data, sort_keys=True)
                    cache_key = f"{key_prefix}{func.__name__}:{hashlib.md5(key_str.encode()).hexdigest()}"
                
                # Try cache first
                cached = await self.get(cache_key)
                if cached is not None:
                    logger.debug(f"Cache hit for {func.__name__}")
                    return cached
                
                # Execute function
                logger.debug(f"Cache miss for {func.__name__}, executing...")
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                # Cache result
                await self.set(cache_key, result, ttl)
                return result
            
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                # For sync functions, we'd need to handle differently
                # This is a simplified version
                key_data = {
                    "func": func.__name__,
                    "args": str(args),
                    "kwargs": str(sorted(kwargs.items()))
                }
                key_str = json.dumps(key_data, sort_keys=True)
                cache_key = f"{key_prefix}{func.__name__}:{hashlib.md5(key_str.encode()).hexdigest()}"
                
                # In production, use sync cache adapter
                return func(*args, **kwargs)
            
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            return sync_wrapper
        
        return decorator


# Import asyncio for async functions
import asyncio















