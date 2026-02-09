"""
Response Cache
=============

Ultra-fast response caching.
"""

import logging
import hashlib
import json
import time
from typing import Dict, Any, Optional, Callable
from functools import wraps
from aws.modules.ports.cache_port import CachePort

logger = logging.getLogger(__name__)


class ResponseCache:
    """Ultra-fast response cache."""
    
    def __init__(self, cache: CachePort, default_ttl: int = 300):
        self.cache = cache
        self.default_ttl = default_ttl
        self._stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0
        }
    
    def cache_response(
        self,
        ttl: Optional[int] = None,
        key_builder: Optional[Callable] = None,
        vary_headers: Optional[List[str]] = None
    ):
        """Decorator to cache function responses."""
        def decorator(func: Callable):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                # Build cache key
                if key_builder:
                    cache_key = key_builder(*args, **kwargs)
                else:
                    cache_key = self._build_key(func.__name__, args, kwargs, vary_headers)
                
                # Try cache
                cached = await self.cache.get(cache_key)
                if cached is not None:
                    self._stats["hits"] += 1
                    logger.debug(f"Cache hit: {cache_key}")
                    return cached
                
                # Cache miss
                self._stats["misses"] += 1
                logger.debug(f"Cache miss: {cache_key}")
                
                # Execute function
                result = await func(*args, **kwargs)
                
                # Cache result
                await self.cache.set(cache_key, result, ttl=ttl or self.default_ttl)
                self._stats["sets"] += 1
                
                return result
            
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                # For sync functions
                cache_key = self._build_key(func.__name__, args, kwargs, vary_headers)
                
                # Try cache (sync)
                # In production, use sync cache adapter
                result = func(*args, **kwargs)
                return result
            
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            return sync_wrapper
        
        return decorator
    
    def _build_key(
        self,
        func_name: str,
        args: tuple,
        kwargs: dict,
        vary_headers: Optional[List[str]]
    ) -> str:
        """Build cache key."""
        key_data = {
            "func": func_name,
            "args": str(args),
            "kwargs": json.dumps(kwargs, sort_keys=True, default=str)
        }
        
        if vary_headers:
            key_data["headers"] = vary_headers
        
        key_str = json.dumps(key_data, sort_keys=True)
        return f"response_cache:{func_name}:{hashlib.md5(key_str.encode()).hexdigest()}"
    
    async def invalidate_pattern(self, pattern: str):
        """Invalidate cache entries matching pattern."""
        await self.cache.clear(pattern)
        logger.info(f"Invalidated cache pattern: {pattern}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self._stats["hits"] + self._stats["misses"]
        hit_rate = (self._stats["hits"] / total * 100) if total > 0 else 0
        
        return {
            **self._stats,
            "total": total,
            "hit_rate": f"{hit_rate:.2f}%"
        }


# Import asyncio
import asyncio















