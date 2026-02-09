"""
Query Cache System
Aggressive caching for fast responses
"""

import hashlib
import json
from typing import Any, Optional, Dict
from datetime import datetime, timedelta
from functools import wraps
import asyncio


class QueryCache:
    """Fast query result cache"""
    
    def __init__(self, default_ttl: int = 300):
        self._cache: Dict[str, tuple[Any, datetime]] = {}
        self.default_ttl = default_ttl
        self._lock = asyncio.Lock()
    
    def _generate_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments"""
        key_data = {
            "args": str(args),
            "kwargs": sorted(kwargs.items())
        }
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get from cache"""
        async with self._lock:
            if key not in self._cache:
                return None
            
            value, cached_at = self._cache[key]
            
            # Check expiration
            if datetime.utcnow() - cached_at > timedelta(seconds=self.default_ttl):
                del self._cache[key]
                return None
            
            return value
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set in cache"""
        async with self._lock:
            self._cache[key] = (value, datetime.utcnow())
    
    async def delete(self, key: str) -> bool:
        """Delete from cache"""
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False
    
    async def clear(self) -> None:
        """Clear all cache"""
        async with self._lock:
            self._cache.clear()


# Global cache instance
_query_cache = QueryCache(default_ttl=60)


def cache_query(ttl: int = 60):
    """Decorator to cache query results"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = _query_cache._generate_key(*args, **kwargs)
            
            # Check cache
            cached_result = await _query_cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Cache result
            await _query_cache.set(cache_key, result, ttl)
            
            return result
        
        return wrapper
    return decorator


class BatchQueryOptimizer:
    """Optimize batch queries to avoid N+1 problem"""
    
    def __init__(self):
        self.pending_queries: Dict[str, asyncio.Future] = {}
    
    async def batch_get(
        self,
        ids: list[str],
        get_func: callable,
        batch_size: int = 100
    ) -> Dict[str, Any]:
        """Get multiple items in batches"""
        results = {}
        
        # Process in batches
        for i in range(0, len(ids), batch_size):
            batch = ids[i:i + batch_size]
            
            # Execute batch query
            batch_results = await asyncio.gather(*[
                get_func(id) for id in batch
            ])
            
            # Merge results
            for id, result in zip(batch, batch_results):
                if result:
                    results[id] = result
        
        return results






