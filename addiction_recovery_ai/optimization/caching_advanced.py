"""
Advanced Caching Strategies
Multi-layer caching with TTL, invalidation, and cache warming
"""

import logging
import hashlib
import json
import asyncio
from typing import Any, Optional, Callable, Dict
from datetime import datetime, timedelta
from functools import wraps

from core.interfaces import ICacheService
from core.service_container import get_container

logger = logging.getLogger(__name__)


class CacheStrategy:
    """Cache strategy configuration"""
    
    def __init__(
        self,
        ttl: int = 300,
        key_prefix: str = "",
        invalidate_on: Optional[list] = None,
        warm_on_startup: bool = False
    ):
        self.ttl = ttl
        self.key_prefix = key_prefix
        self.invalidate_on = invalidate_on or []
        self.warm_on_startup = warm_on_startup


class AdvancedCache:
    """
    Advanced caching with multiple strategies
    
    Features:
    - Multi-layer caching (L1: memory, L2: Redis)
    - Cache warming
    - Cache invalidation
    - TTL management
    - Cache statistics
    """
    
    def __init__(self):
        self._l1_cache: Dict[str, tuple] = {}  # In-memory cache
        self._l1_max_size = 1000
        self._cache_service: Optional[ICacheService] = None
        self._stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "invalidations": 0
        }
    
    def _get_cache_service(self) -> ICacheService:
        """Get cache service (lazy initialization)"""
        if self._cache_service is None:
            container = get_container()
            self._cache_service = container.get_cache_service()
        return self._cache_service
    
    async def get(
        self,
        key: str,
        strategy: Optional[CacheStrategy] = None
    ) -> Optional[Any]:
        """Get value from cache (L1 then L2)"""
        full_key = self._build_key(key, strategy)
        
        # Try L1 cache first
        if full_key in self._l1_cache:
            value, expiry = self._l1_cache[full_key]
            if datetime.now() < expiry:
                self._stats["hits"] += 1
                return value
            else:
                # Expired, remove from L1
                del self._l1_cache[full_key]
        
        # Try L2 cache (Redis)
        try:
            cache_service = self._get_cache_service()
            value = await cache_service.get(full_key)
            
            if value is not None:
                # Store in L1 for faster access
                self._set_l1(full_key, value, strategy)
                self._stats["hits"] += 1
                return value
        except Exception as e:
            logger.warning(f"L2 cache error: {str(e)}")
        
        self._stats["misses"] += 1
        return None
    
    async def set(
        self,
        key: str,
        value: Any,
        strategy: Optional[CacheStrategy] = None
    ) -> None:
        """Set value in cache (L1 and L2)"""
        full_key = self._build_key(key, strategy)
        ttl = strategy.ttl if strategy else 300
        
        # Set in L1
        self._set_l1(full_key, value, strategy)
        
        # Set in L2
        try:
            cache_service = self._get_cache_service()
            await cache_service.set(full_key, value, ttl=ttl)
        except Exception as e:
            logger.warning(f"L2 cache set error: {str(e)}")
        
        self._stats["sets"] += 1
    
    def _set_l1(self, key: str, value: Any, strategy: Optional[CacheStrategy] = None) -> None:
        """Set value in L1 cache"""
        ttl = strategy.ttl if strategy else 300
        expiry = datetime.now() + timedelta(seconds=ttl)
        
        # Evict if cache is full
        if len(self._l1_cache) >= self._l1_max_size:
            # Remove oldest entry
            oldest_key = min(self._l1_cache.keys(), key=lambda k: self._l1_cache[k][1])
            del self._l1_cache[oldest_key]
        
        self._l1_cache[key] = (value, expiry)
    
    async def invalidate(self, pattern: str) -> int:
        """Invalidate cache entries matching pattern"""
        count = 0
        
        # Invalidate L1
        keys_to_remove = [k for k in self._l1_cache.keys() if pattern in k]
        for key in keys_to_remove:
            del self._l1_cache[key]
            count += 1
        
        # Invalidate L2 (would need Redis SCAN in production)
        self._stats["invalidations"] += count
        return count
    
    def _build_key(self, key: str, strategy: Optional[CacheStrategy] = None) -> str:
        """Build cache key with prefix"""
        prefix = strategy.key_prefix if strategy else ""
        return f"{prefix}:{key}" if prefix else key
    
    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics"""
        total = self._stats["hits"] + self._stats["misses"]
        hit_rate = (self._stats["hits"] / total * 100) if total > 0 else 0
        
        return {
            **self._stats,
            "hit_rate": round(hit_rate, 2),
            "l1_size": len(self._l1_cache)
        }


def cached(
    ttl: int = 300,
    key_prefix: str = "",
    key_func: Optional[Callable] = None
):
    """
    Decorator for caching function results
    
    Args:
        ttl: Time to live in seconds
        key_prefix: Prefix for cache key
        key_func: Function to generate cache key from arguments
    """
    def decorator(func: Callable) -> Callable:
        cache = AdvancedCache()
        strategy = CacheStrategy(ttl=ttl, key_prefix=key_prefix)
        
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                # Default: hash arguments
                key_data = json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True)
                cache_key = hashlib.md5(key_data.encode()).hexdigest()
            
            # Try cache
            cached_value = await cache.get(cache_key, strategy)
            if cached_value is not None:
                return cached_value
            
            # Execute function
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            # Cache result
            await cache.set(cache_key, result, strategy)
            
            return result
        
        return wrapper
    return decorator


# Global cache instance
_cache_instance: Optional[AdvancedCache] = None


def get_advanced_cache() -> AdvancedCache:
    """Get global advanced cache instance"""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = AdvancedCache()
    return _cache_instance

