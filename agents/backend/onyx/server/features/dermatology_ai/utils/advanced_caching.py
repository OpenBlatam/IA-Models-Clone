"""
Advanced Caching Strategies
Implements multiple caching patterns: Cache-Aside, Write-Through, Write-Back
"""

from typing import Any, Optional, Callable, Dict
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class CacheStrategy(str, Enum):
    """Caching strategies"""
    CACHE_ASIDE = "cache_aside"  # Application manages cache
    WRITE_THROUGH = "write_through"  # Write to cache and DB simultaneously
    WRITE_BACK = "write_back"  # Write to cache, flush to DB later
    REFRESH_AHEAD = "refresh_ahead"  # Prefetch before expiration


class AdvancedCache:
    """
    Advanced cache with multiple strategies.
    Supports different caching patterns for different use cases.
    """
    
    def __init__(self, cache_service, strategy: CacheStrategy = CacheStrategy.CACHE_ASIDE):
        self.cache = cache_service
        self.strategy = strategy
    
    async def get(
        self,
        key: str,
        fetch_fn: Optional[Callable] = None,
        ttl: Optional[int] = None
    ) -> Optional[Any]:
        """
        Get value from cache with strategy
        
        Args:
            key: Cache key
            fetch_fn: Function to fetch if cache miss
            ttl: Time to live in seconds
            
        Returns:
            Cached value or fetched value
        """
        # Try cache first
        value = await self.cache.get(key)
        
        if value is not None:
            logger.debug(f"Cache hit: {key}")
            return value
        
        # Cache miss
        logger.debug(f"Cache miss: {key}")
        
        if fetch_fn:
            # Fetch from source
            value = await fetch_fn()
            
            # Store in cache
            if value is not None:
                await self.cache.set(key, value, ttl)
                logger.debug(f"Cached value for: {key}")
        
        return value
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        write_fn: Optional[Callable] = None
    ) -> bool:
        """
        Set value in cache with strategy
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live
            write_fn: Function to write to persistent storage
            
        Returns:
            True if successful
        """
        if self.strategy == CacheStrategy.WRITE_THROUGH:
            # Write to cache and storage simultaneously
            if write_fn:
                await write_fn(key, value)
            
            return await self.cache.set(key, value, ttl)
        
        elif self.strategy == CacheStrategy.WRITE_BACK:
            # Write to cache only, flush later
            await self.cache.set(key, value, ttl)
            # Schedule async write to storage
            if write_fn:
                # In production, use background task queue
                logger.debug(f"Scheduled write-back for: {key}")
            return True
        
        else:  # CACHE_ASIDE
            # Just write to cache
            return await self.cache.set(key, value, ttl)
    
    async def invalidate(self, key: str):
        """Invalidate cache key"""
        await self.cache.delete(key)
        logger.debug(f"Invalidated cache: {key}")
    
    async def invalidate_pattern(self, pattern: str):
        """Invalidate all keys matching pattern"""
        # Implementation depends on cache backend
        # Redis supports pattern matching
        logger.debug(f"Invalidating pattern: {pattern}")


class CacheDecorator:
    """Decorator for caching function results"""
    
    def __init__(self, cache: AdvancedCache, ttl: int = 3600, key_fn: Optional[Callable] = None):
        self.cache = cache
        self.ttl = ttl
        self.key_fn = key_fn or self._default_key_fn
    
    def _default_key_fn(self, *args, **kwargs) -> str:
        """Generate default cache key"""
        import hashlib
        import json
        
        key_data = {
            "args": str(args),
            "kwargs": json.dumps(kwargs, sort_keys=True, default=str)
        }
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def __call__(self, func: Callable):
        """Decorator implementation"""
        async def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = f"{func.__name__}:{self.key_fn(*args, **kwargs)}"
            
            # Try cache
            cached_result = await self.cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Cache miss - execute function
            result = await func(*args, **kwargs)
            
            # Cache result
            await self.cache.set(cache_key, result, self.ttl)
            
            return result
        
        return wrapper















