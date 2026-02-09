"""
Advanced Caching System for Color Grading AI
=============================================

Advanced caching with multiple strategies, invalidation, and optimization.
"""

import logging
import asyncio
import hashlib
import json
from typing import Dict, Any, Optional, List, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import time

logger = logging.getLogger(__name__)


class CacheStrategy(Enum):
    """Cache strategies."""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    FIFO = "fifo"  # First In First Out
    TTL = "ttl"  # Time To Live
    ADAPTIVE = "adaptive"  # Adaptive based on usage


@dataclass
class CacheEntry:
    """Cache entry."""
    key: str
    value: Any
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    ttl: Optional[timedelta] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class AdvancedCache:
    """
    Advanced caching system.
    
    Features:
    - Multiple cache strategies
    - Automatic invalidation
    - Cache warming
    - Statistics tracking
    - Memory management
    - Async support
    """
    
    def __init__(
        self,
        max_size: int = 1000,
        strategy: CacheStrategy = CacheStrategy.LRU,
        default_ttl: Optional[timedelta] = None
    ):
        """
        Initialize advanced cache.
        
        Args:
            max_size: Maximum cache size
            strategy: Cache strategy
            default_ttl: Default TTL
        """
        self.max_size = max_size
        self.strategy = strategy
        self.default_ttl = default_ttl
        self._cache: Dict[str, CacheEntry] = {}
        self._lock = asyncio.Lock()
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "sets": 0,
        }
    
    def _generate_key(self, *args, **kwargs) -> str:
        """Generate cache key."""
        key_data = {
            "args": args,
            "kwargs": sorted(kwargs.items())
        }
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    async def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None
        """
        async with self._lock:
            entry = self._cache.get(key)
            
            if not entry:
                self._stats["misses"] += 1
                return None
            
            # Check TTL
            if entry.ttl:
                if datetime.now() - entry.created_at > entry.ttl:
                    del self._cache[key]
                    self._stats["misses"] += 1
                    return None
            
            # Update access
            entry.last_accessed = datetime.now()
            entry.access_count += 1
            self._stats["hits"] += 1
            
            return entry.value
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[timedelta] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Optional TTL
            metadata: Optional metadata
        """
        async with self._lock:
            # Check if we need to evict
            if len(self._cache) >= self.max_size and key not in self._cache:
                await self._evict()
            
            entry = CacheEntry(
                key=key,
                value=value,
                ttl=ttl or self.default_ttl,
                metadata=metadata or {}
            )
            
            self._cache[key] = entry
            self._stats["sets"] += 1
    
    async def _evict(self):
        """Evict entry based on strategy."""
        if not self._cache:
            return
        
        if self.strategy == CacheStrategy.LRU:
            # Remove least recently used
            key_to_remove = min(
                self._cache.items(),
                key=lambda x: x[1].last_accessed
            )[0]
        
        elif self.strategy == CacheStrategy.LFU:
            # Remove least frequently used
            key_to_remove = min(
                self._cache.items(),
                key=lambda x: x[1].access_count
            )[0]
        
        elif self.strategy == CacheStrategy.FIFO:
            # Remove oldest
            key_to_remove = min(
                self._cache.items(),
                key=lambda x: x[1].created_at
            )[0]
        
        elif self.strategy == CacheStrategy.TTL:
            # Remove expired or oldest
            now = datetime.now()
            expired = [
                (k, v) for k, v in self._cache.items()
                if v.ttl and (now - v.created_at) > v.ttl
            ]
            if expired:
                key_to_remove = expired[0][0]
            else:
                key_to_remove = min(
                    self._cache.items(),
                    key=lambda x: x[1].created_at
                )[0]
        
        elif self.strategy == CacheStrategy.ADAPTIVE:
            # Adaptive: combine LRU and LFU
            scores = {
                k: v.access_count / max((datetime.now() - v.last_accessed).total_seconds(), 1)
                for k, v in self._cache.items()
            }
            key_to_remove = min(scores.items(), key=lambda x: x[1])[0]
        
        else:
            # Default: remove first
            key_to_remove = next(iter(self._cache))
        
        del self._cache[key_to_remove]
        self._stats["evictions"] += 1
        logger.debug(f"Evicted cache key: {key_to_remove}")
    
    async def invalidate(self, key: str) -> bool:
        """
        Invalidate cache entry.
        
        Args:
            key: Cache key
            
        Returns:
            True if invalidated
        """
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False
    
    async def invalidate_pattern(self, pattern: str) -> int:
        """
        Invalidate entries matching pattern.
        
        Args:
            pattern: Pattern to match
            
        Returns:
            Number of entries invalidated
        """
        async with self._lock:
            keys_to_remove = [k for k in self._cache.keys() if pattern in k]
            for key in keys_to_remove:
                del self._cache[key]
            return len(keys_to_remove)
    
    async def clear(self):
        """Clear all cache."""
        async with self._lock:
            self._cache.clear()
            logger.info("Cache cleared")
    
    async def warm_cache(self, items: Dict[str, Any]):
        """
        Warm cache with items.
        
        Args:
            items: Dictionary of key-value pairs
        """
        for key, value in items.items():
            await self.set(key, value)
        logger.info(f"Warmed cache with {len(items)} items")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self._stats["hits"] + self._stats["misses"]
        hit_rate = self._stats["hits"] / total_requests if total_requests > 0 else 0.0
        
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "strategy": self.strategy.value,
            "hits": self._stats["hits"],
            "misses": self._stats["misses"],
            "hit_rate": hit_rate,
            "evictions": self._stats["evictions"],
            "sets": self._stats["sets"],
        }
    
    def cache_decorator(
        self,
        ttl: Optional[timedelta] = None,
        key_func: Optional[Callable] = None
    ):
        """
        Decorator for caching function results.
        
        Args:
            ttl: Optional TTL
            key_func: Optional key generation function
        """
        def decorator(func: Callable):
            async def wrapper(*args, **kwargs):
                # Generate key
                if key_func:
                    cache_key = key_func(*args, **kwargs)
                else:
                    cache_key = self._generate_key(*args, **kwargs)
                
                # Try to get from cache
                cached = await self.get(cache_key)
                if cached is not None:
                    return cached
                
                # Execute function
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                # Cache result
                await self.set(cache_key, result, ttl=ttl)
                
                return result
            
            return wrapper
        return decorator




