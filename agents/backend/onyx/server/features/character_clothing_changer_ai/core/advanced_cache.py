"""
Advanced Cache System
=====================

Advanced caching system with multiple strategies and backends.
"""

import asyncio
import logging
import hashlib
import json
from typing import Dict, Any, Optional, List, Callable, Awaitable, TypeVar
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)

T = TypeVar('T')


class CacheStrategy(Enum):
    """Cache strategy."""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    FIFO = "fifo"  # First In First Out
    TTL = "ttl"  # Time To Live


@dataclass
class CacheEntry:
    """Cache entry."""
    key: str
    value: Any
    created_at: datetime = field(default_factory=datetime.now)
    accessed_at: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    expires_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_expired(self) -> bool:
        """Check if entry is expired."""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at
    
    def touch(self):
        """Update access information."""
        self.accessed_at = datetime.now()
        self.access_count += 1


class AdvancedCache:
    """Advanced cache with multiple strategies."""
    
    def __init__(
        self,
        max_size: int = 1000,
        strategy: CacheStrategy = CacheStrategy.LRU,
        default_ttl: Optional[float] = None
    ):
        """
        Initialize advanced cache.
        
        Args:
            max_size: Maximum cache size
            strategy: Cache eviction strategy
            default_ttl: Default TTL in seconds
        """
        self.max_size = max_size
        self.strategy = strategy
        self.default_ttl = default_ttl
        self.entries: Dict[str, CacheEntry] = {}
        self._lock = asyncio.Lock()
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "sets": 0
        }
    
    def _generate_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments."""
        key_data = {
            "args": args,
            "kwargs": kwargs
        }
        key_str = json.dumps(key_data, sort_keys=True)
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
            if key not in self.entries:
                self.stats["misses"] += 1
                return None
            
            entry = self.entries[key]
            
            # Check expiration
            if entry.is_expired():
                del self.entries[key]
                self.stats["misses"] += 1
                return None
            
            # Update access
            entry.touch()
            self.stats["hits"] += 1
            return entry.value
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Optional TTL in seconds
            metadata: Optional metadata
        """
        async with self._lock:
            # Check if we need to evict
            if len(self.entries) >= self.max_size and key not in self.entries:
                await self._evict()
            
            expires_at = None
            if ttl or self.default_ttl:
                ttl_value = ttl or self.default_ttl
                expires_at = datetime.now() + timedelta(seconds=ttl_value)
            
            entry = CacheEntry(
                key=key,
                value=value,
                expires_at=expires_at,
                metadata=metadata or {}
            )
            
            self.entries[key] = entry
            self.stats["sets"] += 1
    
    async def _evict(self):
        """Evict entry based on strategy."""
        if not self.entries:
            return
        
        if self.strategy == CacheStrategy.LRU:
            # Evict least recently used
            key_to_evict = min(
                self.entries.keys(),
                key=lambda k: self.entries[k].accessed_at
            )
        elif self.strategy == CacheStrategy.LFU:
            # Evict least frequently used
            key_to_evict = min(
                self.entries.keys(),
                key=lambda k: self.entries[k].access_count
            )
        elif self.strategy == CacheStrategy.FIFO:
            # Evict first in
            key_to_evict = min(
                self.entries.keys(),
                key=lambda k: self.entries[k].created_at
            )
        else:
            # Default: evict first
            key_to_evict = next(iter(self.entries.keys()))
        
        del self.entries[key_to_evict]
        self.stats["evictions"] += 1
        logger.debug(f"Evicted cache entry: {key_to_evict}")
    
    async def delete(self, key: str) -> bool:
        """
        Delete value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if deleted
        """
        async with self._lock:
            if key in self.entries:
                del self.entries[key]
                return True
            return False
    
    async def clear(self):
        """Clear all cache entries."""
        async with self._lock:
            self.entries.clear()
            logger.info("Cache cleared")
    
    async def cleanup(self) -> int:
        """
        Cleanup expired entries.
        
        Returns:
            Number of entries cleaned
        """
        async with self._lock:
            expired_keys = [
                key for key, entry in self.entries.items()
                if entry.is_expired()
            ]
            
            for key in expired_keys:
                del self.entries[key]
            
            return len(expired_keys)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.stats["hits"] + self.stats["misses"]
        hit_rate = self.stats["hits"] / total_requests if total_requests > 0 else 0.0
        
        return {
            "size": len(self.entries),
            "max_size": self.max_size,
            "strategy": self.strategy.value,
            "hits": self.stats["hits"],
            "misses": self.stats["misses"],
            "hit_rate": hit_rate,
            "evictions": self.stats["evictions"],
            "sets": self.stats["sets"]
        }
    
    async def cached(
        self,
        func: Callable[..., Awaitable[T]],
        ttl: Optional[float] = None,
        key_func: Optional[Callable[..., str]] = None
    ) -> Callable[..., Awaitable[T]]:
        """
        Decorator for caching function results.
        
        Args:
            func: Function to cache
            ttl: Optional TTL
            key_func: Optional key generation function
            
        Returns:
            Cached function
        """
        async def wrapper(*args, **kwargs):
            if key_func:
                key = key_func(*args, **kwargs)
            else:
                key = self._generate_key(*args, **kwargs)
            
            # Try to get from cache
            cached_value = await self.get(key)
            if cached_value is not None:
                return cached_value
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Cache result
            await self.set(key, result, ttl=ttl)
            
            return result
        
        return wrapper

