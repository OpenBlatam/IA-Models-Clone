"""
Advanced Cache Strategy System
===============================

Advanced cache strategy system with multiple eviction policies and TTL management.
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import OrderedDict

logger = logging.getLogger(__name__)


class CacheEvictionPolicy(Enum):
    """Cache eviction policies."""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    FIFO = "fifo"  # First In First Out
    TTL = "ttl"  # Time To Live
    RANDOM = "random"


@dataclass
class CacheEntry:
    """Cache entry."""
    key: str
    value: Any
    created_at: datetime = field(default_factory=datetime.now)
    accessed_at: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    ttl: Optional[float] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_expired(self) -> bool:
        """Check if entry is expired."""
        if not self.ttl:
            return False
        age = (datetime.now() - self.created_at).total_seconds()
        return age > self.ttl
    
    def touch(self):
        """Update access information."""
        self.accessed_at = datetime.now()
        self.access_count += 1


class AdvancedCacheStrategy:
    """Advanced cache with multiple eviction strategies."""
    
    def __init__(
        self,
        max_size: int = 1000,
        eviction_policy: CacheEvictionPolicy = CacheEvictionPolicy.LRU,
        default_ttl: Optional[float] = None
    ):
        """
        Initialize advanced cache strategy.
        
        Args:
            max_size: Maximum cache size
            eviction_policy: Eviction policy
            default_ttl: Default TTL in seconds
        """
        self.max_size = max_size
        self.eviction_policy = eviction_policy
        self.default_ttl = default_ttl
        self.cache: Dict[str, CacheEntry] = {}
        self.access_order: OrderedDict = OrderedDict()  # For LRU
        self.access_frequency: Dict[str, int] = {}  # For LFU
        self.lock = asyncio.Lock()
    
    async def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None
        """
        async with self.lock:
            if key not in self.cache:
                return None
            
            entry = self.cache[key]
            
            # Check expiration
            if entry.is_expired():
                await self._remove(key)
                return None
            
            # Update access info
            entry.touch()
            self._update_access_info(key)
            
            return entry.value
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[float] = None,
        tags: Optional[List[str]] = None
    ):
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Optional TTL in seconds
            tags: Optional tags
        """
        async with self.lock:
            # Check if we need to evict
            if len(self.cache) >= self.max_size and key not in self.cache:
                await self._evict()
            
            entry = CacheEntry(
                key=key,
                value=value,
                ttl=ttl or self.default_ttl,
                tags=tags or []
            )
            
            self.cache[key] = entry
            self._update_access_info(key)
    
    async def _evict(self):
        """Evict entry based on policy."""
        if not self.cache:
            return
        
        if self.eviction_policy == CacheEvictionPolicy.LRU:
            # Remove least recently used
            if self.access_order:
                lru_key = next(iter(self.access_order))
                await self._remove(lru_key)
        
        elif self.eviction_policy == CacheEvictionPolicy.LFU:
            # Remove least frequently used
            if self.access_frequency:
                lfu_key = min(self.access_frequency.items(), key=lambda x: x[1])[0]
                await self._remove(lfu_key)
        
        elif self.eviction_policy == CacheEvictionPolicy.FIFO:
            # Remove oldest (first created)
            oldest_key = min(
                self.cache.items(),
                key=lambda x: x[1].created_at
            )[0]
            await self._remove(oldest_key)
        
        elif self.eviction_policy == CacheEvictionPolicy.TTL:
            # Remove expired entries first, then oldest
            expired_keys = [k for k, v in self.cache.items() if v.is_expired()]
            if expired_keys:
                await self._remove(expired_keys[0])
            else:
                oldest_key = min(
                    self.cache.items(),
                    key=lambda x: x[1].created_at
                )[0]
                await self._remove(oldest_key)
        
        else:  # RANDOM
            # Remove random entry
            import random
            random_key = random.choice(list(self.cache.keys()))
            await self._remove(random_key)
    
    async def _remove(self, key: str):
        """Remove entry from cache."""
        if key in self.cache:
            del self.cache[key]
        if key in self.access_order:
            del self.access_order[key]
        if key in self.access_frequency:
            del self.access_frequency[key]
    
    def _update_access_info(self, key: str):
        """Update access information for eviction policies."""
        # For LRU
        if key in self.access_order:
            self.access_order.move_to_end(key)
        else:
            self.access_order[key] = True
        
        # For LFU
        self.access_frequency[key] = self.access_frequency.get(key, 0) + 1
    
    async def delete(self, key: str):
        """
        Delete entry from cache.
        
        Args:
            key: Cache key
        """
        async with self.lock:
            await self._remove(key)
    
    async def clear(self):
        """Clear all cache entries."""
        async with self.lock:
            self.cache.clear()
            self.access_order.clear()
            self.access_frequency.clear()
    
    async def invalidate_by_tags(self, tags: List[str]):
        """
        Invalidate entries by tags.
        
        Args:
            tags: List of tags
        """
        async with self.lock:
            keys_to_remove = [
                key for key, entry in self.cache.items()
                if any(tag in entry.tags for tag in tags)
            ]
            
            for key in keys_to_remove:
                await self._remove(key)
    
    async def cleanup_expired(self):
        """Cleanup expired entries."""
        async with self.lock:
            expired_keys = [
                key for key, entry in self.cache.items()
                if entry.is_expired()
            ]
            
            for key in expired_keys:
                await self._remove(key)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Statistics dictionary
        """
        total_size = len(self.cache)
        expired_count = sum(1 for e in self.cache.values() if e.is_expired())
        
        return {
            "size": total_size,
            "max_size": self.max_size,
            "eviction_policy": self.eviction_policy.value,
            "expired_entries": expired_count,
            "hit_rate": 0.0  # Would need to track hits/misses
        }



