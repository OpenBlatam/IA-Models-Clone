"""
L1 Cache Layer - In-memory cache
"""

import asyncio
import time
from typing import Any, Optional, Dict
from dataclasses import dataclass, field

try:
    from cachetools import TTLCache
    CACHETOOLS_AVAILABLE = True
except ImportError:
    CACHETOOLS_AVAILABLE = False
    from collections import OrderedDict


@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    key: str
    value: Any
    created_at: float
    expires_at: float
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    size_bytes: int = 0
    compressed: bool = False


class L1Cache:
    """L1 in-memory cache layer"""
    
    def __init__(self, max_size: int = 1000, ttl: int = 300):
        """Initialize L1 cache
        
        Args:
            max_size: Maximum number of entries
            ttl: Time-to-live in seconds
        """
        if CACHETOOLS_AVAILABLE:
            self._cache: TTLCache = TTLCache(maxsize=max_size, ttl=ttl)
        else:
            self._cache: OrderedDict = OrderedDict()
        
        self.max_size = max_size
        self.ttl = ttl
        self._lock = asyncio.Lock()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None
        """
        async with self._lock:
            if CACHETOOLS_AVAILABLE:
                return self._cache.get(key)
            else:
                if key in self._cache:
                    entry = self._cache[key]
                    if entry.expires_at > time.time():
                        self._cache.move_to_end(key)
                        entry.access_count += 1
                        entry.last_accessed = time.time()
                        return entry.value
                    else:
                        del self._cache[key]
        return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (defaults to instance ttl)
            
        Returns:
            True if successful
        """
        ttl = ttl or self.ttl
        expires_at = time.time() + ttl
        
        async with self._lock:
            if CACHETOOLS_AVAILABLE:
                self._cache[key] = value
            else:
                while len(self._cache) >= self.max_size and key not in self._cache:
                    if self._cache:
                        oldest_key = next(iter(self._cache))
                        del self._cache[oldest_key]
                
                entry = CacheEntry(
                    key=key,
                    value=value,
                    created_at=time.time(),
                    expires_at=expires_at,
                    size_bytes=0
                )
                self._cache[key] = entry
                self._cache.move_to_end(key)
        return True
    
    async def delete(self, key: str) -> bool:
        """Delete key from cache
        
        Args:
            key: Cache key
            
        Returns:
            True if deleted, False if not found
        """
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
        return False
    
    async def clear(self):
        """Clear all entries"""
        async with self._lock:
            self._cache.clear()
    
    def size(self) -> int:
        """Get current cache size"""
        return len(self._cache)
    
    def keys(self) -> list:
        """Get all keys"""
        return list(self._cache.keys())
    
    async def evict_oldest(self) -> Optional[str]:
        """Evict oldest entry
        
        Returns:
            Evicted key or None
        """
        async with self._lock:
            if CACHETOOLS_AVAILABLE:
                # TTLCache handles eviction automatically, but we can still evict if needed
                if len(self._cache) >= self.max_size and self._cache:
                    # Get first key (oldest in TTLCache)
                    oldest_key = next(iter(self._cache))
                    del self._cache[oldest_key]
                    return oldest_key
            else:
                if self._cache:
                    oldest_key = next(iter(self._cache))
                    del self._cache[oldest_key]
                    return oldest_key
        return None

