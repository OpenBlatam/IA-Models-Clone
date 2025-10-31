"""
Caching system for Export IA.
"""

import asyncio
import hashlib
import json
import time
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    value: Any
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.now)
    size_bytes: int = 0


class MemoryCache:
    """In-memory cache with TTL and LRU eviction."""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: Dict[str, CacheEntry] = {}
        self._lock = asyncio.Lock()
    
    def _generate_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments."""
        key_data = {
            'args': args,
            'kwargs': sorted(kwargs.items())
        }
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _calculate_size(self, value: Any) -> int:
        """Calculate approximate size of value in bytes."""
        try:
            return len(json.dumps(value, default=str).encode())
        except:
            return 1024  # Default size estimate
    
    def _is_expired(self, entry: CacheEntry) -> bool:
        """Check if cache entry is expired."""
        if entry.expires_at is None:
            return False
        return datetime.now() > entry.expires_at
    
    def _evict_lru(self):
        """Evict least recently used entries."""
        if len(self.cache) < self.max_size:
            return
        
        # Sort by last accessed time and access count
        sorted_entries = sorted(
            self.cache.items(),
            key=lambda x: (x[1].last_accessed, x[1].access_count)
        )
        
        # Remove oldest entries
        to_remove = len(self.cache) - self.max_size + 1
        for key, _ in sorted_entries[:to_remove]:
            del self.cache[key]
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        async with self._lock:
            if key not in self.cache:
                return None
            
            entry = self.cache[key]
            
            # Check if expired
            if self._is_expired(entry):
                del self.cache[key]
                return None
            
            # Update access metadata
            entry.access_count += 1
            entry.last_accessed = datetime.now()
            
            return entry.value
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache."""
        async with self._lock:
            ttl = ttl or self.default_ttl
            expires_at = datetime.now() + timedelta(seconds=ttl)
            size_bytes = self._calculate_size(value)
            
            entry = CacheEntry(
                key=key,
                value=value,
                expires_at=expires_at,
                size_bytes=size_bytes
            )
            
            self.cache[key] = entry
            self._evict_lru()
    
    async def delete(self, key: str) -> bool:
        """Delete value from cache."""
        async with self._lock:
            if key in self.cache:
                del self.cache[key]
                return True
            return False
    
    async def clear(self) -> None:
        """Clear all cache entries."""
        async with self._lock:
            self.cache.clear()
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        async with self._lock:
            total_size = sum(entry.size_bytes for entry in self.cache.values())
            total_accesses = sum(entry.access_count for entry in self.cache.values())
            
            return {
                "entries": len(self.cache),
                "max_size": self.max_size,
                "total_size_bytes": total_size,
                "total_accesses": total_accesses,
                "hit_rate": total_accesses / max(len(self.cache), 1)
            }


class CacheManager:
    """Centralized cache management for Export IA."""
    
    def __init__(self):
        self.caches: Dict[str, MemoryCache] = {}
        self._default_cache = MemoryCache()
    
    def get_cache(self, name: str = "default", **kwargs) -> MemoryCache:
        """Get or create a named cache."""
        if name not in self.caches:
            self.caches[name] = MemoryCache(**kwargs)
        return self.caches[name]
    
    async def cached_method(self, cache_name: str = "default", ttl: int = 3600):
        """Decorator for caching method results."""
        def decorator(func):
            async def wrapper(*args, **kwargs):
                cache = self.get_cache(cache_name)
                key = cache._generate_key(func.__name__, *args, **kwargs)
                
                # Try to get from cache
                result = await cache.get(key)
                if result is not None:
                    logger.debug(f"Cache hit for {func.__name__}")
                    return result
                
                # Execute function and cache result
                logger.debug(f"Cache miss for {func.__name__}")
                result = await func(*args, **kwargs)
                await cache.set(key, result, ttl)
                
                return result
            return wrapper
        return decorator
    
    async def get_all_stats(self) -> Dict[str, Any]:
        """Get statistics for all caches."""
        stats = {}
        for name, cache in self.caches.items():
            stats[name] = await cache.get_stats()
        return stats
    
    async def clear_all(self) -> None:
        """Clear all caches."""
        for cache in self.caches.values():
            await cache.clear()


# Global cache manager instance
_cache_manager: Optional[CacheManager] = None


def get_cache_manager() -> CacheManager:
    """Get the global cache manager instance."""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager




