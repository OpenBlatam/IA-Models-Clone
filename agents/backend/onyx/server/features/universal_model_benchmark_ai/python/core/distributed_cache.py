"""
Distributed Cache Module - Distributed caching system.

Provides:
- Distributed cache
- Cache invalidation
- TTL support
- Cache statistics
- Multi-backend support
"""

import logging
import time
import hashlib
import json
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from threading import Lock
from enum import Enum

logger = logging.getLogger(__name__)


class CacheBackend(str, Enum):
    """Cache backend types."""
    MEMORY = "memory"
    REDIS = "redis"
    MEMCACHED = "memcached"


@dataclass
class CacheEntry:
    """Cache entry."""
    key: str
    value: Any
    ttl: Optional[float] = None
    created_at: float = field(default_factory=time.time)
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    
    def is_expired(self) -> bool:
        """Check if entry is expired."""
        if self.ttl is None:
            return False
        return (time.time() - self.created_at) > self.ttl
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "key": self.key,
            "ttl": self.ttl,
            "created_at": self.created_at,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed,
        }


class MemoryCache:
    """In-memory cache backend."""
    
    def __init__(self, max_size: int = 1000):
        """
        Initialize memory cache.
        
        Args:
            max_size: Maximum cache size
        """
        self.cache: Dict[str, CacheEntry] = {}
        self.max_size = max_size
        self.lock = Lock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self.lock:
            entry = self.cache.get(key)
            if entry is None:
                return None
            
            if entry.is_expired():
                del self.cache[key]
                return None
            
            entry.access_count += 1
            entry.last_accessed = time.time()
            return entry.value
    
    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Set value in cache."""
        with self.lock:
            # Evict if needed
            if len(self.cache) >= self.max_size and key not in self.cache:
                self._evict_lru()
            
            self.cache[key] = CacheEntry(key=key, value=value, ttl=ttl)
    
    def delete(self, key: str) -> None:
        """Delete key from cache."""
        with self.lock:
            self.cache.pop(key, None)
    
    def clear(self) -> None:
        """Clear all cache."""
        with self.lock:
            self.cache.clear()
    
    def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if not self.cache:
            return
        
        lru_key = min(self.cache.keys(), key=lambda k: self.cache[k].last_accessed)
        del self.cache[lru_key]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "entries": [e.to_dict() for e in self.cache.values()],
            }


class DistributedCache:
    """Distributed cache manager."""
    
    def __init__(self, backend: CacheBackend = CacheBackend.MEMORY, **kwargs):
        """
        Initialize distributed cache.
        
        Args:
            backend: Cache backend
            **kwargs: Backend-specific options
        """
        self.backend_type = backend
        
        if backend == CacheBackend.MEMORY:
            self.backend = MemoryCache(max_size=kwargs.get("max_size", 1000))
        elif backend == CacheBackend.REDIS:
            # Redis backend would be implemented here
            try:
                import redis
                self.backend = redis.Redis(**kwargs)
            except ImportError:
                logger.warning("Redis not available, falling back to memory cache")
                self.backend = MemoryCache(max_size=kwargs.get("max_size", 1000))
        else:
            self.backend = MemoryCache(max_size=kwargs.get("max_size", 1000))
        
        self.stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "deletes": 0,
        }
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None
        """
        value = self.backend.get(key)
        
        if value is not None:
            self.stats["hits"] += 1
        else:
            self.stats["misses"] += 1
        
        return value
    
    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
        """
        if self.backend_type == CacheBackend.REDIS:
            self.backend.setex(key, int(ttl) if ttl else None, json.dumps(value))
        else:
            self.backend.set(key, value, ttl)
        
        self.stats["sets"] += 1
    
    def delete(self, key: str) -> None:
        """
        Delete key from cache.
        
        Args:
            key: Cache key
        """
        self.backend.delete(key)
        self.stats["deletes"] += 1
    
    def clear(self) -> None:
        """Clear all cache."""
        self.backend.clear()
        self.stats = {"hits": 0, "misses": 0, "sets": 0, "deletes": 0}
    
    def get_or_set(self, key: str, func: Callable, ttl: Optional[float] = None) -> Any:
        """
        Get value or set if not exists.
        
        Args:
            key: Cache key
            func: Function to call if key not found
            ttl: Time to live
            
        Returns:
            Cached or computed value
        """
        value = self.get(key)
        if value is not None:
            return value
        
        value = func()
        self.set(key, value, ttl)
        return value
    
    def invalidate_pattern(self, pattern: str) -> int:
        """
        Invalidate keys matching pattern.
        
        Args:
            pattern: Pattern to match
            
        Returns:
            Number of keys invalidated
        """
        # Simple pattern matching (can be enhanced)
        import re
        regex = re.compile(pattern.replace("*", ".*"))
        
        if hasattr(self.backend, "keys"):
            keys = self.backend.keys()
            count = 0
            for key in keys:
                if regex.match(key):
                    self.delete(key)
                    count += 1
            return count
        
        return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        stats = self.stats.copy()
        
        if hasattr(self.backend, "get_stats"):
            backend_stats = self.backend.get_stats()
            stats.update(backend_stats)
        
        total_requests = stats["hits"] + stats["misses"]
        stats["hit_rate"] = stats["hits"] / total_requests if total_requests > 0 else 0.0
        
        return stats












