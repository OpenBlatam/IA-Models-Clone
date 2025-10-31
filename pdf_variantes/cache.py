"""
Cache Management
================

Caching system for PDF Variantes to improve performance.
"""

import logging
from typing import Any, Optional, Callable, Dict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import hashlib
import json

logger = logging.getLogger(__name__)


class CachePolicy(str, Enum):
    """Cache eviction policies."""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    FIFO = "fifo"  # First In First Out
    TTL = "ttl"  # Time To Live


@dataclass
class CacheEntry:
    """A cache entry."""
    key: str
    value: Any
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_accessed: datetime = field(default_factory=datetime.utcnow)
    access_count: int = 1
    ttl_seconds: Optional[int] = None
    
    def is_expired(self) -> bool:
        """Check if entry is expired."""
        
        if self.ttl_seconds is None:
            return False
        
        age = (datetime.utcnow() - self.created_at).total_seconds()
        return age > self.ttl_seconds
    
    def update_access(self):
        """Update access statistics."""
        self.last_accessed = datetime.utcnow()
        self.access_count += 1
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
            "access_count": self.access_count,
            "ttl_seconds": self.ttl_seconds,
            "is_expired": self.is_expired()
        }


class CacheManager:
    """Cache manager."""
    
    def __init__(
        self,
        max_size: int = 1000,
        default_ttl: Optional[int] = None,
        policy: CachePolicy = CachePolicy.LRU
    ):
        self.cache: Dict[str, CacheEntry] = {}
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.policy = policy
        logger.info(f"Initialized Cache Manager (max_size={max_size}, policy={policy.value})")
    
    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ):
        """Set a cache entry."""
        
        # Evict if necessary
        if len(self.cache) >= self.max_size:
            self._evict()
        
        entry = CacheEntry(
            key=key,
            value=value,
            ttl_seconds=ttl or self.default_ttl
        )
        
        self.cache[key] = entry
        logger.debug(f"Cached: {key}")
    
    def get(self, key: str) -> Optional[Any]:
        """Get a cache entry."""
        
        entry = self.cache.get(key)
        
        if not entry:
            return None
        
        # Check if expired
        if entry.is_expired():
            del self.cache[key]
            logger.debug(f"Cache expired: {key}")
            return None
        
        # Update access
        entry.update_access()
        
        return entry.value
    
    def delete(self, key: str) -> bool:
        """Delete a cache entry."""
        
        if key in self.cache:
            del self.cache[key]
            logger.debug(f"Deleted from cache: {key}")
            return True
        
        return False
    
    def clear(self):
        """Clear all cache."""
        
        count = len(self.cache)
        self.cache.clear()
        logger.info(f"Cleared cache ({count} entries)")
    
    def exists(self, key: str) -> bool:
        """Check if key exists."""
        
        return key in self.cache and not self.cache[key].is_expired()
    
    def size(self) -> int:
        """Get cache size."""
        return len(self.cache)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        
        total_entries = len(self.cache)
        expired_entries = sum(1 for e in self.cache.values() if e.is_expired())
        
        if total_entries > 0:
            avg_access_count = sum(e.access_count for e in self.cache.values()) / total_entries
        else:
            avg_access_count = 0
        
        return {
            "total_entries": total_entries,
            "active_entries": total_entries - expired_entries,
            "expired_entries": expired_entries,
            "max_size": self.max_size,
            "utilization_percent": (total_entries / self.max_size * 100) if self.max_size > 0 else 0,
            "avg_access_count": avg_access_count,
            "policy": self.policy.value
        }
    
    def _evict(self):
        """Evict entries based on policy."""
        
        if not self.cache:
            return
        
        if self.policy == CachePolicy.LRU:
            # Remove least recently accessed
            key_to_remove = min(
                self.cache.keys(),
                key=lambda k: self.cache[k].last_accessed
            )
            del self.cache[key_to_remove]
            
        elif self.policy == CachePolicy.LFU:
            # Remove least frequently accessed
            key_to_remove = min(
                self.cache.keys(),
                key=lambda k: self.cache[k].access_count
            )
            del self.cache[key_to_remove]
            
        elif self.policy == CachePolicy.FIFO:
            # Remove oldest
            key_to_remove = min(
                self.cache.keys(),
                key=lambda k: self.cache[k].created_at
            )
            del self.cache[key_to_remove]
            
        elif self.policy == CachePolicy.TTL:
            # Remove expired
            expired_keys = [k for k, v in self.cache.items() if v.is_expired()]
            for key in expired_keys:
                del self.cache[key]
        
        logger.debug(f"Evicted entry: {key_to_remove if 'key_to_remove' in locals() else 'expired'}")
    
    def get_entries_info(self) -> List[Dict[str, Any]]:
        """Get information about all cache entries."""
        
        return [
            entry.to_dict()
            for entry in self.cache.values()
            if not entry.is_expired()
        ]
    
    def cleanup_expired(self):
        """Remove all expired entries."""
        
        expired_keys = [k for k, v in self.cache.items() if v.is_expired()]
        
        for key in expired_keys:
            del self.cache[key]
        
        if expired_keys:
            logger.info(f"Cleaned up {len(expired_keys)} expired entries")
    
    def get_or_set(
        self,
        key: str,
        func: Callable[[], Any],
        ttl: Optional[int] = None
    ) -> Any:
        """Get value from cache or compute and cache it."""
        
        value = self.get(key)
        
        if value is None:
            value = func()
            self.set(key, value, ttl)
        
        return value
