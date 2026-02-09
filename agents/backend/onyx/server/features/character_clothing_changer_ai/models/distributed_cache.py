"""
Distributed Cache for Flux2 Clothing Changer
============================================

Distributed caching system with consistency.
"""

import time
import hashlib
import json
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from collections import OrderedDict
import logging

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry."""
    key: str
    value: Any
    timestamp: float
    ttl: float
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
    
    @property
    def is_expired(self) -> bool:
        """Check if entry is expired."""
        return time.time() - self.timestamp > self.ttl


class DistributedCache:
    """Distributed caching system."""
    
    def __init__(
        self,
        max_size: int = 10000,
        default_ttl: float = 3600.0,
    ):
        """
        Initialize distributed cache.
        
        Args:
            max_size: Maximum cache size
            default_ttl: Default time-to-live in seconds
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.tag_index: Dict[str, List[str]] = {}
    
    def _generate_key(self, key: str) -> str:
        """Generate cache key."""
        if isinstance(key, str):
            return hashlib.md5(key.encode()).hexdigest()
        return hashlib.md5(json.dumps(key, sort_keys=True).encode()).hexdigest()
    
    def get(
        self,
        key: str,
    ) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None
        """
        cache_key = self._generate_key(key)
        
        if cache_key not in self.cache:
            return None
        
        entry = self.cache[cache_key]
        
        if entry.is_expired:
            del self.cache[cache_key]
            # Remove from tag index
            for tag in entry.tags:
                if tag in self.tag_index and cache_key in self.tag_index[tag]:
                    self.tag_index[tag].remove(cache_key)
            return None
        
        # Move to end (LRU)
        self.cache.move_to_end(cache_key)
        
        return entry.value
    
    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[float] = None,
        tags: Optional[List[str]] = None,
    ) -> None:
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Optional time-to-live
            tags: Optional tags
        """
        cache_key = self._generate_key(key)
        ttl = ttl or self.default_ttl
        
        # Remove existing entry
        if cache_key in self.cache:
            old_entry = self.cache[cache_key]
            # Remove from tag index
            for tag in old_entry.tags:
                if tag in self.tag_index and cache_key in self.tag_index[tag]:
                    self.tag_index[tag].remove(cache_key)
            del self.cache[cache_key]
        
        # Create new entry
        entry = CacheEntry(
            key=cache_key,
            value=value,
            timestamp=time.time(),
            ttl=ttl,
            tags=tags or [],
        )
        
        self.cache[cache_key] = entry
        
        # Update tag index
        for tag in entry.tags:
            if tag not in self.tag_index:
                self.tag_index[tag] = []
            if cache_key not in self.tag_index[tag]:
                self.tag_index[tag].append(cache_key)
        
        # Evict if over size
        if len(self.cache) > self.max_size:
            oldest_key = next(iter(self.cache))
            self._evict(oldest_key)
    
    def _evict(self, cache_key: str) -> None:
        """Evict entry from cache."""
        if cache_key in self.cache:
            entry = self.cache[cache_key]
            # Remove from tag index
            for tag in entry.tags:
                if tag in self.tag_index and cache_key in self.tag_index[tag]:
                    self.tag_index[tag].remove(cache_key)
            del self.cache[cache_key]
    
    def delete(self, key: str) -> bool:
        """
        Delete value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if deleted
        """
        cache_key = self._generate_key(key)
        
        if cache_key in self.cache:
            self._evict(cache_key)
            return True
        return False
    
    def invalidate_by_tag(self, tag: str) -> int:
        """
        Invalidate entries by tag.
        
        Args:
            tag: Tag to invalidate
            
        Returns:
            Number of entries invalidated
        """
        if tag not in self.tag_index:
            return 0
        
        keys_to_evict = self.tag_index[tag].copy()
        for cache_key in keys_to_evict:
            self._evict(cache_key)
        
        return len(keys_to_evict)
    
    def clear(self) -> None:
        """Clear all cache entries."""
        self.cache.clear()
        self.tag_index.clear()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics."""
        expired_count = sum(1 for entry in self.cache.values() if entry.is_expired)
        
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "expired_entries": expired_count,
            "tags": len(self.tag_index),
            "total_tagged_entries": sum(len(keys) for keys in self.tag_index.values()),
        }


