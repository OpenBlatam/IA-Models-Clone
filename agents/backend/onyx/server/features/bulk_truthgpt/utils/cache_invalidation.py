"""
Cache Invalidation
==================

Advanced cache invalidation strategies and patterns.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Set, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class InvalidationStrategy(str, Enum):
    """Cache invalidation strategies."""
    TTL = "ttl"
    TIME_BASED = "time_based"
    EVENT_BASED = "event_based"
    TAG_BASED = "tag_based"
    MANUAL = "manual"

@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    value: Any
    tags: Set[str]
    created_at: datetime
    expires_at: Optional[datetime] = None
    access_count: int = 0
    last_accessed: Optional[datetime] = None

class CacheInvalidationManager:
    """Advanced cache invalidation manager."""
    
    def __init__(self):
        self.cache_entries: Dict[str, CacheEntry] = {}
        self.tag_index: Dict[str, Set[str]] = {}
        self.invalidation_callbacks: List[Callable] = []
        self.cleanup_task = None
        self.is_running = False
    
    def register_entry(
        self,
        key: str,
        value: Any,
        tags: Optional[Set[str]] = None,
        ttl: Optional[float] = None
    ):
        """Register a cache entry."""
        now = datetime.now()
        expires_at = None
        
        if ttl:
            expires_at = now + timedelta(seconds=ttl)
        
        entry = CacheEntry(
            key=key,
            value=value,
            tags=tags or set(),
            created_at=now,
            expires_at=expires_at
        )
        
        self.cache_entries[key] = entry
        
        # Update tag index
        for tag in entry.tags:
            if tag not in self.tag_index:
                self.tag_index[tag] = set()
            self.tag_index[tag].add(key)
        
        logger.debug(f"Cache entry registered: {key}")
    
    def invalidate_key(self, key: str):
        """Invalidate a specific key."""
        if key in self.cache_entries:
            entry = self.cache_entries[key]
            
            # Remove from tag index
            for tag in entry.tags:
                if tag in self.tag_index:
                    self.tag_index[tag].discard(key)
                    if not self.tag_index[tag]:
                        del self.tag_index[tag]
            
            del self.cache_entries[key]
            logger.info(f"Cache entry invalidated: {key}")
            
            # Call callbacks
            for callback in self.invalidation_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        asyncio.create_task(callback(key, "key"))
                    else:
                        callback(key, "key")
                except Exception as e:
                    logger.error(f"Invalidation callback error: {e}")
    
    def invalidate_tag(self, tag: str):
        """Invalidate all entries with a tag."""
        if tag not in self.tag_index:
            return
        
        keys_to_invalidate = list(self.tag_index[tag])
        
        for key in keys_to_invalidate:
            self.invalidate_key(key)
        
        logger.info(f"Invalidated {len(keys_to_invalidate)} entries with tag: {tag}")
    
    def invalidate_tags(self, tags: Set[str]):
        """Invalidate all entries with any of the tags."""
        keys_to_invalidate = set()
        
        for tag in tags:
            if tag in self.tag_index:
                keys_to_invalidate.update(self.tag_index[tag])
        
        for key in keys_to_invalidate:
            self.invalidate_key(key)
        
        logger.info(f"Invalidated {len(keys_to_invalidate)} entries with tags: {tags}")
    
    def invalidate_pattern(self, pattern: str):
        """Invalidate entries matching a pattern."""
        import re
        
        regex = re.compile(pattern)
        keys_to_invalidate = [
            key for key in self.cache_entries.keys()
            if regex.match(key)
        ]
        
        for key in keys_to_invalidate:
            self.invalidate_key(key)
        
        logger.info(f"Invalidated {len(keys_to_invalidate)} entries matching pattern: {pattern}")
    
    def invalidate_all(self):
        """Invalidate all cache entries."""
        keys = list(self.cache_entries.keys())
        
        for key in keys:
            self.invalidate_key(key)
        
        logger.info(f"Invalidated all {len(keys)} cache entries")
    
    async def _cleanup_expired(self):
        """Cleanup expired entries."""
        now = datetime.now()
        expired_keys = [
            key for key, entry in self.cache_entries.items()
            if entry.expires_at and now > entry.expires_at
        ]
        
        for key in expired_keys:
            self.invalidate_key(key)
        
        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
    
    async def start_cleanup(self, interval: float = 60.0):
        """Start cleanup task."""
        if self.is_running:
            return
        
        self.is_running = True
        
        async def cleanup_loop():
            while self.is_running:
                try:
                    await self._cleanup_expired()
                    await asyncio.sleep(interval)
                except Exception as e:
                    logger.error(f"Cleanup loop error: {e}")
                    await asyncio.sleep(interval)
        
        self.cleanup_task = asyncio.create_task(cleanup_loop())
        logger.info("Cache invalidation cleanup started")
    
    async def stop_cleanup(self):
        """Stop cleanup task."""
        self.is_running = False
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
        logger.info("Cache invalidation cleanup stopped")
    
    def register_invalidation_callback(self, callback: Callable):
        """Register callback for invalidation events."""
        self.invalidation_callbacks.append(callback)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache invalidation statistics."""
        total_entries = len(self.cache_entries)
        expired_entries = sum(
            1 for entry in self.cache_entries.values()
            if entry.expires_at and datetime.now() > entry.expires_at
        )
        
        return {
            "total_entries": total_entries,
            "expired_entries": expired_entries,
            "active_entries": total_entries - expired_entries,
            "total_tags": len(self.tag_index),
            "tag_distribution": {
                tag: len(keys) for tag, keys in self.tag_index.items()
            }
        }

# Global instance
cache_invalidation = CacheInvalidationManager()
















