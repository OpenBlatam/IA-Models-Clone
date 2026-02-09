"""
Tag management module for cache invalidation
"""

from typing import Dict, List, Set, Optional
import asyncio
import fnmatch


class TagManager:
    """Manages cache tags for invalidation"""
    
    def __init__(self):
        """Initialize tag manager"""
        self._key_tags: Dict[str, List[str]] = {}
        self._tag_keys: Dict[str, Set[str]] = {}
        self._lock = asyncio.Lock()
    
    async def add_tags(self, key: str, tags: List[str]):
        """Add tags to a cache key
        
        Args:
            key: Cache key
            tags: List of tags
        """
        async with self._lock:
            self._key_tags[key] = tags
            for tag in tags:
                if tag not in self._tag_keys:
                    self._tag_keys[tag] = set()
                self._tag_keys[tag].add(key)
    
    async def remove_tags(self, key: str):
        """Remove tags from a cache key
        
        Args:
            key: Cache key
        """
        async with self._lock:
            if key in self._key_tags:
                tags = self._key_tags.pop(key)
                for tag in tags:
                    if tag in self._tag_keys:
                        self._tag_keys[tag].discard(key)
                        if not self._tag_keys[tag]:
                            del self._tag_keys[tag]
    
    async def get_keys_by_tag(self, tag: str) -> List[str]:
        """Get all keys with a specific tag
        
        Args:
            tag: Tag to search for
            
        Returns:
            List of keys with the tag
        """
        async with self._lock:
            if tag in self._tag_keys:
                return list(self._tag_keys[tag])
            return []
    
    async def get_keys_by_pattern(self, pattern: str, all_keys: List[str]) -> List[str]:
        """Get keys matching a pattern
        
        Args:
            pattern: Pattern to match
            all_keys: All available keys
            
        Returns:
            List of matching keys
        """
        return [k for k in all_keys if fnmatch.fnmatch(k, pattern)]
    
    async def get_tag_count(self) -> int:
        """Get total number of tags"""
        async with self._lock:
            return len(self._tag_keys)
    
    async def get_tagged_key_count(self) -> int:
        """Get total number of tagged keys"""
        async with self._lock:
            return len(self._key_tags)
    
    def get_tags_for_key(self, key: str) -> Optional[List[str]]:
        """Get tags for a specific key (synchronous for internal use)
        
        Args:
            key: Cache key
            
        Returns:
            List of tags or None
        """
        return self._key_tags.get(key)

