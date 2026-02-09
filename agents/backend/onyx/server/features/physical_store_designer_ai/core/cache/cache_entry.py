"""
Cache Entry

Represents a cache entry with metadata.
"""

import time
from typing import Any


class CacheEntry:
    """Represents a cache entry with metadata"""
    
    def __init__(self, value: Any, ttl: float, created_at: float):
        self.value = value
        self.ttl = ttl
        self.created_at = created_at
        self.access_count = 0
        self.last_accessed = created_at
    
    @property
    def expires_at(self) -> float:
        """Calculate expiration timestamp"""
        return self.created_at + self.ttl
    
    @property
    def is_expired(self) -> bool:
        """Check if entry is expired"""
        return time.time() > self.expires_at
    
    def access(self) -> Any:
        """Access the entry and update metadata"""
        self.access_count += 1
        self.last_accessed = time.time()
        return self.value




