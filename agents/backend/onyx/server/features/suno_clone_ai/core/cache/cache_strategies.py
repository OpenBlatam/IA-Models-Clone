"""
Cache Strategies

Different cache eviction strategies.
"""

import logging
import time
from typing import Any, Optional, Dict
from collections import OrderedDict
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class CacheStrategy(ABC):
    """Base cache strategy."""
    
    @abstractmethod
    def should_evict(self, cache: Dict[str, Any]) -> Optional[str]:
        """
        Determine key to evict.
        
        Args:
            cache: Cache dictionary
            
        Returns:
            Key to evict or None
        """
        pass


class LRUCache(CacheStrategy):
    """Least Recently Used cache strategy."""
    
    def __init__(self):
        """Initialize LRU cache."""
        self.access_order = OrderedDict()
    
    def access(self, key: str) -> None:
        """Record key access."""
        if key in self.access_order:
            self.access_order.move_to_end(key)
        else:
            self.access_order[key] = time.time()
    
    def should_evict(self, cache: Dict[str, Any]) -> Optional[str]:
        """Evict least recently used."""
        if self.access_order:
            return next(iter(self.access_order))
        return None


class FIFOCache(CacheStrategy):
    """First In First Out cache strategy."""
    
    def __init__(self):
        """Initialize FIFO cache."""
        self.insertion_order = []
    
    def insert(self, key: str) -> None:
        """Record key insertion."""
        if key not in self.insertion_order:
            self.insertion_order.append(key)
    
    def should_evict(self, cache: Dict[str, Any]) -> Optional[str]:
        """Evict first inserted."""
        if self.insertion_order:
            return self.insertion_order[0]
        return None


class TTLCache(CacheStrategy):
    """Time To Live cache strategy."""
    
    def __init__(self, ttl: float = 3600.0):
        """
        Initialize TTL cache.
        
        Args:
            ttl: Time to live in seconds
        """
        self.ttl = ttl
        self.expiry_times: Dict[str, float] = {}
    
    def set_expiry(self, key: str) -> None:
        """Set expiry time for key."""
        self.expiry_times[key] = time.time() + self.ttl
    
    def should_evict(self, cache: Dict[str, Any]) -> Optional[str]:
        """Evict expired keys."""
        now = time.time()
        
        for key, expiry in list(self.expiry_times.items()):
            if expiry < now:
                del self.expiry_times[key]
                return key
        
        return None



