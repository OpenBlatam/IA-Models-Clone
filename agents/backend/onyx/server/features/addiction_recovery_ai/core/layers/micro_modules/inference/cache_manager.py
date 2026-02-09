"""
Cache Manager - Ultra-Specific Caching
Separated into its own file for maximum modularity
"""

import torch
import logging
from typing import Dict, Any, Optional, Hashable
from abc import ABC, abstractmethod
from collections import OrderedDict

logger = logging.getLogger(__name__)


class CacheManagerBase(ABC):
    """Base class for cache managers"""
    
    def __init__(self, name: str = "CacheManager"):
        self.name = name
    
    @abstractmethod
    def get(self, key: Hashable) -> Optional[Any]:
        """Get value from cache"""
        pass
    
    @abstractmethod
    def set(self, key: Hashable, value: Any) -> None:
        """Set value in cache"""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear cache"""
        pass


class LRUCacheManager(CacheManagerBase):
    """LRU (Least Recently Used) cache manager"""
    
    def __init__(self, max_size: int = 100):
        super().__init__("LRUCacheManager")
        self.max_size = max_size
        self.cache: OrderedDict = OrderedDict()
    
    def get(self, key: Hashable) -> Optional[Any]:
        """Get value from LRU cache"""
        if key in self.cache:
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            return self.cache[key]
        return None
    
    def set(self, key: Hashable, value: Any) -> None:
        """Set value in LRU cache"""
        if key in self.cache:
            # Update existing
            self.cache.move_to_end(key)
        else:
            # Add new
            if len(self.cache) >= self.max_size:
                # Remove least recently used
                self.cache.popitem(last=False)
        
        self.cache[key] = value
    
    def clear(self) -> None:
        """Clear LRU cache"""
        self.cache.clear()
    
    def size(self) -> int:
        """Get cache size"""
        return len(self.cache)


class FIFOCacheManager(CacheManagerBase):
    """FIFO (First In First Out) cache manager"""
    
    def __init__(self, max_size: int = 100):
        super().__init__("FIFOCacheManager")
        self.max_size = max_size
        self.cache: OrderedDict = OrderedDict()
    
    def get(self, key: Hashable) -> Optional[Any]:
        """Get value from FIFO cache"""
        return self.cache.get(key)
    
    def set(self, key: Hashable, value: Any) -> None:
        """Set value in FIFO cache"""
        if key not in self.cache:
            if len(self.cache) >= self.max_size:
                # Remove first item
                self.cache.popitem(last=False)
        
        self.cache[key] = value
    
    def clear(self) -> None:
        """Clear FIFO cache"""
        self.cache.clear()
    
    def size(self) -> int:
        """Get cache size"""
        return len(self.cache)


class TTLCacheManager(CacheManagerBase):
    """TTL (Time To Live) cache manager"""
    
    def __init__(self, max_size: int = 100, ttl: float = 3600.0):
        super().__init__("TTLCacheManager")
        self.max_size = max_size
        self.ttl = ttl
        self.cache: Dict[Hashable, tuple] = {}  # (value, timestamp)
        import time
        self.time = time
    
    def get(self, key: Hashable) -> Optional[Any]:
        """Get value from TTL cache"""
        if key in self.cache:
            value, timestamp = self.cache[key]
            current_time = self.time.time()
            
            if current_time - timestamp < self.ttl:
                return value
            else:
                # Expired
                del self.cache[key]
        
        return None
    
    def set(self, key: Hashable, value: Any) -> None:
        """Set value in TTL cache"""
        current_time = self.time.time()
        
        if len(self.cache) >= self.max_size:
            # Remove expired entries first
            expired_keys = [
                k for k, (v, t) in self.cache.items()
                if current_time - t >= self.ttl
            ]
            for k in expired_keys:
                del self.cache[k]
            
            # If still full, remove oldest
            if len(self.cache) >= self.max_size:
                oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][1])
                del self.cache[oldest_key]
        
        self.cache[key] = (value, current_time)
    
    def clear(self) -> None:
        """Clear TTL cache"""
        self.cache.clear()
    
    def size(self) -> int:
        """Get cache size"""
        return len(self.cache)


class NoCacheManager(CacheManagerBase):
    """No-op cache manager (always returns None)"""
    
    def __init__(self):
        super().__init__("NoCacheManager")
    
    def get(self, key: Hashable) -> Optional[Any]:
        """Get value (always None)"""
        return None
    
    def set(self, key: Hashable, value: Any) -> None:
        """Set value (no-op)"""
        pass
    
    def clear(self) -> None:
        """Clear cache (no-op)"""
        pass


# Factory for cache managers
class CacheManagerFactory:
    """Factory for creating cache managers"""
    
    _registry = {
        'lru': LRUCacheManager,
        'fifo': FIFOCacheManager,
        'ttl': TTLCacheManager,
        'none': NoCacheManager,
    }
    
    @classmethod
    def create(cls, manager_type: str, **kwargs) -> CacheManagerBase:
        """Create cache manager"""
        manager_type = manager_type.lower()
        if manager_type not in cls._registry:
            raise ValueError(f"Unknown cache manager type: {manager_type}")
        return cls._registry[manager_type](**kwargs)
    
    @classmethod
    def register(cls, name: str, manager_class: type):
        """Register custom cache manager"""
        cls._registry[name.lower()] = manager_class


__all__ = [
    "CacheManagerBase",
    "LRUCacheManager",
    "FIFOCacheManager",
    "TTLCacheManager",
    "NoCacheManager",
    "CacheManagerFactory",
]



