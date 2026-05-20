"""
Caching utilities for the Blaze AI module.

This module provides various caching implementations including LRU cache,
TTL cache, and distributed cache support.
"""

from __future__ import annotations

import time
import threading
from collections import OrderedDict
from typing import Any, Dict, Optional, Generic, TypeVar, Callable
from abc import ABC, abstractmethod
import json
import hashlib

K = TypeVar("K")
V = TypeVar("V")


class Cache(ABC, Generic[K, V]):
    """Abstract base class for cache implementations."""
    
    @abstractmethod
    def get(self, key: K) -> Optional[V]:
        """Get value from cache."""
        pass
    
    @abstractmethod
    def set(self, key: K, value: V) -> None:
        """Set value in cache."""
        pass
    
    @abstractmethod
    def delete(self, key: K) -> bool:
        """Delete value from cache."""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all values from cache."""
        pass
    
    @abstractmethod
    def __len__(self) -> int:
        """Get number of items in cache."""
        pass


class LRUCache(Cache[K, V]):
    """
    Least Recently Used (LRU) cache implementation.
    
    Features:
    - Fixed capacity with LRU eviction
    - Thread-safe operations
    - Configurable capacity
    - Statistics tracking
    """
    
    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self._cache: OrderedDict[K, V] = OrderedDict()
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0
    
    def get(self, key: K) -> Optional[V]:
        """Get value from cache, updating access order."""
        with self._lock:
            if key in self._cache:
                # Move to end (most recently used)
                value = self._cache.pop(key)
                self._cache[key] = value
                self._hits += 1
                return value
            else:
                self._misses += 1
                return None
    
    def set(self, key: K, value: V) -> None:
        """Set value in cache, evicting LRU item if necessary."""
        with self._lock:
            if key in self._cache:
                # Update existing key
                self._cache.pop(key)
            elif len(self._cache) >= self.capacity:
                # Evict least recently used item
                self._cache.popitem(last=False)
            
            self._cache[key] = value
    
    def delete(self, key: K) -> bool:
        """Delete value from cache."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False
    
    def clear(self) -> None:
        """Clear all values from cache."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0
    
    def __len__(self) -> int:
        """Get number of items in cache."""
        return len(self._cache)
    
    def __contains__(self, key: K) -> bool:
        """Check if key exists in cache."""
        return key in self._cache
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self._hits + self._misses
        hit_rate = self._hits / total_requests if total_requests > 0 else 0.0
        
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
            "size": len(self._cache),
            "capacity": self.capacity,
            "utilization": len(self._cache) / self.capacity if self.capacity > 0 else 0.0
        }


class TTLCache(Cache[K, V]):
    """
    Time-To-Live (TTL) cache implementation.
    
    Features:
    - Automatic expiration of cached items
    - Configurable TTL per item or globally
    - Background cleanup of expired items
    - Thread-safe operations
    """
    
    def __init__(self, default_ttl: float = 3600.0, cleanup_interval: float = 60.0):
        self.default_ttl = default_ttl
        self.cleanup_interval = cleanup_interval
        self._cache: Dict[K, tuple[V, float]] = {}
        self._lock = threading.RLock()
        self._last_cleanup = time.time()
    
    def get(self, key: K) -> Optional[V]:
        """Get value from cache if not expired."""
        with self._lock:
            self._cleanup_if_needed()
            
            if key in self._cache:
                value, expiry = self._cache[key]
                if time.time() < expiry:
                    return value
                else:
                    # Remove expired item
                    del self._cache[key]
            return None
    
    def set(self, key: K, value: V, ttl: Optional[float] = None) -> None:
        """Set value in cache with optional TTL."""
        with self._lock:
            expiry = time.time() + (ttl or self.default_ttl)
            self._cache[key] = (value, expiry)
    
    def delete(self, key: K) -> bool:
        """Delete value from cache."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False
    
    def clear(self) -> None:
        """Clear all values from cache."""
        with self._lock:
            self._cache.clear()
    
    def __len__(self) -> int:
        """Get number of non-expired items in cache."""
        with self._lock:
            self._cleanup_if_needed()
            return len(self._cache)
    
    def _cleanup_if_needed(self) -> None:
        """Remove expired items if cleanup interval has passed."""
        current_time = time.time()
        if current_time - self._last_cleanup > self.cleanup_interval:
            self._cleanup_expired()
            self._last_cleanup = current_time
    
    def _cleanup_expired(self) -> None:
        """Remove all expired items from cache."""
        current_time = time.time()
        expired_keys = [
            key for key, (_, expiry) in self._cache.items()
            if current_time >= expiry
        ]
        for key in expired_keys:
            del self._cache[key]


class FunctionCache:
    """
    Function result cache with automatic key generation.
    
    Features:
    - Automatic cache key generation from function arguments
    - Configurable TTL and capacity
    - Support for both LRU and TTL caching strategies
    - Thread-safe operations
    """
    
    def __init__(self, capacity: int = 1000, ttl: Optional[float] = None):
        self.capacity = capacity
        self.ttl = ttl
        
        if ttl is not None:
            self._cache = TTLCache(default_ttl=ttl)
        else:
            self._cache = LRUCache(capacity=capacity)
    
    def __call__(self, func: Callable[..., V]) -> Callable[..., V]:
        """Decorator to cache function results."""
        def wrapper(*args, **kwargs) -> V:
            # Generate cache key from function name and arguments
            key = self._make_key(func.__name__, args, kwargs)
            
            # Try to get from cache
            cached_result = self._cache.get(key)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            self._cache.set(key, result)
            return result
        
        return wrapper
    
    def _make_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """Generate cache key from function name and arguments."""
        # Create a hashable representation of arguments
        args_repr = (func_name, args, tuple(sorted(kwargs.items())))
        return hashlib.md5(json.dumps(args_repr, sort_keys=True).encode()).hexdigest()
    
    def clear(self) -> None:
        """Clear all cached results."""
        self._cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if hasattr(self._cache, 'get_stats'):
            return self._cache.get_stats()
        return {"size": len(self._cache)}


class DistributedCache(Cache[K, V]):
    """
    Distributed cache interface for Redis or similar backends.
    
    This is a placeholder implementation that can be extended
    to support Redis, Memcached, or other distributed cache systems.
    """
    
    def __init__(self, connection_string: str, ttl: float = 3600.0):
        self.connection_string = connection_string
        self.ttl = ttl
        self._client = None  # Would be initialized with actual client
        self._lock = threading.RLock()
    
    def get(self, key: K) -> Optional[V]:
        """Get value from distributed cache."""
        # Placeholder implementation
        return None
    
    def set(self, key: K, value: V) -> None:
        """Set value in distributed cache."""
        # Placeholder implementation
        pass
    
    def delete(self, key: K) -> bool:
        """Delete value from distributed cache."""
        # Placeholder implementation
        return False
    
    def clear(self) -> None:
        """Clear all values from distributed cache."""
        # Placeholder implementation
        pass
    
    def __len__(self) -> int:
        """Get number of items in distributed cache."""
        # Placeholder implementation
        return 0


# Cache factory function
def create_cache(
    cache_type: str = "lru",
    capacity: int = 1000,
    ttl: Optional[float] = None,
    connection_string: Optional[str] = None
) -> Cache:
    """
    Factory function to create cache instances.
    
    Args:
        cache_type: Type of cache ("lru", "ttl", "distributed")
        capacity: Cache capacity (for LRU cache)
        ttl: Time-to-live in seconds (for TTL cache)
        connection_string: Connection string for distributed cache
        
    Returns:
        Cache instance
    """
    if cache_type == "lru":
        return LRUCache(capacity=capacity)
    elif cache_type == "ttl":
        return TTLCache(default_ttl=ttl or 3600.0)
    elif cache_type == "distributed":
        if not connection_string:
            raise ValueError("Connection string required for distributed cache")
        return DistributedCache(connection_string, ttl=ttl or 3600.0)
    else:
        raise ValueError(f"Unknown cache type: {cache_type}")


# Export main classes
__all__ = [
    "Cache",
    "LRUCache", 
    "TTLCache",
    "FunctionCache",
    "DistributedCache",
    "create_cache"
]


