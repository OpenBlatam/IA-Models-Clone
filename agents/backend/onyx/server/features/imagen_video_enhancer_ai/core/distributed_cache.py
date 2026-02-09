"""
Distributed Cache
================

Distributed caching system with multiple backends.
"""

import asyncio
import logging
import json
import hashlib
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
from enum import Enum

logger = logging.getLogger(__name__)


class CacheBackend(Enum):
    """Cache backend type."""
    MEMORY = "memory"
    REDIS = "redis"
    MEMCACHED = "memcached"


class CacheBackendInterface(ABC):
    """Interface for cache backends."""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        pass
    
    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[float] = None):
        """Set value in cache."""
        pass
    
    @abstractmethod
    async def delete(self, key: str):
        """Delete key from cache."""
        pass
    
    @abstractmethod
    async def clear(self):
        """Clear all cache."""
        pass
    
    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        pass


class MemoryCacheBackend(CacheBackendInterface):
    """In-memory cache backend."""
    
    def __init__(self):
        """Initialize memory cache."""
        self._cache: Dict[str, tuple[Any, Optional[datetime]]] = {}
        self._lock = asyncio.Lock()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        async with self._lock:
            if key not in self._cache:
                return None
            
            value, expiry = self._cache[key]
            
            # Check expiry
            if expiry and datetime.now() > expiry:
                del self._cache[key]
                return None
            
            return value
    
    async def set(self, key: str, value: Any, ttl: Optional[float] = None):
        """Set value in cache."""
        async with self._lock:
            expiry = None
            if ttl:
                expiry = datetime.now() + timedelta(seconds=ttl)
            self._cache[key] = (value, expiry)
    
    async def delete(self, key: str):
        """Delete key from cache."""
        async with self._lock:
            self._cache.pop(key, None)
    
    async def clear(self):
        """Clear all cache."""
        async with self._lock:
            self._cache.clear()
    
    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        async with self._lock:
            if key not in self._cache:
                return False
            
            _, expiry = self._cache[key]
            if expiry and datetime.now() > expiry:
                del self._cache[key]
                return False
            
            return True


class DistributedCache:
    """Distributed cache with multiple backends."""
    
    def __init__(
        self,
        backend: CacheBackend = CacheBackend.MEMORY,
        default_ttl: float = 3600.0,
        key_prefix: str = ""
    ):
        """
        Initialize distributed cache.
        
        Args:
            backend: Cache backend type
            default_ttl: Default TTL in seconds
            key_prefix: Prefix for all keys
        """
        self.backend_type = backend
        self.default_ttl = default_ttl
        self.key_prefix = key_prefix
        
        # Initialize backend
        if backend == CacheBackend.MEMORY:
            self.backend = MemoryCacheBackend()
        else:
            raise ValueError(f"Backend not implemented: {backend}")
    
    def _make_key(self, key: str) -> str:
        """Make full cache key with prefix."""
        if self.key_prefix:
            return f"{self.key_prefix}:{key}"
        return key
    
    async def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None
        """
        full_key = self._make_key(key)
        return await self.backend.get(full_key)
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[float] = None
    ):
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (uses default if None)
        """
        full_key = self._make_key(key)
        ttl = ttl or self.default_ttl
        await self.backend.set(full_key, value, ttl)
    
    async def get_or_set(
        self,
        key: str,
        factory: Callable[[], Any],
        ttl: Optional[float] = None
    ) -> Any:
        """
        Get value or set if not exists.
        
        Args:
            key: Cache key
            factory: Function to generate value if not cached
            ttl: Time to live
            
        Returns:
            Cached or generated value
        """
        value = await self.get(key)
        if value is None:
            if asyncio.iscoroutinefunction(factory):
                value = await factory()
            else:
                value = factory()
            await self.set(key, value, ttl)
        return value
    
    async def delete(self, key: str):
        """Delete key from cache."""
        full_key = self._make_key(key)
        await self.backend.delete(full_key)
    
    async def clear(self):
        """Clear all cache."""
        await self.backend.clear()
    
    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        full_key = self._make_key(key)
        return await self.backend.exists(full_key)
    
    def make_key(self, *parts: str) -> str:
        """
        Make cache key from parts.
        
        Args:
            *parts: Key parts
            
        Returns:
            Combined key
        """
        return ":".join(str(part) for part in parts)
    
    def hash_key(self, data: Any) -> str:
        """
        Create hash key from data.
        
        Args:
            data: Data to hash
            
        Returns:
            Hash string
        """
        if isinstance(data, str):
            content = data
        else:
            content = json.dumps(data, sort_keys=True)
        
        return hashlib.md5(content.encode()).hexdigest()




