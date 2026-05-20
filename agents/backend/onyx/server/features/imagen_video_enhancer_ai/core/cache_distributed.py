"""
Distributed Cache System
========================

Advanced distributed cache system with multiple backends and consistency guarantees.
"""

import asyncio
import logging
import hashlib
import json
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class CacheBackend(Enum):
    """Cache backend types."""
    MEMORY = "memory"
    REDIS = "redis"
    MEMCACHED = "memcached"
    FILE = "file"
    HYBRID = "hybrid"  # Multiple backends


class CacheConsistency(Enum):
    """Cache consistency levels."""
    EVENTUAL = "eventual"  # Eventual consistency
    STRONG = "strong"  # Strong consistency
    SESSION = "session"  # Session consistency


@dataclass
class CacheEntry:
    """Cache entry."""
    key: str
    value: Any
    ttl: Optional[int] = None
    created_at: datetime = field(default_factory=datetime.now)
    accessed_at: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_expired(self) -> bool:
        """Check if entry is expired."""
        if not self.ttl:
            return False
        age = (datetime.now() - self.created_at).total_seconds()
        return age > self.ttl
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "key": self.key,
            "value": self.value,
            "ttl": self.ttl,
            "created_at": self.created_at.isoformat(),
            "accessed_at": self.accessed_at.isoformat(),
            "access_count": self.access_count,
            "tags": self.tags,
            "metadata": self.metadata
        }


class DistributedCache:
    """Distributed cache with multiple backends."""
    
    def __init__(
        self,
        backends: List[CacheBackend],
        consistency: CacheConsistency = CacheConsistency.EVENTUAL,
        default_ttl: Optional[int] = None
    ):
        """
        Initialize distributed cache.
        
        Args:
            backends: List of cache backends
            consistency: Consistency level
            default_ttl: Default TTL in seconds
        """
        self.backends = backends
        self.consistency = consistency
        self.default_ttl = default_ttl
        self._memory_cache: Dict[str, CacheEntry] = {}
        self._backend_clients: Dict[CacheBackend, Any] = {}
        self._initialize_backends()
    
    def _initialize_backends(self):
        """Initialize cache backends."""
        for backend in self.backends:
            if backend == CacheBackend.MEMORY:
                # Memory backend is already initialized
                continue
            elif backend == CacheBackend.REDIS:
                try:
                    import redis.asyncio as redis
                    # Initialize Redis client (requires configuration)
                    # self._backend_clients[backend] = redis.Redis(...)
                    logger.info("Redis backend available")
                except ImportError:
                    logger.warning("Redis not available")
            elif backend == CacheBackend.MEMCACHED:
                try:
                    import aiomcache
                    # Initialize Memcached client (requires configuration)
                    # self._backend_clients[backend] = aiomcache.Client(...)
                    logger.info("Memcached backend available")
                except ImportError:
                    logger.warning("Memcached not available")
            elif backend == CacheBackend.FILE:
                # File backend uses local filesystem
                logger.info("File backend initialized")
    
    async def get(self, key: str, backend: Optional[CacheBackend] = None) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            backend: Optional specific backend
            
        Returns:
            Cached value or None
        """
        backends_to_check = [backend] if backend else self.backends
        
        for cache_backend in backends_to_check:
            try:
                value = await self._get_from_backend(key, cache_backend)
                if value is not None:
                    # Update access info
                    if cache_backend == CacheBackend.MEMORY and key in self._memory_cache:
                        entry = self._memory_cache[key]
                        entry.accessed_at = datetime.now()
                        entry.access_count += 1
                    return value
            except Exception as e:
                logger.warning(f"Error getting from {cache_backend.value}: {e}")
        
        return None
    
    async def _get_from_backend(self, key: str, backend: CacheBackend) -> Optional[Any]:
        """Get value from specific backend."""
        if backend == CacheBackend.MEMORY:
            if key in self._memory_cache:
                entry = self._memory_cache[key]
                if entry.is_expired():
                    del self._memory_cache[key]
                    return None
                return entry.value
            return None
        elif backend == CacheBackend.REDIS:
            # Redis implementation
            client = self._backend_clients.get(backend)
            if client:
                value = await client.get(key)
                return json.loads(value) if value else None
        elif backend == CacheBackend.MEMCACHED:
            # Memcached implementation
            client = self._backend_clients.get(backend)
            if client:
                value = await client.get(key.encode())
                return json.loads(value) if value else None
        elif backend == CacheBackend.FILE:
            # File implementation
            import os
            from pathlib import Path
            cache_dir = Path("cache")
            cache_file = cache_dir / f"{hashlib.md5(key.encode()).hexdigest()}.json"
            if cache_file.exists():
                with open(cache_file, 'r') as f:
                    entry_data = json.load(f)
                    entry = CacheEntry(**entry_data)
                    if entry.is_expired():
                        cache_file.unlink()
                        return None
                    return entry.value
        return None
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: TTL in seconds
            tags: Optional tags
            metadata: Optional metadata
        """
        ttl = ttl or self.default_ttl
        entry = CacheEntry(
            key=key,
            value=value,
            ttl=ttl,
            tags=tags or [],
            metadata=metadata or {}
        )
        
        # Set in all backends based on consistency
        if self.consistency == CacheConsistency.STRONG:
            # Write to all backends
            tasks = [self._set_in_backend(key, entry, backend) for backend in self.backends]
            await asyncio.gather(*tasks, return_exceptions=True)
        else:
            # Write to primary backend (first in list)
            if self.backends:
                await self._set_in_backend(key, entry, self.backends[0])
    
    async def _set_in_backend(self, key: str, entry: CacheEntry, backend: CacheBackend):
        """Set value in specific backend."""
        if backend == CacheBackend.MEMORY:
            self._memory_cache[key] = entry
        elif backend == CacheBackend.REDIS:
            client = self._backend_clients.get(backend)
            if client:
                value = json.dumps(entry.to_dict())
                await client.setex(key, entry.ttl or 0, value)
        elif backend == CacheBackend.MEMCACHED:
            client = self._backend_clients.get(backend)
            if client:
                value = json.dumps(entry.to_dict())
                await client.set(key.encode(), value.encode(), exptime=entry.ttl or 0)
        elif backend == CacheBackend.FILE:
            from pathlib import Path
            cache_dir = Path("cache")
            cache_dir.mkdir(exist_ok=True)
            cache_file = cache_dir / f"{hashlib.md5(key.encode()).hexdigest()}.json"
            with open(cache_file, 'w') as f:
                json.dump(entry.to_dict(), f)
    
    async def delete(self, key: str):
        """
        Delete value from cache.
        
        Args:
            key: Cache key
        """
        tasks = [self._delete_from_backend(key, backend) for backend in self.backends]
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _delete_from_backend(self, key: str, backend: CacheBackend):
        """Delete value from specific backend."""
        if backend == CacheBackend.MEMORY:
            self._memory_cache.pop(key, None)
        elif backend == CacheBackend.REDIS:
            client = self._backend_clients.get(backend)
            if client:
                await client.delete(key)
        elif backend == CacheBackend.MEMCACHED:
            client = self._backend_clients.get(backend)
            if client:
                await client.delete(key.encode())
        elif backend == CacheBackend.FILE:
            from pathlib import Path
            cache_dir = Path("cache")
            cache_file = cache_dir / f"{hashlib.md5(key.encode()).hexdigest()}.json"
            if cache_file.exists():
                cache_file.unlink()
    
    async def invalidate_by_tags(self, tags: List[str]):
        """
        Invalidate cache entries by tags.
        
        Args:
            tags: List of tags
        """
        # For memory backend
        keys_to_delete = []
        for key, entry in self._memory_cache.items():
            if any(tag in entry.tags for tag in tags):
                keys_to_delete.append(key)
        
        for key in keys_to_delete:
            await self.delete(key)
    
    async def clear(self):
        """Clear all cache."""
        self._memory_cache.clear()
        # Clear other backends
        for backend in self.backends:
            if backend != CacheBackend.MEMORY:
                await self._clear_backend(backend)
    
    async def _clear_backend(self, backend: CacheBackend):
        """Clear specific backend."""
        if backend == CacheBackend.REDIS:
            client = self._backend_clients.get(backend)
            if client:
                await client.flushdb()
        elif backend == CacheBackend.MEMCACHED:
            client = self._backend_clients.get(backend)
            if client:
                await client.flush_all()
        elif backend == CacheBackend.FILE:
            from pathlib import Path
            cache_dir = Path("cache")
            if cache_dir.exists():
                for cache_file in cache_dir.glob("*.json"):
                    cache_file.unlink()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Statistics dictionary
        """
        return {
            "backends": [backend.value for backend in self.backends],
            "consistency": self.consistency.value,
            "memory_entries": len(self._memory_cache),
            "memory_size_mb": sum(
                len(str(entry.value).encode()) for entry in self._memory_cache.values()
            ) / (1024 * 1024)
        }



