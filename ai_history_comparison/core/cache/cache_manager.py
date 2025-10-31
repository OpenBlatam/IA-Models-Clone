"""
Cache Manager Implementation

Centralized cache management with multiple backends,
intelligent eviction strategies, and performance optimization.
"""

import asyncio
import logging
import pickle
import hashlib
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Callable, Type
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import threading
import weakref

logger = logging.getLogger(__name__)


class CacheBackendType(Enum):
    """Cache backend types"""
    MEMORY = "memory"
    REDIS = "redis"
    DISK = "disk"
    HYBRID = "hybrid"


class EvictionPolicy(Enum):
    """Cache eviction policies"""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    TTL = "ttl"  # Time To Live
    SIZE = "size"  # Size-based
    RANDOM = "random"


@dataclass
class CacheConfig:
    """Cache configuration"""
    backend_type: CacheBackendType = CacheBackendType.MEMORY
    max_size: int = 1000
    default_ttl: float = 3600.0  # 1 hour
    eviction_policy: EvictionPolicy = EvictionPolicy.LRU
    compression: bool = True
    serialization: str = "pickle"  # pickle, json, msgpack
    namespace: str = "default"
    key_prefix: str = ""
    auto_cleanup: bool = True
    cleanup_interval: float = 300.0  # 5 minutes
    metrics_enabled: bool = True
    custom_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    key: str
    value: Any
    created_at: datetime
    accessed_at: datetime
    access_count: int = 0
    ttl: Optional[float] = None
    size_bytes: int = 0
    tags: List[str] = field(default_factory=list)
    
    def is_expired(self) -> bool:
        """Check if entry is expired"""
        if self.ttl is None:
            return False
        
        age = (datetime.utcnow() - self.created_at).total_seconds()
        return age > self.ttl
    
    def update_access(self):
        """Update access information"""
        self.accessed_at = datetime.utcnow()
        self.access_count += 1
    
    def get_age(self) -> float:
        """Get entry age in seconds"""
        return (datetime.utcnow() - self.created_at).total_seconds()
    
    def get_idle_time(self) -> float:
        """Get idle time in seconds"""
        return (datetime.utcnow() - self.accessed_at).total_seconds()


class CacheBackend(ABC):
    """Abstract cache backend interface"""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self._lock = asyncio.Lock()
        self._metrics = CacheMetrics() if config.metrics_enabled else None
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        pass
    
    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[float] = None, 
                  tags: Optional[List[str]] = None) -> bool:
        """Set value in cache"""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete value from cache"""
        pass
    
    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        pass
    
    @abstractmethod
    async def clear(self) -> bool:
        """Clear all cache entries"""
        pass
    
    @abstractmethod
    async def keys(self, pattern: str = "*") -> List[str]:
        """Get cache keys matching pattern"""
        pass
    
    @abstractmethod
    async def size(self) -> int:
        """Get cache size"""
        pass
    
    @abstractmethod
    async def memory_usage(self) -> int:
        """Get memory usage in bytes"""
        pass
    
    def _serialize(self, value: Any) -> bytes:
        """Serialize value for storage"""
        if self.config.serialization == "pickle":
            return pickle.dumps(value)
        elif self.config.serialization == "json":
            import json
            return json.dumps(value).encode('utf-8')
        elif self.config.serialization == "msgpack":
            import msgpack
            return msgpack.packb(value)
        else:
            raise ValueError(f"Unsupported serialization: {self.config.serialization}")
    
    def _deserialize(self, data: bytes) -> Any:
        """Deserialize value from storage"""
        if self.config.serialization == "pickle":
            return pickle.loads(data)
        elif self.config.serialization == "json":
            import json
            return json.loads(data.decode('utf-8'))
        elif self.config.serialization == "msgpack":
            import msgpack
            return msgpack.unpackb(data, raw=False)
        else:
            raise ValueError(f"Unsupported serialization: {self.config.serialization}")
    
    def _create_key(self, key: str) -> str:
        """Create full cache key with prefix and namespace"""
        full_key = f"{self.config.namespace}:{self.config.key_prefix}{key}"
        return full_key
    
    def _record_metrics(self, operation: str, hit: bool = False, size: int = 0):
        """Record cache metrics"""
        if self._metrics:
            self._metrics.record_operation(operation, hit, size)


class MemoryCache(CacheBackend):
    """In-memory cache backend"""
    
    def __init__(self, config: CacheConfig):
        super().__init__(config)
        self._cache: Dict[str, CacheEntry] = {}
        self._access_order: List[str] = []
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False
    
    async def start(self):
        """Start cache backend"""
        self._running = True
        if self.config.auto_cleanup:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
    
    async def stop(self):
        """Stop cache backend"""
        self._running = False
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        full_key = self._create_key(key)
        
        async with self._lock:
            if full_key not in self._cache:
                self._record_metrics("get", hit=False)
                return None
            
            entry = self._cache[full_key]
            
            if entry.is_expired():
                del self._cache[full_key]
                if full_key in self._access_order:
                    self._access_order.remove(full_key)
                self._record_metrics("get", hit=False)
                return None
            
            entry.update_access()
            self._record_metrics("get", hit=True, size=entry.size_bytes)
            return entry.value
    
    async def set(self, key: str, value: Any, ttl: Optional[float] = None, 
                  tags: Optional[List[str]] = None) -> bool:
        """Set value in cache"""
        full_key = self._create_key(key)
        ttl = ttl or self.config.default_ttl
        
        async with self._lock:
            # Create cache entry
            entry = CacheEntry(
                key=full_key,
                value=value,
                created_at=datetime.utcnow(),
                accessed_at=datetime.utcnow(),
                ttl=ttl,
                size_bytes=len(self._serialize(value)),
                tags=tags or []
            )
            
            # Check size limit
            if len(self._cache) >= self.config.max_size:
                await self._evict_entries()
            
            self._cache[full_key] = entry
            if full_key not in self._access_order:
                self._access_order.append(full_key)
            
            self._record_metrics("set", size=entry.size_bytes)
            return True
    
    async def delete(self, key: str) -> bool:
        """Delete value from cache"""
        full_key = self._create_key(key)
        
        async with self._lock:
            if full_key in self._cache:
                del self._cache[full_key]
                if full_key in self._access_order:
                    self._access_order.remove(full_key)
                self._record_metrics("delete")
                return True
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        full_key = self._create_key(key)
        
        async with self._lock:
            if full_key not in self._cache:
                return False
            
            entry = self._cache[full_key]
            if entry.is_expired():
                del self._cache[full_key]
                if full_key in self._access_order:
                    self._access_order.remove(full_key)
                return False
            
            return True
    
    async def clear(self) -> bool:
        """Clear all cache entries"""
        async with self._lock:
            self._cache.clear()
            self._access_order.clear()
            self._record_metrics("clear")
            return True
    
    async def keys(self, pattern: str = "*") -> List[str]:
        """Get cache keys matching pattern"""
        import fnmatch
        
        async with self._lock:
            keys = []
            for key in self._cache.keys():
                # Remove namespace and prefix for return
                clean_key = key.replace(f"{self.config.namespace}:{self.config.key_prefix}", "")
                if fnmatch.fnmatch(clean_key, pattern):
                    keys.append(clean_key)
            return keys
    
    async def size(self) -> int:
        """Get cache size"""
        async with self._lock:
            return len(self._cache)
    
    async def memory_usage(self) -> int:
        """Get memory usage in bytes"""
        async with self._lock:
            return sum(entry.size_bytes for entry in self._cache.values())
    
    async def _evict_entries(self):
        """Evict entries based on policy"""
        if self.config.eviction_policy == EvictionPolicy.LRU:
            await self._evict_lru()
        elif self.config.eviction_policy == EvictionPolicy.LFU:
            await self._evict_lfu()
        elif self.config.eviction_policy == EvictionPolicy.TTL:
            await self._evict_ttl()
        elif self.config.eviction_policy == EvictionPolicy.SIZE:
            await self._evict_size()
        else:  # RANDOM
            await self._evict_random()
    
    async def _evict_lru(self):
        """Evict least recently used entries"""
        # Remove oldest entries
        to_remove = len(self._cache) - self.config.max_size + 1
        for _ in range(to_remove):
            if self._access_order:
                oldest_key = self._access_order.pop(0)
                if oldest_key in self._cache:
                    del self._cache[oldest_key]
    
    async def _evict_lfu(self):
        """Evict least frequently used entries"""
        # Sort by access count and remove least used
        entries = sorted(self._cache.items(), key=lambda x: x[1].access_count)
        to_remove = len(self._cache) - self.config.max_size + 1
        
        for i in range(to_remove):
            if i < len(entries):
                key = entries[i][0]
                del self._cache[key]
                if key in self._access_order:
                    self._access_order.remove(key)
    
    async def _evict_ttl(self):
        """Evict expired entries"""
        expired_keys = []
        for key, entry in self._cache.items():
            if entry.is_expired():
                expired_keys.append(key)
        
        for key in expired_keys:
            del self._cache[key]
            if key in self._access_order:
                self._access_order.remove(key)
    
    async def _evict_size(self):
        """Evict largest entries"""
        # Sort by size and remove largest
        entries = sorted(self._cache.items(), key=lambda x: x[1].size_bytes, reverse=True)
        to_remove = len(self._cache) - self.config.max_size + 1
        
        for i in range(to_remove):
            if i < len(entries):
                key = entries[i][0]
                del self._cache[key]
                if key in self._access_order:
                    self._access_order.remove(key)
    
    async def _evict_random(self):
        """Evict random entries"""
        import random
        
        to_remove = len(self._cache) - self.config.max_size + 1
        keys = list(self._cache.keys())
        
        for _ in range(to_remove):
            if keys:
                key = random.choice(keys)
                del self._cache[key]
                if key in self._access_order:
                    self._access_order.remove(key)
                keys.remove(key)
    
    async def _cleanup_loop(self):
        """Background cleanup loop"""
        while self._running:
            try:
                await asyncio.sleep(self.config.cleanup_interval)
                await self._cleanup_expired()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cache cleanup error: {e}")
    
    async def _cleanup_expired(self):
        """Clean up expired entries"""
        async with self._lock:
            expired_keys = []
            for key, entry in self._cache.items():
                if entry.is_expired():
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self._cache[key]
                if key in self._access_order:
                    self._access_order.remove(key)
            
            if expired_keys:
                logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")


class CacheManager:
    """Centralized cache manager"""
    
    def __init__(self, config: Optional[CacheConfig] = None):
        self.config = config or CacheConfig()
        self._backends: Dict[str, CacheBackend] = {}
        self._default_backend: Optional[str] = None
        self._lock = asyncio.Lock()
    
    async def start(self):
        """Start cache manager"""
        # Create default backend
        await self.create_backend("default", self.config)
        self._default_backend = "default"
    
    async def stop(self):
        """Stop cache manager"""
        for backend in self._backends.values():
            if hasattr(backend, 'stop'):
                await backend.stop()
        self._backends.clear()
    
    async def create_backend(self, name: str, config: CacheConfig) -> CacheBackend:
        """Create a cache backend"""
        if config.backend_type == CacheBackendType.MEMORY:
            backend = MemoryCache(config)
        elif config.backend_type == CacheBackendType.REDIS:
            # Redis backend would be implemented here
            raise NotImplementedError("Redis backend not implemented yet")
        elif config.backend_type == CacheBackendType.DISK:
            # Disk backend would be implemented here
            raise NotImplementedError("Disk backend not implemented yet")
        else:
            raise ValueError(f"Unsupported backend type: {config.backend_type}")
        
        if hasattr(backend, 'start'):
            await backend.start()
        
        self._backends[name] = backend
        return backend
    
    def get_backend(self, name: Optional[str] = None) -> CacheBackend:
        """Get cache backend"""
        backend_name = name or self._default_backend
        if not backend_name or backend_name not in self._backends:
            raise ValueError(f"Backend '{backend_name}' not found")
        return self._backends[backend_name]
    
    async def get(self, key: str, backend: Optional[str] = None) -> Optional[Any]:
        """Get value from cache"""
        cache_backend = self.get_backend(backend)
        return await cache_backend.get(key)
    
    async def set(self, key: str, value: Any, ttl: Optional[float] = None,
                  tags: Optional[List[str]] = None, backend: Optional[str] = None) -> bool:
        """Set value in cache"""
        cache_backend = self.get_backend(backend)
        return await cache_backend.set(key, value, ttl, tags)
    
    async def delete(self, key: str, backend: Optional[str] = None) -> bool:
        """Delete value from cache"""
        cache_backend = self.get_backend(backend)
        return await cache_backend.delete(key)
    
    async def exists(self, key: str, backend: Optional[str] = None) -> bool:
        """Check if key exists in cache"""
        cache_backend = self.get_backend(backend)
        return await cache_backend.exists(key)
    
    async def clear(self, backend: Optional[str] = None) -> bool:
        """Clear cache"""
        cache_backend = self.get_backend(backend)
        return await cache_backend.clear()
    
    async def keys(self, pattern: str = "*", backend: Optional[str] = None) -> List[str]:
        """Get cache keys"""
        cache_backend = self.get_backend(backend)
        return await cache_backend.keys(pattern)
    
    def get_stats(self, backend: Optional[str] = None) -> Dict[str, Any]:
        """Get cache statistics"""
        cache_backend = self.get_backend(backend)
        return {
            "size": cache_backend.size(),
            "memory_usage": cache_backend.memory_usage(),
            "metrics": cache_backend._metrics.get_stats() if cache_backend._metrics else {}
        }


class CacheMetrics:
    """Cache performance metrics"""
    
    def __init__(self):
        self._operations = {
            "get": {"hits": 0, "misses": 0, "total": 0},
            "set": {"total": 0, "size": 0},
            "delete": {"total": 0},
            "clear": {"total": 0}
        }
        self._lock = threading.Lock()
    
    def record_operation(self, operation: str, hit: bool = False, size: int = 0):
        """Record cache operation"""
        with self._lock:
            if operation in self._operations:
                self._operations[operation]["total"] += 1
                if operation == "get":
                    if hit:
                        self._operations[operation]["hits"] += 1
                    else:
                        self._operations[operation]["misses"] += 1
                elif operation == "set":
                    self._operations[operation]["size"] += size
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            stats = {}
            for op, data in self._operations.items():
                stats[op] = data.copy()
                if op == "get" and data["total"] > 0:
                    stats[op]["hit_rate"] = data["hits"] / data["total"]
                else:
                    stats[op]["hit_rate"] = 0.0
            return stats
    
    def reset(self):
        """Reset metrics"""
        with self._lock:
            for data in self._operations.values():
                for key in data:
                    data[key] = 0


# Global cache manager instance
cache_manager = CacheManager()


# Convenience functions
async def get_cache(key: str, backend: Optional[str] = None) -> Optional[Any]:
    """Get value from cache"""
    return await cache_manager.get(key, backend)


async def set_cache(key: str, value: Any, ttl: Optional[float] = None,
                   tags: Optional[List[str]] = None, backend: Optional[str] = None) -> bool:
    """Set value in cache"""
    return await cache_manager.set(key, value, ttl, tags, backend)


async def delete_cache(key: str, backend: Optional[str] = None) -> bool:
    """Delete value from cache"""
    return await cache_manager.delete(key, backend)





















