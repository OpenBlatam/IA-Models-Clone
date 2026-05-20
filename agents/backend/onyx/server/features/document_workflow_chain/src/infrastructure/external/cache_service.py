"""
Cache Service
=============

Advanced caching service with multiple backends and strategies.
"""

from __future__ import annotations
import asyncio
import json
import logging
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
import hashlib

from ...shared.events.event_bus import get_event_bus, DomainEvent, EventMetadata


logger = logging.getLogger(__name__)


@dataclass
class CacheConfig:
    """Cache configuration"""
    backend: str = "memory"  # memory, redis, memcached
    default_ttl: int = 300  # seconds
    max_size: int = 1000
    compression: bool = True
    serialization: str = "json"  # json, pickle
    key_prefix: str = "workflow:"
    redis_url: Optional[str] = None
    memcached_servers: Optional[List[str]] = None


@dataclass
class CacheEntry:
    """Cache entry"""
    key: str
    value: Any
    ttl: int
    created_at: datetime
    accessed_at: datetime
    access_count: int = 0
    
    def is_expired(self) -> bool:
        """Check if entry is expired"""
        if self.ttl <= 0:
            return False
        return datetime.utcnow() > self.created_at + timedelta(seconds=self.ttl)
    
    def touch(self):
        """Update access time and count"""
        self.accessed_at = datetime.utcnow()
        self.access_count += 1


class CacheBackend(ABC):
    """Abstract cache backend interface"""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get value by key"""
        pass
    
    @abstractmethod
    async def set(self, key: str, value: Any, ttl: int = 0) -> None:
        """Set value with TTL"""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete value by key"""
        pass
    
    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists"""
        pass
    
    @abstractmethod
    async def clear(self) -> None:
        """Clear all entries"""
        pass
    
    @abstractmethod
    async def keys(self, pattern: str = "*") -> List[str]:
        """Get keys matching pattern"""
        pass
    
    @abstractmethod
    async def size(self) -> int:
        """Get cache size"""
        pass


class MemoryCacheBackend(CacheBackend):
    """In-memory cache backend with LRU eviction"""
    
    def __init__(self, max_size: int = 1000):
        self._cache: Dict[str, CacheEntry] = {}
        self._max_size = max_size
        self._lock = asyncio.Lock()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value by key"""
        async with self._lock:
            if key not in self._cache:
                return None
            
            entry = self._cache[key]
            
            # Check expiration
            if entry.is_expired():
                del self._cache[key]
                return None
            
            # Update access info
            entry.touch()
            return entry.value
    
    async def set(self, key: str, value: Any, ttl: int = 0) -> None:
        """Set value with TTL"""
        async with self._lock:
            # Evict if at capacity
            if len(self._cache) >= self._max_size and key not in self._cache:
                await self._evict_lru()
            
            entry = CacheEntry(
                key=key,
                value=value,
                ttl=ttl,
                created_at=datetime.utcnow(),
                accessed_at=datetime.utcnow()
            )
            
            self._cache[key] = entry
    
    async def delete(self, key: str) -> bool:
        """Delete value by key"""
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists"""
        async with self._lock:
            if key not in self._cache:
                return False
            
            entry = self._cache[key]
            if entry.is_expired():
                del self._cache[key]
                return False
            
            return True
    
    async def clear(self) -> None:
        """Clear all entries"""
        async with self._lock:
            self._cache.clear()
    
    async def keys(self, pattern: str = "*") -> List[str]:
        """Get keys matching pattern"""
        async with self._lock:
            if pattern == "*":
                return list(self._cache.keys())
            
            # Simple pattern matching
            import fnmatch
            return [key for key in self._cache.keys() if fnmatch.fnmatch(key, pattern)]
    
    async def size(self) -> int:
        """Get cache size"""
        async with self._lock:
            return len(self._cache)
    
    async def _evict_lru(self) -> None:
        """Evict least recently used entry"""
        if not self._cache:
            return
        
        # Find LRU entry
        lru_key = min(
            self._cache.keys(),
            key=lambda k: self._cache[k].accessed_at
        )
        
        del self._cache[lru_key]


class RedisCacheBackend(CacheBackend):
    """Redis cache backend"""
    
    def __init__(self, redis_url: str):
        self.redis_url = redis_url
        self._client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Redis client"""
        try:
            import redis.asyncio as redis
            self._client = redis.from_url(self.redis_url)
        except ImportError:
            logger.error("Redis library not installed")
            raise
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value by key"""
        try:
            value = await self._client.get(key)
            if value is None:
                return None
            
            return json.loads(value)
        except Exception as e:
            logger.error(f"Redis get failed for key {key}: {e}")
            return None
    
    async def set(self, key: str, value: Any, ttl: int = 0) -> None:
        """Set value with TTL"""
        try:
            serialized_value = json.dumps(value)
            if ttl > 0:
                await self._client.setex(key, ttl, serialized_value)
            else:
                await self._client.set(key, serialized_value)
        except Exception as e:
            logger.error(f"Redis set failed for key {key}: {e}")
    
    async def delete(self, key: str) -> bool:
        """Delete value by key"""
        try:
            result = await self._client.delete(key)
            return result > 0
        except Exception as e:
            logger.error(f"Redis delete failed for key {key}: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists"""
        try:
            result = await self._client.exists(key)
            return result > 0
        except Exception as e:
            logger.error(f"Redis exists failed for key {key}: {e}")
            return False
    
    async def clear(self) -> None:
        """Clear all entries"""
        try:
            await self._client.flushdb()
        except Exception as e:
            logger.error(f"Redis clear failed: {e}")
    
    async def keys(self, pattern: str = "*") -> List[str]:
        """Get keys matching pattern"""
        try:
            keys = await self._client.keys(pattern)
            return [key.decode() if isinstance(key, bytes) else key for key in keys]
        except Exception as e:
            logger.error(f"Redis keys failed for pattern {pattern}: {e}")
            return []
    
    async def size(self) -> int:
        """Get cache size"""
        try:
            return await self._client.dbsize()
        except Exception as e:
            logger.error(f"Redis size failed: {e}")
            return 0


class CacheService:
    """
    Advanced caching service
    
    Provides multi-backend caching with TTL, compression, and statistics.
    """
    
    def __init__(self, config: Optional[CacheConfig] = None):
        self.config = config or CacheConfig()
        self._backend: Optional[CacheBackend] = None
        self._event_bus = get_event_bus()
        self._statistics = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "deletes": 0,
            "evictions": 0
        }
        self._initialize_backend()
    
    def _initialize_backend(self):
        """Initialize cache backend"""
        if self.config.backend == "memory":
            self._backend = MemoryCacheBackend(self.config.max_size)
        elif self.config.backend == "redis":
            if not self.config.redis_url:
                raise ValueError("Redis URL required for Redis backend")
            self._backend = RedisCacheBackend(self.config.redis_url)
        else:
            raise ValueError(f"Unknown cache backend: {self.config.backend}")
    
    def _get_full_key(self, key: str) -> str:
        """Get full cache key with prefix"""
        return f"{self.config.key_prefix}{key}"
    
    def _serialize_value(self, value: Any) -> Any:
        """Serialize value based on configuration"""
        if self.config.serialization == "json":
            return value  # JSON serialization handled by backend
        elif self.config.serialization == "pickle":
            return pickle.dumps(value)
        else:
            return value
    
    def _deserialize_value(self, value: Any) -> Any:
        """Deserialize value based on configuration"""
        if self.config.serialization == "pickle" and isinstance(value, bytes):
            return pickle.loads(value)
        return value
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value by key"""
        try:
            full_key = self._get_full_key(key)
            value = await self._backend.get(full_key)
            
            if value is not None:
                self._statistics["hits"] += 1
                value = self._deserialize_value(value)
                await self._publish_cache_hit_event(key)
            else:
                self._statistics["misses"] += 1
                await self._publish_cache_miss_event(key)
            
            return value
            
        except Exception as e:
            logger.error(f"Cache get failed for key {key}: {e}")
            self._statistics["misses"] += 1
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value with TTL"""
        try:
            full_key = self._get_full_key(key)
            ttl = ttl or self.config.default_ttl
            
            serialized_value = self._serialize_value(value)
            await self._backend.set(full_key, serialized_value, ttl)
            
            self._statistics["sets"] += 1
            await self._publish_cache_set_event(key, ttl)
            
        except Exception as e:
            logger.error(f"Cache set failed for key {key}: {e}")
    
    async def delete(self, key: str) -> bool:
        """Delete value by key"""
        try:
            full_key = self._get_full_key(key)
            result = await self._backend.delete(full_key)
            
            if result:
                self._statistics["deletes"] += 1
                await self._publish_cache_delete_event(key)
            
            return result
            
        except Exception as e:
            logger.error(f"Cache delete failed for key {key}: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists"""
        try:
            full_key = self._get_full_key(key)
            return await self._backend.exists(full_key)
        except Exception as e:
            logger.error(f"Cache exists failed for key {key}: {e}")
            return False
    
    async def clear(self) -> None:
        """Clear all entries"""
        try:
            await self._backend.clear()
            await self._publish_cache_clear_event()
        except Exception as e:
            logger.error(f"Cache clear failed: {e}")
    
    async def keys(self, pattern: str = "*") -> List[str]:
        """Get keys matching pattern"""
        try:
            full_pattern = f"{self.config.key_prefix}{pattern}"
            keys = await self._backend.keys(full_pattern)
            
            # Remove prefix from keys
            prefix_len = len(self.config.key_prefix)
            return [key[prefix_len:] for key in keys]
            
        except Exception as e:
            logger.error(f"Cache keys failed for pattern {pattern}: {e}")
            return []
    
    async def get_or_set(self, key: str, factory: callable, ttl: Optional[int] = None) -> Any:
        """Get value or set using factory function"""
        value = await self.get(key)
        if value is not None:
            return value
        
        # Generate value using factory
        value = await factory() if asyncio.iscoroutinefunction(factory) else factory()
        
        # Cache the value
        await self.set(key, value, ttl)
        return value
    
    async def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate all keys matching pattern"""
        try:
            keys = await self.keys(pattern)
            count = 0
            
            for key in keys:
                if await self.delete(key):
                    count += 1
            
            await self._publish_cache_invalidate_event(pattern, count)
            return count
            
        except Exception as e:
            logger.error(f"Cache invalidate pattern failed for {pattern}: {e}")
            return 0
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics"""
        try:
            total_requests = self._statistics["hits"] + self._statistics["misses"]
            hit_rate = (self._statistics["hits"] / total_requests * 100) if total_requests > 0 else 0
            
            return {
                **self._statistics,
                "hit_rate": hit_rate,
                "size": await self._backend.size(),
                "backend": self.config.backend,
                "config": {
                    "default_ttl": self.config.default_ttl,
                    "max_size": self.config.max_size,
                    "compression": self.config.compression,
                    "serialization": self.config.serialization,
                    "key_prefix": self.config.key_prefix
                }
            }
        except Exception as e:
            logger.error(f"Failed to get cache statistics: {e}")
            return self._statistics
    
    async def _publish_cache_hit_event(self, key: str):
        """Publish cache hit event"""
        event = DomainEvent(
            event_type="cache.hit",
            data={"key": key, "timestamp": datetime.utcnow().isoformat()},
            metadata=EventMetadata(source="cache_service", priority=4)  # LOW
        )
        await self._event_bus.publish(event)
    
    async def _publish_cache_miss_event(self, key: str):
        """Publish cache miss event"""
        event = DomainEvent(
            event_type="cache.miss",
            data={"key": key, "timestamp": datetime.utcnow().isoformat()},
            metadata=EventMetadata(source="cache_service", priority=4)  # LOW
        )
        await self._event_bus.publish(event)
    
    async def _publish_cache_set_event(self, key: str, ttl: int):
        """Publish cache set event"""
        event = DomainEvent(
            event_type="cache.set",
            data={"key": key, "ttl": ttl, "timestamp": datetime.utcnow().isoformat()},
            metadata=EventMetadata(source="cache_service", priority=4)  # LOW
        )
        await self._event_bus.publish(event)
    
    async def _publish_cache_delete_event(self, key: str):
        """Publish cache delete event"""
        event = DomainEvent(
            event_type="cache.delete",
            data={"key": key, "timestamp": datetime.utcnow().isoformat()},
            metadata=EventMetadata(source="cache_service", priority=4)  # LOW
        )
        await self._event_bus.publish(event)
    
    async def _publish_cache_clear_event(self):
        """Publish cache clear event"""
        event = DomainEvent(
            event_type="cache.clear",
            data={"timestamp": datetime.utcnow().isoformat()},
            metadata=EventMetadata(source="cache_service", priority=3)  # NORMAL
        )
        await self._event_bus.publish(event)
    
    async def _publish_cache_invalidate_event(self, pattern: str, count: int):
        """Publish cache invalidate event"""
        event = DomainEvent(
            event_type="cache.invalidate",
            data={"pattern": pattern, "count": count, "timestamp": datetime.utcnow().isoformat()},
            metadata=EventMetadata(source="cache_service", priority=3)  # NORMAL
        )
        await self._event_bus.publish(event)




