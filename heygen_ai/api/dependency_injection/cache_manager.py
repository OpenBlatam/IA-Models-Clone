from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import asyncio
import time
import json
import hashlib
from typing import Dict, List, Any, Optional, Union, Callable, Awaitable
from datetime import datetime, timezone, timedelta
from contextlib import asynccontextmanager
import structlog
from dataclasses import dataclass, asdict
from enum import Enum
import weakref
from functools import lru_cache, wraps
import inspect
import gc
        import redis.asyncio as redis
from typing import Any, List, Dict, Optional
import logging
#!/usr/bin/env python3
"""
Cache Manager for FastAPI Dependency Injection
Manages caching of dependencies and service instances.
"""


logger = structlog.get_logger()

# =============================================================================
# Cache Types
# =============================================================================

class CacheType(Enum):
    """Cache type enumeration."""
    MEMORY = "memory"
    REDIS = "redis"
    HYBRID = "hybrid"

class CacheStrategy(Enum):
    """Cache strategy enumeration."""
    LRU = "lru"
    LFU = "lfu"
    FIFO = "fifo"
    TTL = "ttl"

@dataclass
class CacheConfig:
    """Cache configuration."""
    cache_type: CacheType = CacheType.MEMORY
    strategy: CacheStrategy = CacheStrategy.LRU
    max_size: int = 1000
    default_ttl: int = 300
    enable_compression: bool = False
    enable_serialization: bool = True
    enable_monitoring: bool = True
    enable_metrics: bool = True
    cleanup_interval: float = 60.0
    max_memory_mb: int = 100

@dataclass
class CacheStats:
    """Cache statistics."""
    hits: int = 0
    misses: int = 0
    sets: int = 0
    deletes: int = 0
    evictions: int = 0
    total_size: int = 0
    memory_usage_mb: float = 0.0
    hit_rate: float = 0.0
    last_cleanup: Optional[datetime] = None

# =============================================================================
# Base Cache Classes
# =============================================================================

class CacheBase:
    """Base class for all cache implementations."""
    
    def __init__(self, config: CacheConfig):
        
    """__init__ function."""
self.config = config
        self.stats = CacheStats()
        self._cache: Dict[str, Any] = {}
        self._access_times: Dict[str, datetime] = {}
        self._cleanup_task: Optional[asyncio.Task] = None
        self._is_initialized = False
    
    async def initialize(self) -> None:
        """Initialize the cache."""
        if self._is_initialized:
            return
        
        try:
            await self._initialize_internal()
            self._is_initialized = True
            
            # Start cleanup task if monitoring is enabled
            if self.config.enable_monitoring:
                self._start_cleanup_task()
            
            logger.info(f"Cache {self.__class__.__name__} initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize cache {self.__class__.__name__}: {e}")
            raise
    
    async def _initialize_internal(self) -> None:
        """Internal initialization method to be implemented by subclasses."""
        pass
    
    async def cleanup(self) -> None:
        """Cleanup the cache."""
        if not self._is_initialized:
            return
        
        # Stop cleanup task
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        try:
            await self._cleanup_internal()
            self._is_initialized = False
            logger.info(f"Cache {self.__class__.__name__} cleaned up successfully")
            
        except Exception as e:
            logger.error(f"Failed to cleanup cache {self.__class__.__name__}: {e}")
            raise
    
    async def _cleanup_internal(self) -> None:
        """Internal cleanup method to be implemented by subclasses."""
        pass
    
    def _generate_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments."""
        key_data = {
            "args": str(args),
            "kwargs": str(sorted(kwargs.items()))
        }
        return hashlib.md5(json.dumps(key_data, sort_keys=True).encode()).hexdigest()
    
    def _update_stats(self, hit: bool = False, set: bool = False, delete: bool = False):
        """Update cache statistics."""
        if hit:
            self.stats.hits += 1
        else:
            self.stats.misses += 1
        
        if set:
            self.stats.sets += 1
        
        if delete:
            self.stats.deletes += 1
        
        # Calculate hit rate
        total_requests = self.stats.hits + self.stats.misses
        if total_requests > 0:
            self.stats.hit_rate = self.stats.hits / total_requests
    
    def _start_cleanup_task(self) -> None:
        """Start cleanup task."""
        if self._cleanup_task:
            return
        
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
    
    async def _cleanup_loop(self) -> None:
        """Cleanup loop."""
        while self._is_initialized:
            try:
                await self._perform_cleanup()
                self.stats.last_cleanup = datetime.now(timezone.utc)
                
                await asyncio.sleep(self.config.cleanup_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cache cleanup failed: {e}")
                await asyncio.sleep(self.config.cleanup_interval)
    
    async def _perform_cleanup(self) -> None:
        """Perform cache cleanup."""
        # Remove expired entries
        current_time = datetime.now(timezone.utc)
        expired_keys = []
        
        for key, access_time in self._access_times.items():
            if (current_time - access_time).total_seconds() > self.config.default_ttl:
                expired_keys.append(key)
        
        for key in expired_keys:
            await self.delete(key)
            self.stats.evictions += 1
        
        # Evict based on strategy if cache is full
        if len(self._cache) > self.config.max_size:
            await self._evict_entries()
    
    async def _evict_entries(self) -> None:
        """Evict entries based on cache strategy."""
        if self.config.strategy == CacheStrategy.LRU:
            # Remove least recently used
            sorted_keys = sorted(
                self._access_times.keys(),
                key=lambda k: self._access_times[k]
            )
            keys_to_evict = sorted_keys[:len(self._cache) - self.config.max_size]
            
        elif self.config.strategy == CacheStrategy.FIFO:
            # Remove first in (oldest)
            keys_to_evict = list(self._cache.keys())[:len(self._cache) - self.config.max_size]
            
        else:  # LFU or TTL
            # Remove random entries (simplified)
            keys_to_evict = list(self._cache.keys())[:len(self._cache) - self.config.max_size]
        
        for key in keys_to_evict:
            await self.delete(key)
            self.stats.evictions += 1
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if not self._is_initialized:
            return None
        
        value = await self._get_internal(key)
        
        if value is not None:
            self._access_times[key] = datetime.now(timezone.utc)
            self._update_stats(hit=True)
        else:
            self._update_stats(hit=False)
        
        return value
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache."""
        if not self._is_initialized:
            return False
        
        success = await self._set_internal(key, value, ttl or self.config.default_ttl)
        
        if success:
            self._access_times[key] = datetime.now(timezone.utc)
            self._update_stats(set=True)
        
        return success
    
    async def delete(self, key: str) -> bool:
        """Delete value from cache."""
        if not self._is_initialized:
            return False
        
        success = await self._delete_internal(key)
        
        if success:
            self._access_times.pop(key, None)
            self._update_stats(delete=True)
        
        return success
    
    async def clear(self) -> bool:
        """Clear all cache entries."""
        if not self._is_initialized:
            return False
        
        success = await self._clear_internal()
        
        if success:
            self._access_times.clear()
        
        return success
    
    async def _get_internal(self, key: str) -> Optional[Any]:
        """Internal get method to be implemented by subclasses."""
        raise NotImplementedError
    
    async def _set_internal(self, key: str, value: Any, ttl: int) -> bool:
        """Internal set method to be implemented by subclasses."""
        raise NotImplementedError
    
    async def _delete_internal(self, key: str) -> bool:
        """Internal delete method to be implemented by subclasses."""
        raise NotImplementedError
    
    async def _clear_internal(self) -> bool:
        """Internal clear method to be implemented by subclasses."""
        raise NotImplementedError
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "hits": self.stats.hits,
            "misses": self.stats.misses,
            "sets": self.stats.sets,
            "deletes": self.stats.deletes,
            "evictions": self.stats.evictions,
            "total_size": len(self._cache),
            "memory_usage_mb": self.stats.memory_usage_mb,
            "hit_rate": self.stats.hit_rate,
            "last_cleanup": self.stats.last_cleanup.isoformat() if self.stats.last_cleanup else None,
            "config": {
                "cache_type": self.config.cache_type.value,
                "strategy": self.config.strategy.value,
                "max_size": self.config.max_size,
                "default_ttl": self.config.default_ttl
            }
        }

# =============================================================================
# Memory Cache
# =============================================================================

class MemoryCache(CacheBase):
    """In-memory cache implementation."""
    
    def __init__(self, config: CacheConfig):
        
    """__init__ function."""
super().__init__(config)
        self._expiry_times: Dict[str, datetime] = {}
    
    async def _initialize_internal(self) -> None:
        """Initialize memory cache."""
        # Nothing special to initialize for memory cache
        pass
    
    async def _cleanup_internal(self) -> None:
        """Cleanup memory cache."""
        self._cache.clear()
        self._access_times.clear()
        self._expiry_times.clear()
    
    async def _get_internal(self, key: str) -> Optional[Any]:
        """Get value from memory cache."""
        if key not in self._cache:
            return None
        
        # Check expiry
        if key in self._expiry_times:
            if datetime.now(timezone.utc) > self._expiry_times[key]:
                await self.delete(key)
                return None
        
        return self._cache[key]
    
    async def _set_internal(self, key: str, value: Any, ttl: int) -> bool:
        """Set value in memory cache."""
        try:
            self._cache[key] = value
            
            if ttl > 0:
                self._expiry_times[key] = datetime.now(timezone.utc) + timedelta(seconds=ttl)
            
            # Check size limit
            if len(self._cache) > self.config.max_size:
                await self._evict_entries()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to set value in memory cache: {e}")
            return False
    
    async def _delete_internal(self, key: str) -> bool:
        """Delete value from memory cache."""
        try:
            self._cache.pop(key, None)
            self._expiry_times.pop(key, None)
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete value from memory cache: {e}")
            return False
    
    async def _clear_internal(self) -> bool:
        """Clear memory cache."""
        try:
            self._cache.clear()
            self._expiry_times.clear()
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear memory cache: {e}")
            return False

# =============================================================================
# Redis Cache
# =============================================================================

class RedisCache(CacheBase):
    """Redis cache implementation."""
    
    def __init__(self, config: CacheConfig, redis_url: str):
        
    """__init__ function."""
super().__init__(config)
        self.redis_url = redis_url
        self.client = None
    
    async def _initialize_internal(self) -> None:
        """Initialize Redis cache."""
        
        self.client = redis.from_url(
            self.redis_url,
            encoding="utf-8",
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5,
            retry_on_timeout=True,
            health_check_interval=30
        )
        
        # Test connection
        await self.client.ping()
    
    async def _cleanup_internal(self) -> None:
        """Cleanup Redis cache."""
        if self.client:
            await self.client.close()
    
    async def _get_internal(self, key: str) -> Optional[Any]:
        """Get value from Redis cache."""
        try:
            value = await self.client.get(key)
            if value is None:
                return None
            
            if self.config.enable_serialization:
                return json.loads(value)
            else:
                return value
                
        except Exception as e:
            logger.error(f"Failed to get value from Redis cache: {e}")
            return None
    
    async def _set_internal(self, key: str, value: Any, ttl: int) -> bool:
        """Set value in Redis cache."""
        try:
            if self.config.enable_serialization:
                serialized_value = json.dumps(value)
            else:
                serialized_value = str(value)
            
            await self.client.setex(key, ttl, serialized_value)
            return True
            
        except Exception as e:
            logger.error(f"Failed to set value in Redis cache: {e}")
            return False
    
    async def _delete_internal(self, key: str) -> bool:
        """Delete value from Redis cache."""
        try:
            await self.client.delete(key)
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete value from Redis cache: {e}")
            return False
    
    async def _clear_internal(self) -> bool:
        """Clear Redis cache."""
        try:
            await self.client.flushdb()
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear Redis cache: {e}")
            return False

# =============================================================================
# Hybrid Cache
# =============================================================================

class HybridCache(CacheBase):
    """Hybrid cache implementation (memory + Redis)."""
    
    def __init__(self, config: CacheConfig, redis_url: str):
        
    """__init__ function."""
super().__init__(config)
        self.memory_cache = MemoryCache(config)
        self.redis_cache = RedisCache(config, redis_url)
    
    async def _initialize_internal(self) -> None:
        """Initialize hybrid cache."""
        await self.memory_cache.initialize()
        await self.redis_cache.initialize()
    
    async def _cleanup_internal(self) -> None:
        """Cleanup hybrid cache."""
        await self.memory_cache.cleanup()
        await self.redis_cache.cleanup()
    
    async def _get_internal(self, key: str) -> Optional[Any]:
        """Get value from hybrid cache."""
        # Try memory cache first
        value = await self.memory_cache.get(key)
        if value is not None:
            return value
        
        # Try Redis cache
        value = await self.redis_cache.get(key)
        if value is not None:
            # Store in memory cache for future access
            await self.memory_cache.set(key, value, self.config.default_ttl)
            return value
        
        return None
    
    async def _set_internal(self, key: str, value: Any, ttl: int) -> bool:
        """Set value in hybrid cache."""
        try:
            # Set in both caches
            memory_success = await self.memory_cache.set(key, value, ttl)
            redis_success = await self.redis_cache.set(key, value, ttl)
            
            return memory_success and redis_success
            
        except Exception as e:
            logger.error(f"Failed to set value in hybrid cache: {e}")
            return False
    
    async def _delete_internal(self, key: str) -> bool:
        """Delete value from hybrid cache."""
        try:
            # Delete from both caches
            memory_success = await self.memory_cache.delete(key)
            redis_success = await self.redis_cache.delete(key)
            
            return memory_success and redis_success
            
        except Exception as e:
            logger.error(f"Failed to delete value from hybrid cache: {e}")
            return False
    
    async def _clear_internal(self) -> bool:
        """Clear hybrid cache."""
        try:
            # Clear both caches
            memory_success = await self.memory_cache.clear()
            redis_success = await self.redis_cache.clear()
            
            return memory_success and redis_success
            
        except Exception as e:
            logger.error(f"Failed to clear hybrid cache: {e}")
            return False

# =============================================================================
# Cache Manager
# =============================================================================

class CacheManager:
    """Main cache manager for managing all caches."""
    
    def __init__(self) -> Any:
        self.caches: Dict[str, CacheBase] = {}
        self._initialized = False
    
    def register_cache(self, name: str, cache: CacheBase) -> None:
        """Register a cache."""
        self.caches[name] = cache
        logger.info(f"Registered cache: {name} ({cache.__class__.__name__})")
    
    def get_cache(self, name: str) -> Optional[CacheBase]:
        """Get a cache by name."""
        return self.caches.get(name)
    
    async def initialize_all(self) -> None:
        """Initialize all caches."""
        if self._initialized:
            return
        
        logger.info("Initializing all caches...")
        
        for name, cache in self.caches.items():
            try:
                await cache.initialize()
            except Exception as e:
                logger.error(f"Failed to initialize cache {name}: {e}")
                raise
        
        self._initialized = True
        logger.info("All caches initialized successfully")
    
    async def cleanup_all(self) -> None:
        """Cleanup all caches."""
        if not self._initialized:
            return
        
        logger.info("Cleaning up all caches...")
        
        for name, cache in self.caches.items():
            try:
                await cache.cleanup()
            except Exception as e:
                logger.error(f"Failed to cleanup cache {name}: {e}")
        
        self._initialized = False
        logger.info("All caches cleaned up successfully")
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of all caches."""
        health_status = {
            "initialized": self._initialized,
            "total_caches": len(self.caches),
            "caches": {}
        }
        
        for name, cache in self.caches.items():
            health_status["caches"][name] = cache.get_stats()
        
        return health_status

# =============================================================================
# Cache Decorators
# =============================================================================

def cached(ttl: int = 300, cache_name: str = "default"):
    """Decorator for caching function results."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            # This would be used with the cache manager
            # The actual caching would happen at the function level
            return await func(*args, **kwargs)
        return wrapper
    return decorator

# =============================================================================
# Export
# =============================================================================

__all__ = [
    "CacheType",
    "CacheStrategy",
    "CacheConfig",
    "CacheStats",
    "CacheBase",
    "MemoryCache",
    "RedisCache",
    "HybridCache",
    "CacheManager",
    "cached",
] 