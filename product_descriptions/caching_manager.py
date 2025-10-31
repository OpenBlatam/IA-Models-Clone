from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from collections import defaultdict, OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from typing_extensions import TypedDict
import aioredis
import orjson
from pydantic import BaseModel, Field
from typing import Any, List, Dict, Optional
"""
Advanced Caching Manager for Product Descriptions API

This module provides comprehensive caching capabilities including:
- Redis caching for distributed environments
- In-memory caching for fast access
- Hybrid caching strategies
- Cache invalidation patterns
- Cache warming and preloading
- Cache statistics and monitoring
- TTL management and eviction policies
"""



# Configure logging
logger = logging.getLogger(__name__)


class CacheStrategy(Enum):
    """Cache strategy types"""
    REDIS = "redis"
    MEMORY = "memory"
    HYBRID = "hybrid"
    LAYERED = "layered"


class CacheLevel(Enum):
    """Cache levels for layered caching"""
    L1 = "l1"  # Memory cache (fastest)
    L2 = "l2"  # Redis cache (distributed)
    L3 = "l3"  # Database/External (slowest)


class EvictionPolicy(Enum):
    """Cache eviction policies"""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    FIFO = "fifo"  # First In, First Out
    TTL = "ttl"  # Time To Live


@dataclass
class CacheConfig:
    """Configuration for caching"""
    strategy: CacheStrategy = CacheStrategy.HYBRID
    redis_url: str = "redis://localhost:6379"
    redis_db: int = 0
    redis_password: Optional[str] = None
    memory_max_size: int = 1000
    memory_ttl: int = 300  # 5 minutes
    redis_ttl: int = 3600  # 1 hour
    enable_compression: bool = True
    enable_serialization: bool = True
    eviction_policy: EvictionPolicy = EvictionPolicy.LRU
    enable_stats: bool = True
    enable_warming: bool = True
    warming_batch_size: int = 100
    retry_attempts: int = 3
    retry_delay: float = 0.1


class CacheStats(TypedDict):
    """Cache statistics"""
    hits: int
    misses: int
    sets: int
    deletes: int
    errors: int
    hit_rate: float
    total_requests: int


class CacheItem:
    """Represents a cached item with metadata"""
    
    def __init__(
        self,
        key: str,
        value: Any,
        ttl: int = 3600,
        created_at: Optional[float] = None,
        access_count: int = 0,
        last_accessed: Optional[float] = None
    ):
        
    """__init__ function."""
self.key = key
        self.value = value
        self.ttl = ttl
        self.created_at = created_at or time.time()
        self.access_count = access_count
        self.last_accessed = last_accessed or time.time()
    
    @property
    def is_expired(self) -> bool:
        """Check if item is expired"""
        return time.time() - self.created_at > self.ttl
    
    def access(self) -> Any:
        """Mark item as accessed"""
        self.access_count += 1
        self.last_accessed = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "key": self.key,
            "value": self.value,
            "ttl": self.ttl,
            "created_at": self.created_at,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CacheItem":
        """Create from dictionary"""
        return cls(**data)


class BaseCache(ABC):
    """Abstract base class for cache implementations"""
    
    def __init__(self, config: CacheConfig):
        
    """__init__ function."""
self.config = config
        self.stats = CacheStats(
            hits=0, misses=0, sets=0, deletes=0, errors=0,
            hit_rate=0.0, total_requests=0
        )
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        pass
    
    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache"""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete value from cache"""
        pass
    
    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists"""
        pass
    
    @abstractmethod
    async def clear(self) -> bool:
        """Clear all cache"""
        pass
    
    def update_stats(self, operation: str, success: bool = True):
        """Update cache statistics"""
        self.stats["total_requests"] += 1
        
        if operation == "get":
            if success:
                self.stats["hits"] += 1
            else:
                self.stats["misses"] += 1
        elif operation == "set":
            self.stats["sets"] += 1
        elif operation == "delete":
            self.stats["deletes"] += 1
        elif operation == "error":
            self.stats["errors"] += 1
        
        # Calculate hit rate
        total_gets = self.stats["hits"] + self.stats["misses"]
        if total_gets > 0:
            self.stats["hit_rate"] = self.stats["hits"] / total_gets
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics"""
        return self.stats.copy()


class MemoryCache(BaseCache):
    """In-memory cache implementation with LRU/LFU eviction"""
    
    def __init__(self, config: CacheConfig):
        
    """__init__ function."""
super().__init__(config)
        self.max_size = config.memory_max_size
        self.ttl = config.memory_ttl
        self.eviction_policy = config.eviction_policy
        
        if self.eviction_policy == EvictionPolicy.LRU:
            self.cache: OrderedDict[str, CacheItem] = OrderedDict()
        else:
            self.cache: Dict[str, CacheItem] = {}
        
        # Background task for cleanup
        self._cleanup_task: Optional[asyncio.Task] = None
        self._start_cleanup_task()
    
    def _start_cleanup_task(self) -> Any:
        """Start background cleanup task"""
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
    
    async def _cleanup_loop(self) -> Any:
        """Background cleanup loop"""
        while True:
            try:
                await asyncio.sleep(60)  # Cleanup every minute
                await self._cleanup_expired()
            except Exception as e:
                logger.error(f"Cache cleanup error: {e}")
    
    async def _cleanup_expired(self) -> Any:
        """Remove expired items"""
        expired_keys = []
        for key, item in self.cache.items():
            if item.is_expired:
                expired_keys.append(key)
        
        for key in expired_keys:
            await self.delete(key)
    
    def _evict_if_needed(self) -> Any:
        """Evict items if cache is full"""
        if len(self.cache) < self.max_size:
            return
        
        if self.eviction_policy == EvictionPolicy.LRU:
            # Remove least recently used
            self.cache.popitem(last=False)
        elif self.eviction_policy == EvictionPolicy.LFU:
            # Remove least frequently used
            min_key = min(self.cache.keys(), key=lambda k: self.cache[k].access_count)
            del self.cache[min_key]
        elif self.eviction_policy == EvictionPolicy.FIFO:
            # Remove first item
            first_key = next(iter(self.cache))
            del self.cache[first_key]
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from memory cache"""
        try:
            item = self.cache.get(key)
            if item is None:
                self.update_stats("get", False)
                return None
            
            if item.is_expired:
                await self.delete(key)
                self.update_stats("get", False)
                return None
            
            item.access()
            
            # Move to end for LRU
            if self.eviction_policy == EvictionPolicy.LRU:
                self.cache.move_to_end(key)
            
            self.update_stats("get", True)
            return item.value
            
        except Exception as e:
            logger.error(f"Memory cache get error: {e}")
            self.update_stats("error")
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in memory cache"""
        try:
            item = CacheItem(
                key=key,
                value=value,
                ttl=ttl or self.ttl
            )
            
            self.cache[key] = item
            self._evict_if_needed()
            
            self.update_stats("set", True)
            return True
            
        except Exception as e:
            logger.error(f"Memory cache set error: {e}")
            self.update_stats("error")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete value from memory cache"""
        try:
            if key in self.cache:
                del self.cache[key]
                self.update_stats("delete", True)
                return True
            return False
        except Exception as e:
            logger.error(f"Memory cache delete error: {e}")
            self.update_stats("error")
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in memory cache"""
        try:
            item = self.cache.get(key)
            if item is None:
                return False
            
            if item.is_expired:
                await self.delete(key)
                return False
            
            return True
        except Exception as e:
            logger.error(f"Memory cache exists error: {e}")
            return False
    
    async def clear(self) -> bool:
        """Clear all memory cache"""
        try:
            self.cache.clear()
            return True
        except Exception as e:
            logger.error(f"Memory cache clear error: {e}")
            return False
    
    async def close(self) -> Any:
        """Close memory cache and cleanup tasks"""
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass


class RedisCache(BaseCache):
    """Redis cache implementation"""
    
    def __init__(self, config: CacheConfig):
        
    """__init__ function."""
super().__init__(config)
        self.redis_url = config.redis_url
        self.redis_db = config.redis_db
        self.redis_password = config.redis_password
        self.ttl = config.redis_ttl
        self.redis: Optional[aioredis.Redis] = None
        self._connection_lock = asyncio.Lock()
    
    async def _get_redis(self) -> aioredis.Redis:
        """Get Redis connection with lazy initialization"""
        if self.redis is None:
            async with self._connection_lock:
                if self.redis is None:
                    self.redis = await aioredis.from_url(
                        self.redis_url,
                        db=self.redis_db,
                        password=self.redis_password,
                        encoding="utf-8",
                        decode_responses=True
                    )
        return self.redis
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from Redis cache"""
        try:
            redis = await self._get_redis()
            value = await redis.get(key)
            
            if value is None:
                self.update_stats("get", False)
                return None
            
            # Deserialize if needed
            if self.config.enable_serialization:
                try:
                    value = orjson.loads(value)
                except Exception:
                    pass  # Return as string if deserialization fails
            
            self.update_stats("get", True)
            return value
            
        except Exception as e:
            logger.error(f"Redis cache get error: {e}")
            self.update_stats("error")
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in Redis cache"""
        try:
            redis = await self._get_redis()
            
            # Serialize if needed
            if self.config.enable_serialization and not isinstance(value, (str, bytes)):
                value = orjson.dumps(value).decode()
            
            await redis.setex(key, ttl or self.ttl, value)
            self.update_stats("set", True)
            return True
            
        except Exception as e:
            logger.error(f"Redis cache set error: {e}")
            self.update_stats("error")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete value from Redis cache"""
        try:
            redis = await self._get_redis()
            result = await redis.delete(key)
            self.update_stats("delete", True)
            return result > 0
        except Exception as e:
            logger.error(f"Redis cache delete error: {e}")
            self.update_stats("error")
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in Redis cache"""
        try:
            redis = await self._get_redis()
            return await redis.exists(key) > 0
        except Exception as e:
            logger.error(f"Redis cache exists error: {e}")
            return False
    
    async def clear(self) -> bool:
        """Clear all Redis cache"""
        try:
            redis = await self._get_redis()
            await redis.flushdb()
            return True
        except Exception as e:
            logger.error(f"Redis cache clear error: {e}")
            return False
    
    async def close(self) -> Any:
        """Close Redis connection"""
        if self.redis:
            await self.redis.close()
            self.redis = None


class HybridCache(BaseCache):
    """Hybrid cache combining memory and Redis"""
    
    def __init__(self, config: CacheConfig):
        
    """__init__ function."""
super().__init__(config)
        self.memory_cache = MemoryCache(config)
        self.redis_cache = RedisCache(config)
        self.write_through = True  # Write to both caches
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from hybrid cache (memory first, then Redis)"""
        # Try memory cache first
        value = await self.memory_cache.get(key)
        if value is not None:
            return value
        
        # Try Redis cache
        value = await self.redis_cache.get(key)
        if value is not None:
            # Populate memory cache
            await self.memory_cache.set(key, value)
            return value
        
        return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in hybrid cache"""
        if self.write_through:
            # Write to both caches
            memory_success = await self.memory_cache.set(key, value, ttl)
            redis_success = await self.redis_cache.set(key, value, ttl)
            return memory_success and redis_success
        else:
            # Write to memory only, Redis will be populated on first access
            return await self.memory_cache.set(key, value, ttl)
    
    async def delete(self, key: str) -> bool:
        """Delete value from hybrid cache"""
        memory_success = await self.memory_cache.delete(key)
        redis_success = await self.redis_cache.delete(key)
        return memory_success or redis_success
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in hybrid cache"""
        return await self.memory_cache.exists(key) or await self.redis_cache.exists(key)
    
    async def clear(self) -> bool:
        """Clear all hybrid cache"""
        memory_success = await self.memory_cache.clear()
        redis_success = await self.redis_cache.clear()
        return memory_success and redis_success
    
    def get_stats(self) -> CacheStats:
        """Get combined cache statistics"""
        memory_stats = self.memory_cache.get_stats()
        redis_stats = self.redis_cache.get_stats()
        
        return CacheStats(
            hits=memory_stats["hits"] + redis_stats["hits"],
            misses=memory_stats["misses"] + redis_stats["misses"],
            sets=memory_stats["sets"] + redis_stats["sets"],
            deletes=memory_stats["deletes"] + redis_stats["deletes"],
            errors=memory_stats["errors"] + redis_stats["errors"],
            hit_rate=(memory_stats["hits"] + redis_stats["hits"]) / 
                    max(1, memory_stats["total_requests"] + redis_stats["total_requests"]),
            total_requests=memory_stats["total_requests"] + redis_stats["total_requests"]
        )
    
    async def close(self) -> Any:
        """Close hybrid cache"""
        await self.memory_cache.close()
        await self.redis_cache.close()


class CacheManager:
    """Main cache manager for the application"""
    
    def __init__(self, config: CacheConfig):
        
    """__init__ function."""
self.config = config
        self.cache: Optional[BaseCache] = None
        self._initialized = False
        self._lock = asyncio.Lock()
    
    async def initialize(self) -> Any:
        """Initialize cache based on strategy"""
        if self._initialized:
            return
        
        async with self._lock:
            if self._initialized:
                return
            
            if self.config.strategy == CacheStrategy.REDIS:
                self.cache = RedisCache(self.config)
            elif self.config.strategy == CacheStrategy.MEMORY:
                self.cache = MemoryCache(self.config)
            elif self.config.strategy == CacheStrategy.HYBRID:
                self.cache = HybridCache(self.config)
            else:
                raise ValueError(f"Unsupported cache strategy: {self.config.strategy}")
            
            self._initialized = True
            logger.info(f"Cache initialized with strategy: {self.config.strategy.value}")
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        await self.initialize()
        return await self.cache.get(key)
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache"""
        await self.initialize()
        return await self.cache.set(key, value, ttl)
    
    async def delete(self, key: str) -> bool:
        """Delete value from cache"""
        await self.initialize()
        return await self.cache.delete(key)
    
    async def exists(self, key: str) -> bool:
        """Check if key exists"""
        await self.initialize()
        return await self.cache.exists(key)
    
    async def clear(self) -> bool:
        """Clear all cache"""
        await self.initialize()
        return await self.cache.clear()
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics"""
        if not self._initialized or not self.cache:
            return CacheStats(
                hits=0, misses=0, sets=0, deletes=0, errors=0,
                hit_rate=0.0, total_requests=0
            )
        return self.cache.get_stats()
    
    async def close(self) -> Any:
        """Close cache manager"""
        if self.cache:
            await self.cache.close()
            self._initialized = False


# Cache decorators and utilities
def cached(
    ttl: Optional[int] = None,
    key_prefix: str = "",
    cache_manager: Optional[CacheManager] = None
):
    """Decorator for caching function results"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            # Generate cache key
            cache_key = f"{key_prefix}:{func.__name__}:{hash(str(args) + str(sorted(kwargs.items())))}"
            
            # Get cache manager
            cm = cache_manager or getattr(func, '_cache_manager', None)
            if not cm:
                logger.warning("No cache manager available for caching")
                return await func(*args, **kwargs)
            
            # Try to get from cache
            cached_value = await cm.get(cache_key)
            if cached_value is not None:
                return cached_value
            
            # Execute function and cache result
            result = await func(*args, **kwargs)
            await cm.set(cache_key, result, ttl)
            
            return result
        
        return wrapper
    return decorator


def cache_invalidate(keys: List[str], cache_manager: Optional[CacheManager] = None):
    """Decorator for invalidating cache keys after function execution"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            result = await func(*args, **kwargs)
            
            # Invalidate cache keys
            cm = cache_manager or getattr(func, '_cache_manager', None)
            if cm:
                for key in keys:
                    await cm.delete(key)
            
            return result
        
        return wrapper
    return decorator


class CacheWarmingService:
    """Service for warming up cache with frequently accessed data"""
    
    def __init__(self, cache_manager: CacheManager):
        
    """__init__ function."""
self.cache_manager = cache_manager
        self.warming_tasks: Set[str] = set()
    
    async def warm_cache(self, data_source: Callable, key_pattern: str, batch_size: int = 100):
        """Warm cache with data from source"""
        task_id = f"{key_pattern}:{id(data_source)}"
        if task_id in self.warming_tasks:
            return
        
        self.warming_tasks.add(task_id)
        
        try:
            # Get data from source
            data = await data_source()
            
            if isinstance(data, dict):
                # Cache dictionary items
                for key, value in data.items():
                    cache_key = f"{key_pattern}:{key}"
                    await self.cache_manager.set(cache_key, value)
                    
                    if len(data) > batch_size:
                        await asyncio.sleep(0.01)  # Prevent blocking
            
            elif isinstance(data, list):
                # Cache list items
                for i, item in enumerate(data):
                    cache_key = f"{key_pattern}:{i}"
                    await self.cache_manager.set(cache_key, item)
                    
                    if i % batch_size == 0:
                        await asyncio.sleep(0.01)  # Prevent blocking
            
            logger.info(f"Cache warming completed for {key_pattern}")
            
        except Exception as e:
            logger.error(f"Cache warming failed for {key_pattern}: {e}")
        finally:
            self.warming_tasks.discard(task_id)


# Static data cache patterns
class StaticDataCache:
    """Specialized cache for static data"""
    
    def __init__(self, cache_manager: CacheManager):
        
    """__init__ function."""
self.cache_manager = cache_manager
        self.static_keys: Set[str] = set()
    
    async def cache_static_data(self, key: str, data: Any, ttl: int = 86400) -> bool:
        """Cache static data with long TTL"""
        self.static_keys.add(key)
        return await self.cache_manager.set(key, data, ttl)
    
    async def get_static_data(self, key: str) -> Optional[Any]:
        """Get static data from cache"""
        return await self.cache_manager.get(key)
    
    async def invalidate_static_data(self, key: str) -> bool:
        """Invalidate static data cache"""
        self.static_keys.discard(key)
        return await self.cache_manager.delete(key)
    
    async def refresh_static_data(self, key: str, data_source: Callable) -> bool:
        """Refresh static data from source"""
        try:
            data = await data_source()
            return await self.cache_static_data(key, data)
        except Exception as e:
            logger.error(f"Failed to refresh static data {key}: {e}")
            return False


# Cache monitoring and analytics
class CacheMonitor:
    """Monitor cache performance and health"""
    
    def __init__(self, cache_manager: CacheManager):
        
    """__init__ function."""
self.cache_manager = cache_manager
        self.metrics: List[Dict[str, Any]] = []
        self.alert_thresholds = {
            "hit_rate": 0.8,
            "error_rate": 0.1,
            "response_time": 0.1
        }
    
    async def record_metric(self, metric: Dict[str, Any]):
        """Record cache metric"""
        metric["timestamp"] = time.time()
        self.metrics.append(metric)
        
        # Keep only last 1000 metrics
        if len(self.metrics) > 1000:
            self.metrics = self.metrics[-1000:]
    
    async def get_performance_report(self) -> Dict[str, Any]:
        """Generate performance report"""
        stats = self.cache_manager.get_stats()
        
        report = {
            "cache_stats": stats,
            "performance_metrics": self.metrics[-100:] if self.metrics else [],
            "alerts": await self._check_alerts(stats),
            "recommendations": await self._generate_recommendations(stats)
        }
        
        return report
    
    async def _check_alerts(self, stats: CacheStats) -> List[str]:
        """Check for performance alerts"""
        alerts = []
        
        if stats["hit_rate"] < self.alert_thresholds["hit_rate"]:
            alerts.append(f"Low cache hit rate: {stats['hit_rate']:.2%}")
        
        if stats["total_requests"] > 0:
            error_rate = stats["errors"] / stats["total_requests"]
            if error_rate > self.alert_thresholds["error_rate"]:
                alerts.append(f"High cache error rate: {error_rate:.2%}")
        
        return alerts
    
    async def _generate_recommendations(self, stats: CacheStats) -> List[str]:
        """Generate cache optimization recommendations"""
        recommendations = []
        
        if stats["hit_rate"] < 0.5:
            recommendations.append("Consider increasing cache TTL or cache size")
        
        if stats["misses"] > stats["hits"]:
            recommendations.append("Consider implementing cache warming")
        
        if stats["errors"] > 0:
            recommendations.append("Investigate cache errors and connection issues")
        
        return recommendations


# Default cache configuration
DEFAULT_CACHE_CONFIG = CacheConfig(
    strategy=CacheStrategy.HYBRID,
    redis_url="redis://localhost:6379",
    memory_max_size=1000,
    memory_ttl=300,
    redis_ttl=3600,
    enable_compression=True,
    enable_serialization=True,
    eviction_policy=EvictionPolicy.LRU,
    enable_stats=True,
    enable_warming=True
)

# Global cache manager instance
_cache_manager: Optional[CacheManager] = None


async def get_cache_manager(config: Optional[CacheConfig] = None) -> CacheManager:
    """Get global cache manager instance"""
    global _cache_manager
    
    if _cache_manager is None:
        _cache_manager = CacheManager(config or DEFAULT_CACHE_CONFIG)
        await _cache_manager.initialize()
    
    return _cache_manager


async def close_cache_manager():
    """Close global cache manager"""
    global _cache_manager
    
    if _cache_manager:
        await _cache_manager.close()
        _cache_manager = None 