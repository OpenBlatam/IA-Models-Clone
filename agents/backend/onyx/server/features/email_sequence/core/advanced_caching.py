"""
Advanced Caching System

Provides a sophisticated multi-level caching system with predictive caching,
intelligent cache management, and performance optimization for the email sequence system.
"""

import asyncio
import logging
import time
import json
import hashlib
import pickle
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass
from collections import defaultdict, deque, OrderedDict
from enum import Enum
from datetime import datetime, timedelta
import functools
import weakref

# Cache backends
import redis
import aioredis
from cachetools import TTLCache, LRUCache, LFUCache

logger = logging.getLogger(__name__)

# Constants
DEFAULT_CACHE_TTL = 3600  # 1 hour
DEFAULT_MAX_SIZE = 10000
DEFAULT_CLEANUP_INTERVAL = 300  # 5 minutes
PREDICTION_WINDOW = 3600  # 1 hour


class CacheLevel(Enum):
    """Cache levels"""
    L1 = "l1"  # Memory cache (fastest)
    L2 = "l2"  # Redis cache (fast)
    L3 = "l3"  # Database cache (slower)


class CacheStrategy(Enum):
    """Cache strategies"""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    TTL = "ttl"   # Time To Live
    HYBRID = "hybrid"  # Combination of strategies


class CacheOperation(Enum):
    """Cache operations"""
    GET = "get"
    SET = "set"
    DELETE = "delete"
    UPDATE = "update"
    CLEAR = "clear"


@dataclass
class CacheConfig:
    """Cache configuration"""
    # Memory cache settings
    l1_enabled: bool = True
    l1_max_size: int = 1000
    l1_ttl: int = 300  # 5 minutes
    
    # Redis cache settings
    l2_enabled: bool = True
    l2_ttl: int = 3600  # 1 hour
    l2_max_size: int = 10000
    
    # Database cache settings
    l3_enabled: bool = False
    l3_ttl: int = 86400  # 24 hours
    
    # Strategy settings
    strategy: CacheStrategy = CacheStrategy.HYBRID
    enable_predictive_caching: bool = True
    enable_compression: bool = True
    enable_encryption: bool = False
    
    # Performance settings
    cleanup_interval: int = 300
    max_cleanup_time: int = 30
    enable_metrics: bool = True
    enable_analytics: bool = True
    
    # Predictive caching
    prediction_window: int = 3600
    min_prediction_confidence: float = 0.7
    enable_ml_predictions: bool = True


@dataclass
class CacheMetrics:
    """Cache performance metrics"""
    hits: int = 0
    misses: int = 0
    sets: int = 0
    deletes: int = 0
    evictions: int = 0
    errors: int = 0
    total_operations: int = 0
    avg_response_time: float = 0.0
    hit_rate: float = 0.0
    memory_usage: float = 0.0
    prediction_accuracy: float = 0.0


@dataclass
class CacheItem:
    """Cache item with metadata"""
    key: str
    value: Any
    created_at: datetime
    accessed_at: datetime
    access_count: int = 0
    size: int = 0
    compressed: bool = False
    encrypted: bool = False
    ttl: Optional[int] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.size == 0:
            self.size = self._calculate_size()

    def _calculate_size(self) -> int:
        """Calculate item size in bytes"""
        try:
            return len(pickle.dumps(self.value))
        except:
            return len(str(self.value))

    def is_expired(self) -> bool:
        """Check if item is expired"""
        if self.ttl is None:
            return False
        return datetime.utcnow() > self.created_at + timedelta(seconds=self.ttl)

    def update_access(self):
        """Update access statistics"""
        self.accessed_at = datetime.utcnow()
        self.access_count += 1


class PredictiveCache:
    """Predictive caching based on access patterns"""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.access_patterns = defaultdict(list)
        self.prediction_model = None
        self.prediction_accuracy = 0.0
        
    def record_access(self, key: str, timestamp: datetime = None):
        """Record cache access for pattern analysis"""
        if timestamp is None:
            timestamp = datetime.utcnow()
        
        self.access_patterns[key].append(timestamp)
        
        # Keep only recent accesses
        cutoff = timestamp - timedelta(seconds=self.config.prediction_window)
        self.access_patterns[key] = [
            t for t in self.access_patterns[key] 
            if t > cutoff
        ]
    
    def predict_next_access(self, key: str) -> Optional[datetime]:
        """Predict when key will be accessed next"""
        if key not in self.access_patterns:
            return None
        
        accesses = self.access_patterns[key]
        if len(accesses) < 2:
            return None
        
        # Simple prediction based on average interval
        intervals = []
        for i in range(1, len(accesses)):
            interval = (accesses[i] - accesses[i-1]).total_seconds()
            intervals.append(interval)
        
        if not intervals:
            return None
        
        avg_interval = sum(intervals) / len(intervals)
        last_access = accesses[-1]
        
        return last_access + timedelta(seconds=avg_interval)
    
    def get_prediction_confidence(self, key: str) -> float:
        """Get confidence level for prediction"""
        accesses = self.access_patterns.get(key, [])
        if len(accesses) < 3:
            return 0.0
        
        # Calculate variance in access intervals
        intervals = []
        for i in range(1, len(accesses)):
            interval = (accesses[i] - accesses[i-1]).total_seconds()
            intervals.append(interval)
        
        if not intervals:
            return 0.0
        
        avg_interval = sum(intervals) / len(intervals)
        variance = sum((i - avg_interval) ** 2 for i in intervals) / len(intervals)
        
        # Lower variance = higher confidence
        confidence = max(0.0, 1.0 - (variance / (avg_interval ** 2)))
        return min(1.0, confidence)


class AdvancedCache:
    """
    Advanced multi-level caching system with predictive capabilities.
    """
    
    def __init__(self, config: CacheConfig, redis_client: Optional[redis.Redis] = None):
        self.config = config
        self.redis_client = redis_client
        
        # Initialize cache levels
        self.l1_cache = self._initialize_l1_cache()
        self.l2_cache = self._initialize_l2_cache()
        self.l3_cache = self._initialize_l3_cache()
        
        # Predictive caching
        self.predictive_cache = PredictiveCache(config)
        
        # Metrics and analytics
        self.metrics = CacheMetrics()
        self.operation_times = deque(maxlen=1000)
        
        # Cleanup task
        self.cleanup_task = None
        self.is_running = False
        
        # Event callbacks
        self.event_callbacks: Dict[str, List[Callable]] = defaultdict(list)
        
        logger.info("Advanced Cache initialized")
    
    def _initialize_l1_cache(self) -> Optional[Dict]:
        """Initialize L1 (memory) cache"""
        if not self.config.l1_enabled:
            return None
        
        if self.config.strategy == CacheStrategy.LRU:
            return LRUCache(maxsize=self.config.l1_max_size)
        elif self.config.strategy == CacheStrategy.LFU:
            return LFUCache(maxsize=self.config.l1_max_size)
        elif self.config.strategy == CacheStrategy.TTL:
            return TTLCache(maxsize=self.config.l1_max_size, ttl=self.config.l1_ttl)
        else:  # HYBRID
            return OrderedDict()
    
    def _initialize_l2_cache(self) -> Optional[redis.Redis]:
        """Initialize L2 (Redis) cache"""
        if not self.config.l2_enabled:
            return None
        
        if self.redis_client is None:
            try:
                self.redis_client = redis.Redis(
                    host='localhost',
                    port=6379,
                    db=0,
                    decode_responses=True
                )
                self.redis_client.ping()
            except Exception as e:
                logger.warning(f"Redis not available: {e}")
                return None
        
        return self.redis_client
    
    def _initialize_l3_cache(self) -> Optional[Dict]:
        """Initialize L3 (database) cache"""
        if not self.config.l3_enabled:
            return None
        
        # This would integrate with database caching
        return {}
    
    async def start(self):
        """Start the cache system"""
        self.is_running = True
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("Advanced Cache started")
    
    async def stop(self):
        """Stop the cache system"""
        self.is_running = False
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
        logger.info("Advanced Cache stopped")
    
    async def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache with multi-level fallback"""
        start_time = time.time()
        
        try:
            # Try L1 cache first
            if self.l1_cache is not None:
                value = await self._get_from_l1(key)
                if value is not None:
                    await self._record_hit(key, CacheLevel.L1, time.time() - start_time)
                    return value
            
            # Try L2 cache
            if self.l2_cache is not None:
                value = await self._get_from_l2(key)
                if value is not None:
                    # Store in L1 for future access
                    await self._set_to_l1(key, value)
                    await self._record_hit(key, CacheLevel.L2, time.time() - start_time)
                    return value
            
            # Try L3 cache
            if self.l3_cache is not None:
                value = await self._get_from_l3(key)
                if value is not None:
                    # Store in L1 and L2 for future access
                    await self._set_to_l1(key, value)
                    await self._set_to_l2(key, value)
                    await self._record_hit(key, CacheLevel.L3, time.time() - start_time)
                    return value
            
            # Cache miss
            await self._record_miss(key, time.time() - start_time)
            return default
            
        except Exception as e:
            logger.error(f"Cache get error for key {key}: {e}")
            self.metrics.errors += 1
            return default
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache across all levels"""
        start_time = time.time()
        
        try:
            success = True
            
            # Set in L1
            if self.l1_cache is not None:
                success &= await self._set_to_l1(key, value, ttl)
            
            # Set in L2
            if self.l2_cache is not None:
                success &= await self._set_to_l2(key, value, ttl)
            
            # Set in L3
            if self.l3_cache is not None:
                success &= await self._set_to_l3(key, value, ttl)
            
            if success:
                self.metrics.sets += 1
                await self._trigger_event("set", key, value)
            
            return success
            
        except Exception as e:
            logger.error(f"Cache set error for key {key}: {e}")
            self.metrics.errors += 1
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete value from all cache levels"""
        try:
            success = True
            
            # Delete from L1
            if self.l1_cache is not None:
                success &= await self._delete_from_l1(key)
            
            # Delete from L2
            if self.l2_cache is not None:
                success &= await self._delete_from_l2(key)
            
            # Delete from L3
            if self.l3_cache is not None:
                success &= await self._delete_from_l3(key)
            
            if success:
                self.metrics.deletes += 1
                await self._trigger_event("delete", key)
            
            return success
            
        except Exception as e:
            logger.error(f"Cache delete error for key {key}: {e}")
            self.metrics.errors += 1
            return False
    
    async def clear(self, level: Optional[CacheLevel] = None) -> bool:
        """Clear cache at specified level or all levels"""
        try:
            if level is None or level == CacheLevel.L1:
                if self.l1_cache is not None:
                    self.l1_cache.clear()
            
            if level is None or level == CacheLevel.L2:
                if self.l2_cache is not None:
                    await self.l2_cache.flushdb()
            
            if level is None or level == CacheLevel.L3:
                if self.l3_cache is not None:
                    self.l3_cache.clear()
            
            await self._trigger_event("clear", level=level)
            return True
            
        except Exception as e:
            logger.error(f"Cache clear error: {e}")
            self.metrics.errors += 1
            return False
    
    async def _get_from_l1(self, key: str) -> Optional[Any]:
        """Get value from L1 cache"""
        if key in self.l1_cache:
            item = self.l1_cache[key]
            if isinstance(item, CacheItem):
                if item.is_expired():
                    del self.l1_cache[key]
                    return None
                item.update_access()
                return item.value
            else:
                return item
        return None
    
    async def _get_from_l2(self, key: str) -> Optional[Any]:
        """Get value from L2 cache"""
        try:
            value = self.l2_cache.get(key)
            if value is not None:
                return json.loads(value)
            return None
        except Exception as e:
            logger.error(f"L2 cache get error: {e}")
            return None
    
    async def _get_from_l3(self, key: str) -> Optional[Any]:
        """Get value from L3 cache"""
        # This would implement database caching
        return None
    
    async def _set_to_l1(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in L1 cache"""
        try:
            item = CacheItem(
                key=key,
                value=value,
                created_at=datetime.utcnow(),
                accessed_at=datetime.utcnow(),
                ttl=ttl or self.config.l1_ttl
            )
            
            self.l1_cache[key] = item
            return True
        except Exception as e:
            logger.error(f"L1 cache set error: {e}")
            return False
    
    async def _set_to_l2(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in L2 cache"""
        try:
            ttl = ttl or self.config.l2_ttl
            serialized_value = json.dumps(value)
            self.l2_cache.setex(key, ttl, serialized_value)
            return True
        except Exception as e:
            logger.error(f"L2 cache set error: {e}")
            return False
    
    async def _set_to_l3(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in L3 cache"""
        # This would implement database caching
        return True
    
    async def _delete_from_l1(self, key: str) -> bool:
        """Delete value from L1 cache"""
        try:
            if key in self.l1_cache:
                del self.l1_cache[key]
            return True
        except Exception as e:
            logger.error(f"L1 cache delete error: {e}")
            return False
    
    async def _delete_from_l2(self, key: str) -> bool:
        """Delete value from L2 cache"""
        try:
            self.l2_cache.delete(key)
            return True
        except Exception as e:
            logger.error(f"L2 cache delete error: {e}")
            return False
    
    async def _delete_from_l3(self, key: str) -> bool:
        """Delete value from L3 cache"""
        # This would implement database caching
        return True
    
    async def _record_hit(self, key: str, level: CacheLevel, response_time: float):
        """Record cache hit"""
        self.metrics.hits += 1
        self.metrics.total_operations += 1
        self.operation_times.append(response_time)
        
        # Update average response time
        if self.operation_times:
            self.metrics.avg_response_time = sum(self.operation_times) / len(self.operation_times)
        
        # Update hit rate
        total_ops = self.metrics.hits + self.metrics.misses
        if total_ops > 0:
            self.metrics.hit_rate = self.metrics.hits / total_ops
        
        # Record for predictive caching
        self.predictive_cache.record_access(key)
        
        await self._trigger_event("hit", key, level=level, response_time=response_time)
    
    async def _record_miss(self, key: str, response_time: float):
        """Record cache miss"""
        self.metrics.misses += 1
        self.metrics.total_operations += 1
        self.operation_times.append(response_time)
        
        # Update average response time
        if self.operation_times:
            self.metrics.avg_response_time = sum(self.operation_times) / len(self.operation_times)
        
        # Update hit rate
        total_ops = self.metrics.hits + self.metrics.misses
        if total_ops > 0:
            self.metrics.hit_rate = self.metrics.hits / total_ops
        
        await self._trigger_event("miss", key, response_time=response_time)
    
    async def _cleanup_loop(self):
        """Background cleanup loop"""
        while self.is_running:
            try:
                await self._cleanup_expired_items()
                await asyncio.sleep(self.config.cleanup_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup error: {e}")
                await asyncio.sleep(60)  # Wait before retrying
    
    async def _cleanup_expired_items(self):
        """Clean up expired items from all cache levels"""
        start_time = time.time()
        
        # Clean L1 cache
        if self.l1_cache is not None:
            expired_keys = []
            for key, item in list(self.l1_cache.items()):
                if isinstance(item, CacheItem) and item.is_expired():
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self.l1_cache[key]
                self.metrics.evictions += 1
        
        # L2 and L3 cleanup would be handled by their respective systems
        
        cleanup_time = time.time() - start_time
        if cleanup_time > self.config.max_cleanup_time:
            logger.warning(f"Cache cleanup took {cleanup_time:.2f}s")
    
    def add_event_callback(self, event: str, callback: Callable):
        """Add event callback"""
        self.event_callbacks[event].append(callback)
    
    async def _trigger_event(self, event: str, *args, **kwargs):
        """Trigger event callbacks"""
        for callback in self.event_callbacks[event]:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(*args, **kwargs)
                else:
                    callback(*args, **kwargs)
            except Exception as e:
                logger.error(f"Event callback error: {e}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get cache metrics"""
        return {
            "hits": self.metrics.hits,
            "misses": self.metrics.misses,
            "sets": self.metrics.sets,
            "deletes": self.metrics.deletes,
            "evictions": self.metrics.evictions,
            "errors": self.metrics.errors,
            "total_operations": self.metrics.total_operations,
            "avg_response_time": self.metrics.avg_response_time,
            "hit_rate": self.metrics.hit_rate,
            "memory_usage": self._calculate_memory_usage(),
            "prediction_accuracy": self.predictive_cache.prediction_accuracy
        }
    
    def _calculate_memory_usage(self) -> float:
        """Calculate memory usage percentage"""
        if self.l1_cache is None:
            return 0.0
        
        try:
            if hasattr(self.l1_cache, '__len__'):
                return len(self.l1_cache) / self.config.l1_max_size
            return 0.0
        except:
            return 0.0
    
    def get_predictions(self) -> Dict[str, Any]:
        """Get predictive caching insights"""
        predictions = {}
        for key in list(self.predictive_cache.access_patterns.keys())[:10]:  # Top 10
            next_access = self.predictive_cache.predict_next_access(key)
            confidence = self.predictive_cache.get_prediction_confidence(key)
            
            if next_access and confidence >= self.config.min_prediction_confidence:
                predictions[key] = {
                    "next_access": next_access.isoformat(),
                    "confidence": confidence
                }
        
        return predictions


# Decorator for easy caching
def cached(ttl: Optional[int] = None, key_prefix: str = ""):
    """Decorator for caching function results"""
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key
            key_parts = [key_prefix, func.__name__]
            key_parts.extend(str(arg) for arg in args)
            key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
            cache_key = hashlib.md5(":".join(key_parts).encode()).hexdigest()
            
            # Try to get from cache
            cache = getattr(wrapper, '_cache', None)
            if cache is not None:
                result = await cache.get(cache_key)
                if result is not None:
                    return result
            
            # Execute function and cache result
            result = await func(*args, **kwargs)
            
            if cache is not None:
                await cache.set(cache_key, result, ttl)
            
            return result
        
        return wrapper
    return decorator


# Cache manager for global access
class CacheManager:
    """Global cache manager"""
    
    _instance = None
    _cache = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def initialize(cls, config: CacheConfig, redis_client: Optional[redis.Redis] = None):
        """Initialize the global cache manager"""
        if cls._cache is None:
            cls._cache = AdvancedCache(config, redis_client)
        return cls._cache
    
    @classmethod
    def get_cache(cls) -> Optional[AdvancedCache]:
        """Get the global cache instance"""
        return cls._cache
    
    @classmethod
    async def start(cls):
        """Start the global cache"""
        if cls._cache:
            await cls._cache.start()
    
    @classmethod
    async def stop(cls):
        """Stop the global cache"""
        if cls._cache:
            await cls._cache.stop() 