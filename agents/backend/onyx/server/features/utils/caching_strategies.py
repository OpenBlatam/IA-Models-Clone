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
import logging
import hashlib
import json
from typing import Any, Optional, Dict, List, Union, Callable, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import weakref
import orjson
import redis.asyncio as redis
from pydantic import BaseModel, Field
from typing import Any, List, Dict, Optional
"""
🎯 Advanced Caching Strategies
==============================

Advanced caching strategies including:
- Predictive caching
- Cache warming
- Intelligent cache invalidation
- Cache coherency
- Distributed caching
- Cache analytics
"""



logger = logging.getLogger(__name__)


class CacheStrategy(Enum):
    """Cache strategy types"""
    LRU = "lru"                    # Least Recently Used
    LFU = "lfu"                    # Least Frequently Used
    TTL = "ttl"                    # Time To Live
    PREDICTIVE = "predictive"      # Predictive caching
    ADAPTIVE = "adaptive"          # Adaptive TTL
    WRITE_THROUGH = "write_through"  # Write-through caching
    WRITE_BEHIND = "write_behind"    # Write-behind caching


class CacheEvent(Enum):
    """Cache events for analytics"""
    HIT = "hit"
    MISS = "miss"
    SET = "set"
    DELETE = "delete"
    EXPIRE = "expire"
    INVALIDATE = "invalidate"


@dataclass
class CacheMetrics:
    """Cache performance metrics"""
    hits: int = 0
    misses: int = 0
    sets: int = 0
    deletes: int = 0
    expires: int = 0
    invalidations: int = 0
    total_requests: int = 0
    total_size: int = 0
    max_size: int = 0
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate"""
        return self.hits / max(self.total_requests, 1)
    
    @property
    def miss_rate(self) -> float:
        """Calculate cache miss rate"""
        return self.misses / max(self.total_requests, 1)
    
    @property
    def efficiency(self) -> float:
        """Calculate cache efficiency"""
        return self.hits / max(self.hits + self.misses, 1)


@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    key: str
    value: Any
    created_at: float
    accessed_at: float
    access_count: int = 0
    ttl: Optional[int] = None
    strategy: CacheStrategy = CacheStrategy.TTL
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_expired(self) -> bool:
        """Check if entry is expired"""
        if self.ttl is None:
            return False
        return time.time() > self.created_at + self.ttl
    
    def update_access(self) -> Any:
        """Update access statistics"""
        self.accessed_at = time.time()
        self.access_count += 1
    
    def get_age(self) -> float:
        """Get entry age in seconds"""
        return time.time() - self.created_at
    
    def get_idle_time(self) -> float:
        """Get time since last access"""
        return time.time() - self.accessed_at


class PredictiveCache:
    """
    Predictive cache that learns access patterns and preloads data.
    """
    
    def __init__(self, max_size: int = 1000, prediction_window: int = 100):
        
    """__init__ function."""
self.max_size = max_size
        self.prediction_window = prediction_window
        
        # Cache storage
        self.cache: Dict[str, CacheEntry] = {}
        
        # Access pattern tracking
        self.access_sequence = deque(maxlen=prediction_window)
        self.pattern_frequency = defaultdict(int)
        self.key_relationships = defaultdict(set)
        
        # Metrics
        self.metrics = CacheMetrics(max_size=max_size)
        
        # Prediction model
        self.prediction_threshold = 0.3
        self.prediction_confidence = defaultdict(float)
        
        # Background tasks
        self.prediction_task = None
        self.cleanup_task = None
        
    async def initialize(self) -> Any:
        """Initialize predictive cache"""
        # Start background tasks
        self.prediction_task = asyncio.create_task(self._prediction_loop())
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        logger.info("Predictive cache initialized")
    
    async def cleanup(self) -> Any:
        """Cleanup predictive cache"""
        if self.prediction_task:
            self.prediction_task.cancel()
        if self.cleanup_task:
            self.cleanup_task.cancel()
        
        logger.info("Predictive cache cleaned up")
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache with pattern learning"""
        self.metrics.total_requests += 1
        
        # Record access pattern
        self._record_access(key)
        
        if key in self.cache:
            entry = self.cache[key]
            
            if entry.is_expired():
                # Remove expired entry
                del self.cache[key]
                self.metrics.expires += 1
                self.metrics.misses += 1
                return None
            
            # Update access statistics
            entry.update_access()
            self.metrics.hits += 1
            
            # Trigger predictive loading
            asyncio.create_task(self._predict_and_preload(key))
            
            return entry.value
        
        self.metrics.misses += 1
        return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None,
                 strategy: CacheStrategy = CacheStrategy.PREDICTIVE) -> None:
        """Set value in cache"""
        # Evict if necessary
        if len(self.cache) >= self.max_size:
            await self._evict_entries()
        
        # Create cache entry
        entry = CacheEntry(
            key=key,
            value=value,
            created_at=time.time(),
            accessed_at=time.time(),
            ttl=ttl,
            strategy=strategy
        )
        
        self.cache[key] = entry
        self.metrics.sets += 1
        self.metrics.total_size = len(self.cache)
    
    async def delete(self, key: str) -> None:
        """Delete value from cache"""
        if key in self.cache:
            del self.cache[key]
            self.metrics.deletes += 1
            self.metrics.total_size = len(self.cache)
    
    def _record_access(self, key: str):
        """Record access pattern for prediction"""
        self.access_sequence.append(key)
        
        # Update pattern frequency
        if len(self.access_sequence) >= 2:
            pattern = tuple(self.access_sequence[-2:])
            self.pattern_frequency[pattern] += 1
            
            # Update key relationships
            prev_key = self.access_sequence[-2]
            self.key_relationships[prev_key].add(key)
    
    async def _predict_and_preload(self, current_key: str):
        """Predict and preload likely next keys"""
        try:
            # Find most likely next keys
            likely_keys = self._predict_next_keys(current_key)
            
            for key, confidence in likely_keys:
                if confidence > self.prediction_threshold:
                    # Preload the key if not in cache
                    if key not in self.cache:
                        await self._preload_key(key)
                        
        except Exception as e:
            logger.error(f"Prediction error: {e}")
    
    def _predict_next_keys(self, current_key: str) -> List[Tuple[str, float]]:
        """Predict next likely keys based on patterns"""
        predictions = []
        
        # Check direct relationships
        for related_key in self.key_relationships[current_key]:
            confidence = self._calculate_confidence(current_key, related_key)
            predictions.append((related_key, confidence))
        
        # Check pattern frequency
        for pattern, frequency in self.pattern_frequency.items():
            if pattern[0] == current_key:
                confidence = frequency / max(self.metrics.total_requests, 1)
                predictions.append((pattern[1], confidence))
        
        # Sort by confidence
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:5]  # Return top 5 predictions
    
    def _calculate_confidence(self, from_key: str, to_key: str) -> float:
        """Calculate confidence for key relationship"""
        total_accesses = sum(self.pattern_frequency.values())
        if total_accesses == 0:
            return 0.0
        
        # Count transitions from from_key to to_key
        transition_count = 0
        for pattern, frequency in self.pattern_frequency.items():
            if pattern[0] == from_key and pattern[1] == to_key:
                transition_count += frequency
        
        return transition_count / total_accesses
    
    async def _preload_key(self, key: str):
        """Preload a key (to be implemented by subclasses)"""
        # This should be implemented by subclasses to actually load data
        logger.debug(f"Preloading key: {key}")
    
    async def _evict_entries(self) -> Any:
        """Evict entries based on strategy"""
        if not self.cache:
            return
        
        # Find entries to evict
        evict_candidates = []
        
        for key, entry in self.cache.items():
            score = self._calculate_eviction_score(entry)
            evict_candidates.append((key, score))
        
        # Sort by eviction score (highest first)
        evict_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Evict top candidates
        evict_count = len(self.cache) - self.max_size + 1
        for key, _ in evict_candidates[:evict_count]:
            del self.cache[key]
            self.metrics.invalidations += 1
    
    def _calculate_eviction_score(self, entry: CacheEntry) -> float:
        """Calculate eviction score based on strategy"""
        if entry.strategy == CacheStrategy.LRU:
            return entry.get_idle_time()
        elif entry.strategy == CacheStrategy.LFU:
            return -entry.access_count  # Negative for ascending sort
        elif entry.strategy == CacheStrategy.TTL:
            return entry.get_age()
        else:
            # Default: combination of factors
            return (entry.get_idle_time() * 0.7 + 
                   (1.0 / max(entry.access_count, 1)) * 0.3)
    
    async def _prediction_loop(self) -> Any:
        """Background loop for pattern analysis"""
        while True:
            try:
                await asyncio.sleep(60)  # Run every minute
                
                # Update prediction confidence
                self._update_prediction_confidence()
                
                # Clean old patterns
                self._cleanup_patterns()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Prediction loop error: {e}")
    
    async def _cleanup_loop(self) -> Any:
        """Background loop for cache cleanup"""
        while True:
            try:
                await asyncio.sleep(30)  # Run every 30 seconds
                
                # Remove expired entries
                expired_keys = [
                    key for key, entry in self.cache.items()
                    if entry.is_expired()
                ]
                
                for key in expired_keys:
                    del self.cache[key]
                    self.metrics.expires += 1
                
                self.metrics.total_size = len(self.cache)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")
    
    def _update_prediction_confidence(self) -> Any:
        """Update prediction confidence scores"""
        total_patterns = sum(self.pattern_frequency.values())
        if total_patterns == 0:
            return
        
        for pattern in self.pattern_frequency:
            frequency = self.pattern_frequency[pattern]
            confidence = frequency / total_patterns
            self.prediction_confidence[pattern] = confidence
    
    def _cleanup_patterns(self) -> Any:
        """Clean up old patterns"""
        # Remove low-frequency patterns
        min_frequency = 2
        patterns_to_remove = [
            pattern for pattern, frequency in self.pattern_frequency.items()
            if frequency < min_frequency
        ]
        
        for pattern in patterns_to_remove:
            del self.pattern_frequency[pattern]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            "metrics": {
                "hits": self.metrics.hits,
                "misses": self.metrics.misses,
                "hit_rate": self.metrics.hit_rate,
                "total_requests": self.metrics.total_requests,
                "total_size": self.metrics.total_size,
                "max_size": self.metrics.max_size
            },
            "patterns": {
                "total_patterns": len(self.pattern_frequency),
                "prediction_confidence": dict(self.prediction_confidence),
                "key_relationships": {
                    k: len(v) for k, v in self.key_relationships.items()
                }
            }
        }


class CacheWarmer:
    """
    Cache warmer that preloads frequently accessed data.
    """
    
    def __init__(self, cache: PredictiveCache):
        
    """__init__ function."""
self.cache = cache
        self.warmup_data = {}
        self.warmup_schedule = {}
        self.warmup_task = None
    
    async def initialize(self) -> Any:
        """Initialize cache warmer"""
        self.warmup_task = asyncio.create_task(self._warmup_loop())
        logger.info("Cache warmer initialized")
    
    async def cleanup(self) -> Any:
        """Cleanup cache warmer"""
        if self.warmup_task:
            self.warmup_task.cancel()
        logger.info("Cache warmer cleaned up")
    
    def add_warmup_data(self, key: str, loader_func: Callable, 
                       schedule: str = "on_startup"):
        """Add data to warmup schedule"""
        self.warmup_data[key] = loader_func
        self.warmup_schedule[key] = schedule
    
    async def warm_cache(self, keys: List[str] = None):
        """Warm cache with specified keys or all scheduled keys"""
        if keys is None:
            keys = list(self.warmup_data.keys())
        
        warmup_tasks = []
        for key in keys:
            if key in self.warmup_data:
                task = asyncio.create_task(self._warmup_key(key))
                warmup_tasks.append(task)
        
        if warmup_tasks:
            await asyncio.gather(*warmup_tasks, return_exceptions=True)
            logger.info(f"Cache warmed with {len(warmup_tasks)} keys")
    
    async def _warmup_key(self, key: str):
        """Warm up a specific key"""
        try:
            loader_func = self.warmup_data[key]
            
            if asyncio.iscoroutinefunction(loader_func):
                value = await loader_func()
            else:
                loop = asyncio.get_event_loop()
                value = await loop.run_in_executor(None, loader_func)
            
            await self.cache.set(key, value, ttl=3600)  # 1 hour TTL
            logger.debug(f"Warmed cache key: {key}")
            
        except Exception as e:
            logger.error(f"Failed to warm cache key {key}: {e}")
    
    async def _warmup_loop(self) -> Any:
        """Background loop for scheduled warmup"""
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                
                # Find keys scheduled for periodic warmup
                periodic_keys = [
                    key for key, schedule in self.warmup_schedule.items()
                    if schedule == "periodic"
                ]
                
                if periodic_keys:
                    await self.warm_cache(periodic_keys)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Warmup loop error: {e}")


class CacheInvalidator:
    """
    Intelligent cache invalidator with pattern-based invalidation.
    """
    
    def __init__(self, cache: PredictiveCache):
        
    """__init__ function."""
self.cache = cache
        self.invalidation_patterns = defaultdict(set)
        self.invalidation_rules = []
    
    def add_invalidation_rule(self, pattern: str, keys: List[str]):
        """Add invalidation rule"""
        self.invalidation_patterns[pattern] = set(keys)
    
    def add_conditional_rule(self, condition_func: Callable, keys: List[str]):
        """Add conditional invalidation rule"""
        self.invalidation_rules.append((condition_func, keys))
    
    async def invalidate_pattern(self, pattern: str):
        """Invalidate keys matching pattern"""
        if pattern in self.invalidation_patterns:
            keys_to_invalidate = self.invalidation_patterns[pattern]
            await self._invalidate_keys(keys_to_invalidate)
    
    async def invalidate_conditional(self, context: Dict[str, Any]):
        """Invalidate based on conditional rules"""
        keys_to_invalidate = set()
        
        for condition_func, keys in self.invalidation_rules:
            try:
                if condition_func(context):
                    keys_to_invalidate.update(keys)
            except Exception as e:
                logger.error(f"Condition evaluation error: {e}")
        
        if keys_to_invalidate:
            await self._invalidate_keys(keys_to_invalidate)
    
    async def invalidate_related(self, key: str):
        """Invalidate keys related to the given key"""
        # Find related keys based on patterns
        related_keys = set()
        
        for pattern, keys in self.invalidation_patterns.items():
            if key in pattern or any(key in k for k in keys):
                related_keys.update(keys)
        
        if related_keys:
            await self._invalidate_keys(related_keys)
    
    async def _invalidate_keys(self, keys: Set[str]):
        """Invalidate specific keys"""
        for key in keys:
            await self.cache.delete(key)
        
        logger.info(f"Invalidated {len(keys)} cache keys")


class DistributedCache:
    """
    Distributed cache with Redis backend and local caching.
    """
    
    def __init__(self, redis_url: str = "redis://localhost:6379/0",
                 local_cache_size: int = 1000):
        
    """__init__ function."""
self.redis_url = redis_url
        self.redis_client = None
        self.local_cache = PredictiveCache(max_size=local_cache_size)
        self.distribution_key = hashlib.md5(str(time.time()).encode()).hexdigest()
    
    async def initialize(self) -> Any:
        """Initialize distributed cache"""
        # Initialize Redis client
        self.redis_client = redis.from_url(self.redis_url)
        await self.redis_client.ping()
        
        # Initialize local cache
        await self.local_cache.initialize()
        
        logger.info("Distributed cache initialized")
    
    async def cleanup(self) -> Any:
        """Cleanup distributed cache"""
        if self.redis_client:
            await self.redis_client.close()
        
        await self.local_cache.cleanup()
        
        logger.info("Distributed cache cleaned up")
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from distributed cache"""
        # Try local cache first
        local_value = await self.local_cache.get(key)
        if local_value is not None:
            return local_value
        
        # Try Redis cache
        if self.redis_client:
            try:
                redis_value = await self.redis_client.get(key)
                if redis_value is not None:
                    # Deserialize and cache locally
                    value = orjson.loads(redis_value)
                    await self.local_cache.set(key, value)
                    return value
            except Exception as e:
                logger.error(f"Redis get error: {e}")
        
        return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in distributed cache"""
        # Set in local cache
        await self.local_cache.set(key, value, ttl)
        
        # Set in Redis cache
        if self.redis_client:
            try:
                serialized_value = orjson.dumps(value)
                if ttl:
                    await self.redis_client.setex(key, ttl, serialized_value)
                else:
                    await self.redis_client.set(key, serialized_value)
            except Exception as e:
                logger.error(f"Redis set error: {e}")
    
    async def delete(self, key: str) -> None:
        """Delete value from distributed cache"""
        # Delete from local cache
        await self.local_cache.delete(key)
        
        # Delete from Redis cache
        if self.redis_client:
            try:
                await self.redis_client.delete(key)
            except Exception as e:
                logger.error(f"Redis delete error: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get distributed cache statistics"""
        return {
            "local_cache": self.local_cache.get_stats(),
            "distribution_key": self.distribution_key,
            "redis_available": self.redis_client is not None
        }


# Example usage
async def example_usage():
    """Example of how to use the advanced caching strategies"""
    
    # Create predictive cache
    cache = PredictiveCache(max_size=1000)
    await cache.initialize()
    
    # Create cache warmer
    warmer = CacheWarmer(cache)
    await warmer.initialize()
    
    # Create cache invalidator
    invalidator = CacheInvalidator(cache)
    
    # Add warmup data
    def load_user_data(user_id: str):
        
    """load_user_data function."""
return {"user_id": user_id, "data": "user_data"}
    
    warmer.add_warmup_data("user:1", lambda: load_user_data("1"))
    warmer.add_warmup_data("user:2", lambda: load_user_data("2"))
    
    # Add invalidation rules
    invalidator.add_invalidation_rule("user:*", ["user:1", "user:2", "user:3"])
    
    # Warm cache
    await warmer.warm_cache()
    
    # Use cache
    await cache.set("test_key", "test_value", ttl=3600)
    value = await cache.get("test_key")
    
    # Get statistics
    stats = cache.get_stats()
    print("Cache Stats:", stats)
    
    # Cleanup
    await warmer.cleanup()
    await cache.cleanup()


match __name__:
    case "__main__":
    asyncio.run(example_usage()) 