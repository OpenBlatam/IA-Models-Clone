"""
Intelligent Caching System with ML-based Optimization
Following functional programming principles and advanced caching strategies
"""

import asyncio
import time
import hashlib
import json
import pickle
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import logging
from collections import OrderedDict, defaultdict
import weakref

logger = logging.getLogger(__name__)


# Pure functions for intelligent caching

class CacheStrategy(str, Enum):
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    TTL = "ttl"  # Time To Live
    ML_PREDICTIVE = "ml_predictive"  # ML-based predictive
    ADAPTIVE = "adaptive"  # Adaptive strategy


class CacheItemType(str, Enum):
    POST_CONTENT = "post_content"
    AI_RESPONSE = "ai_response"
    ANALYSIS_RESULT = "analysis_result"
    OPTIMIZATION_RESULT = "optimization_result"
    USER_SESSION = "user_session"
    METRICS_DATA = "metrics_data"


@dataclass(frozen=True)
class CacheItem:
    """Immutable cache item - pure data structure"""
    key: str
    value: Any
    item_type: CacheItemType
    created_at: datetime
    last_accessed: datetime
    access_count: int
    size_bytes: int
    ttl_seconds: int
    priority: float
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary - pure function"""
        return {
            "key": self.key,
            "value": self.value,
            "item_type": self.item_type.value,
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
            "access_count": self.access_count,
            "size_bytes": self.size_bytes,
            "ttl_seconds": self.ttl_seconds,
            "priority": self.priority,
            "metadata": self.metadata
        }


@dataclass(frozen=True)
class CacheMetrics:
    """Immutable cache metrics - pure data structure"""
    total_items: int
    total_size_bytes: int
    hit_count: int
    miss_count: int
    eviction_count: int
    hit_rate: float
    miss_rate: float
    average_access_time: float
    memory_usage_percent: float
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary - pure function"""
        return {
            "total_items": self.total_items,
            "total_size_bytes": self.total_size_bytes,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "eviction_count": self.eviction_count,
            "hit_rate": self.hit_rate,
            "miss_rate": self.miss_rate,
            "average_access_time": self.average_access_time,
            "memory_usage_percent": self.memory_usage_percent,
            "timestamp": self.timestamp.isoformat()
        }


def create_cache_key(
    prefix: str,
    identifier: str,
    parameters: Optional[Dict[str, Any]] = None
) -> str:
    """Create cache key - pure function"""
    key_data = {
        "prefix": prefix,
        "identifier": identifier,
        "parameters": parameters or {}
    }
    
    key_string = json.dumps(key_data, sort_keys=True)
    return f"{prefix}:{hashlib.md5(key_string.encode()).hexdigest()}"


def calculate_item_priority(
    access_count: int,
    last_accessed: datetime,
    item_type: CacheItemType,
    size_bytes: int,
    base_priority: float = 1.0
) -> float:
    """Calculate cache item priority - pure function"""
    # Time decay factor (more recent = higher priority)
    time_decay = max(0.1, 1.0 - (datetime.utcnow() - last_accessed).total_seconds() / 3600)
    
    # Access frequency factor
    frequency_factor = min(2.0, 1.0 + (access_count / 10))
    
    # Size factor (smaller items = higher priority)
    size_factor = max(0.5, 1.0 - (size_bytes / (1024 * 1024)))  # 1MB baseline
    
    # Type priority weights
    type_weights = {
        CacheItemType.POST_CONTENT: 1.0,
        CacheItemType.AI_RESPONSE: 1.2,
        CacheItemType.ANALYSIS_RESULT: 1.1,
        CacheItemType.OPTIMIZATION_RESULT: 1.3,
        CacheItemType.USER_SESSION: 0.8,
        CacheItemType.METRICS_DATA: 0.9
    }
    
    type_weight = type_weights.get(item_type, 1.0)
    
    # Calculate final priority
    priority = base_priority * time_decay * frequency_factor * size_factor * type_weight
    
    return max(0.1, min(10.0, priority))


def calculate_item_size(value: Any) -> int:
    """Calculate item size in bytes - pure function"""
    try:
        if isinstance(value, (str, int, float, bool)):
            return len(str(value).encode('utf-8'))
        elif isinstance(value, (dict, list)):
            return len(json.dumps(value).encode('utf-8'))
        else:
            return len(pickle.dumps(value))
    except Exception:
        return 1024  # Default size if calculation fails


def is_item_expired(item: CacheItem) -> bool:
    """Check if cache item is expired - pure function"""
    if item.ttl_seconds <= 0:
        return False  # No expiration
    
    age_seconds = (datetime.utcnow() - item.created_at).total_seconds()
    return age_seconds > item.ttl_seconds


def predict_access_probability(
    item: CacheItem,
    access_patterns: Dict[str, List[datetime]],
    current_time: datetime
) -> float:
    """Predict access probability using simple ML - pure function"""
    if not access_patterns or item.key not in access_patterns:
        return 0.5  # Default probability
    
    access_times = access_patterns[item.key]
    if len(access_times) < 2:
        return 0.5
    
    # Simple pattern analysis
    recent_accesses = [t for t in access_times if (current_time - t).total_seconds() < 3600]
    
    # Time-based pattern
    hour = current_time.hour
    historical_hours = [t.hour for t in access_times]
    hour_frequency = historical_hours.count(hour) / len(historical_hours)
    
    # Recency factor
    last_access = max(access_times)
    recency_factor = max(0.1, 1.0 - (current_time - last_access).total_seconds() / 3600)
    
    # Frequency factor
    frequency_factor = min(1.0, len(recent_accesses) / 10)
    
    # Calculate probability
    probability = (hour_frequency * 0.4 + recency_factor * 0.3 + frequency_factor * 0.3)
    
    return max(0.0, min(1.0, probability))


def select_eviction_candidates(
    items: Dict[str, CacheItem],
    max_items: int,
    strategy: CacheStrategy
) -> List[str]:
    """Select items for eviction - pure function"""
    if len(items) <= max_items:
        return []
    
    items_to_evict = len(items) - max_items
    
    if strategy == CacheStrategy.LRU:
        # Sort by last accessed time (oldest first)
        sorted_items = sorted(
            items.items(),
            key=lambda x: x[1].last_accessed
        )
    elif strategy == CacheStrategy.LFU:
        # Sort by access count (least frequent first)
        sorted_items = sorted(
            items.items(),
            key=lambda x: x[1].access_count
        )
    elif strategy == CacheStrategy.ML_PREDICTIVE:
        # Sort by predicted access probability (lowest first)
        # This is a simplified version - in practice, you'd use a trained model
        sorted_items = sorted(
            items.items(),
            key=lambda x: x[1].priority
        )
    else:  # TTL or ADAPTIVE
        # Sort by TTL (expired first, then by priority)
        sorted_items = sorted(
            items.items(),
            key=lambda x: (x[1].ttl_seconds, -x[1].priority)
        )
    
    return [key for key, _ in sorted_items[:items_to_evict]]


# Intelligent Cache System Class

class IntelligentCacheSystem:
    """Intelligent Caching System with ML-based optimization"""
    
    def __init__(
        self,
        max_size_bytes: int = 100 * 1024 * 1024,  # 100MB
        max_items: int = 10000,
        default_ttl: int = 3600,  # 1 hour
        strategy: CacheStrategy = CacheStrategy.ADAPTIVE
    ):
        self.max_size_bytes = max_size_bytes
        self.max_items = max_items
        self.default_ttl = default_ttl
        self.strategy = strategy
        
        # Cache storage
        self.cache: Dict[str, CacheItem] = OrderedDict()
        self.access_patterns: Dict[str, List[datetime]] = defaultdict(list)
        
        # Metrics
        self.metrics = {
            "hit_count": 0,
            "miss_count": 0,
            "eviction_count": 0,
            "total_size_bytes": 0,
            "access_times": []
        }
        
        # ML features
        self.access_predictor = None
        self.performance_history: List[float] = []
        
        # Background tasks
        self.cleanup_task: Optional[asyncio.Task] = None
        self.optimization_task: Optional[asyncio.Task] = None
        self.is_running = False
    
    async def start(self) -> None:
        """Start intelligent cache system"""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Start background tasks
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
        self.optimization_task = asyncio.create_task(self._optimization_loop())
        
        logger.info("Intelligent cache system started")
    
    async def stop(self) -> None:
        """Stop intelligent cache system"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Stop background tasks
        if self.cleanup_task:
            self.cleanup_task.cancel()
        if self.optimization_task:
            self.optimization_task.cancel()
        
        logger.info("Intelligent cache system stopped")
    
    async def _cleanup_loop(self) -> None:
        """Background cleanup loop"""
        while self.is_running:
            try:
                await self._cleanup_expired_items()
                await self._evict_items_if_needed()
                await asyncio.sleep(60)  # Run every minute
            except Exception as e:
                logger.error("Error in cache cleanup loop", error=str(e))
                await asyncio.sleep(10)
    
    async def _optimization_loop(self) -> None:
        """Background optimization loop"""
        while self.is_running:
            try:
                await self._optimize_cache_strategy()
                await asyncio.sleep(300)  # Run every 5 minutes
            except Exception as e:
                logger.error("Error in cache optimization loop", error=str(e))
                await asyncio.sleep(30)
    
    async def get(self, key: str) -> Optional[Any]:
        """Get item from cache"""
        start_time = time.time()
        
        try:
            if key not in self.cache:
                self.metrics["miss_count"] += 1
                return None
            
            item = self.cache[key]
            
            # Check if expired
            if is_item_expired(item):
                del self.cache[key]
                self.metrics["miss_count"] += 1
                self.metrics["eviction_count"] += 1
                return None
            
            # Update access information
            updated_item = CacheItem(
                key=item.key,
                value=item.value,
                item_type=item.item_type,
                created_at=item.created_at,
                last_accessed=datetime.utcnow(),
                access_count=item.access_count + 1,
                size_bytes=item.size_bytes,
                ttl_seconds=item.ttl_seconds,
                priority=calculate_item_priority(
                    item.access_count + 1,
                    datetime.utcnow(),
                    item.item_type,
                    item.size_bytes
                ),
                metadata=item.metadata
            )
            
            self.cache[key] = updated_item
            
            # Record access pattern
            self.access_patterns[key].append(datetime.utcnow())
            
            # Update metrics
            self.metrics["hit_count"] += 1
            access_time = time.time() - start_time
            self.metrics["access_times"].append(access_time)
            
            # Keep only recent access times
            if len(self.metrics["access_times"]) > 1000:
                self.metrics["access_times"] = self.metrics["access_times"][-500:]
            
            logger.debug(f"Cache hit for key: {key}")
            return item.value
            
        except Exception as e:
            logger.error(f"Error getting cache item {key}", error=str(e))
            self.metrics["miss_count"] += 1
            return None
    
    async def set(
        self,
        key: str,
        value: Any,
        item_type: CacheItemType = CacheItemType.POST_CONTENT,
        ttl_seconds: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Set item in cache"""
        try:
            # Calculate size
            size_bytes = calculate_item_size(value)
            
            # Check if item would exceed size limit
            if size_bytes > self.max_size_bytes:
                logger.warning(f"Item too large for cache: {size_bytes} bytes")
                return False
            
            # Create cache item
            item = CacheItem(
                key=key,
                value=value,
                item_type=item_type,
                created_at=datetime.utcnow(),
                last_accessed=datetime.utcnow(),
                access_count=1,
                size_bytes=size_bytes,
                ttl_seconds=ttl_seconds or self.default_ttl,
                priority=calculate_item_priority(
                    1, datetime.utcnow(), item_type, size_bytes
                ),
                metadata=metadata or {}
            )
            
            # Remove existing item if present
            if key in self.cache:
                old_item = self.cache[key]
                self.metrics["total_size_bytes"] -= old_item.size_bytes
                del self.cache[key]
            
            # Add new item
            self.cache[key] = item
            self.metrics["total_size_bytes"] += size_bytes
            
            # Evict items if needed
            await self._evict_items_if_needed()
            
            logger.debug(f"Cache set for key: {key}")
            return True
            
        except Exception as e:
            logger.error(f"Error setting cache item {key}", error=str(e))
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete item from cache"""
        try:
            if key in self.cache:
                item = self.cache[key]
                self.metrics["total_size_bytes"] -= item.size_bytes
                del self.cache[key]
                logger.debug(f"Cache delete for key: {key}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error deleting cache item {key}", error=str(e))
            return False
    
    async def clear(self) -> None:
        """Clear all cache items"""
        try:
            self.cache.clear()
            self.metrics["total_size_bytes"] = 0
            logger.info("Cache cleared")
        except Exception as e:
            logger.error("Error clearing cache", error=str(e))
    
    async def _cleanup_expired_items(self) -> None:
        """Clean up expired items"""
        try:
            expired_keys = []
            
            for key, item in self.cache.items():
                if is_item_expired(item):
                    expired_keys.append(key)
            
            for key in expired_keys:
                item = self.cache[key]
                self.metrics["total_size_bytes"] -= item.size_bytes
                del self.cache[key]
                self.metrics["eviction_count"] += 1
            
            if expired_keys:
                logger.info(f"Cleaned up {len(expired_keys)} expired items")
                
        except Exception as e:
            logger.error("Error cleaning up expired items", error=str(e))
    
    async def _evict_items_if_needed(self) -> None:
        """Evict items if cache limits are exceeded"""
        try:
            # Check size limit
            while (self.metrics["total_size_bytes"] > self.max_size_bytes or 
                   len(self.cache) > self.max_items):
                
                candidates = select_eviction_candidates(
                    self.cache, self.max_items, self.strategy
                )
                
                if not candidates:
                    break
                
                # Evict first candidate
                key_to_evict = candidates[0]
                item = self.cache[key_to_evict]
                
                self.metrics["total_size_bytes"] -= item.size_bytes
                del self.cache[key_to_evict]
                self.metrics["eviction_count"] += 1
                
                logger.debug(f"Evicted item: {key_to_evict}")
                
        except Exception as e:
            logger.error("Error evicting items", error=str(e))
    
    async def _optimize_cache_strategy(self) -> None:
        """Optimize cache strategy based on performance"""
        try:
            # Calculate current performance
            hit_rate = self.get_hit_rate()
            self.performance_history.append(hit_rate)
            
            # Keep only recent performance data
            if len(self.performance_history) > 100:
                self.performance_history = self.performance_history[-50:]
            
            # Adaptive strategy selection
            if len(self.performance_history) >= 10:
                recent_avg = sum(self.performance_history[-10:]) / 10
                overall_avg = sum(self.performance_history) / len(self.performance_history)
                
                # If recent performance is significantly worse, try different strategy
                if recent_avg < overall_avg * 0.9:
                    await self._switch_strategy()
            
        except Exception as e:
            logger.error("Error optimizing cache strategy", error=str(e))
    
    async def _switch_strategy(self) -> None:
        """Switch to a different cache strategy"""
        try:
            strategies = [CacheStrategy.LRU, CacheStrategy.LFU, CacheStrategy.ML_PREDICTIVE]
            current_index = strategies.index(self.strategy)
            next_index = (current_index + 1) % len(strategies)
            
            old_strategy = self.strategy
            self.strategy = strategies[next_index]
            
            logger.info(f"Switched cache strategy from {old_strategy} to {self.strategy}")
            
        except Exception as e:
            logger.error("Error switching cache strategy", error=str(e))
    
    def get_hit_rate(self) -> float:
        """Get cache hit rate - pure function"""
        total_requests = self.metrics["hit_count"] + self.metrics["miss_count"]
        if total_requests == 0:
            return 0.0
        return self.metrics["hit_count"] / total_requests
    
    def get_miss_rate(self) -> float:
        """Get cache miss rate - pure function"""
        return 1.0 - self.get_hit_rate()
    
    def get_average_access_time(self) -> float:
        """Get average access time - pure function"""
        access_times = self.metrics["access_times"]
        if not access_times:
            return 0.0
        return sum(access_times) / len(access_times)
    
    def get_memory_usage_percent(self) -> float:
        """Get memory usage percentage - pure function"""
        return (self.metrics["total_size_bytes"] / self.max_size_bytes) * 100
    
    def get_cache_metrics(self) -> CacheMetrics:
        """Get cache metrics"""
        return CacheMetrics(
            total_items=len(self.cache),
            total_size_bytes=self.metrics["total_size_bytes"],
            hit_count=self.metrics["hit_count"],
            miss_count=self.metrics["miss_count"],
            eviction_count=self.metrics["eviction_count"],
            hit_rate=self.get_hit_rate(),
            miss_rate=self.get_miss_rate(),
            average_access_time=self.get_average_access_time(),
            memory_usage_percent=self.get_memory_usage_percent(),
            timestamp=datetime.utcnow()
        )
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get detailed cache statistics"""
        metrics = self.get_cache_metrics()
        
        # Item type distribution
        type_distribution = defaultdict(int)
        for item in self.cache.values():
            type_distribution[item.item_type.value] += 1
        
        # Size distribution
        size_ranges = {
            "small": 0,    # < 1KB
            "medium": 0,   # 1KB - 10KB
            "large": 0     # > 10KB
        }
        
        for item in self.cache.values():
            if item.size_bytes < 1024:
                size_ranges["small"] += 1
            elif item.size_bytes < 10240:
                size_ranges["medium"] += 1
            else:
                size_ranges["large"] += 1
        
        return {
            "metrics": metrics.to_dict(),
            "strategy": self.strategy.value,
            "type_distribution": dict(type_distribution),
            "size_distribution": size_ranges,
            "performance_history": self.performance_history[-20:],  # Last 20 measurements
            "timestamp": datetime.utcnow().isoformat()
        }


# Factory functions

def create_intelligent_cache(
    max_size_bytes: int = 100 * 1024 * 1024,
    max_items: int = 10000,
    default_ttl: int = 3600,
    strategy: CacheStrategy = CacheStrategy.ADAPTIVE
) -> IntelligentCacheSystem:
    """Create intelligent cache system - pure function"""
    return IntelligentCacheSystem(max_size_bytes, max_items, default_ttl, strategy)


async def get_intelligent_cache() -> IntelligentCacheSystem:
    """Get intelligent cache system instance"""
    cache = create_intelligent_cache()
    await cache.start()
    return cache

