"""
Advanced Caching System
======================

Multi-level caching system with intelligent eviction and optimization.
"""

import asyncio
import logging
import time
import hashlib
import pickle
import json
from typing import Dict, List, Optional, Any, Union, Callable, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import OrderedDict, defaultdict
import threading
import weakref
from functools import wraps
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import redis
import aioredis

logger = logging.getLogger(__name__)

class CacheLevel(str, Enum):
    """Cache levels."""
    L1 = "l1"  # In-memory cache
    L2 = "l2"  # Redis cache
    L3 = "l3"  # Disk cache

class EvictionPolicy(str, Enum):
    """Cache eviction policies."""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    TTL = "ttl"  # Time To Live
    SIZE = "size"  # Size-based

@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    value: Any
    created_at: datetime
    accessed_at: datetime
    access_count: int = 0
    size: int = 0
    ttl: Optional[int] = None
    level: CacheLevel = CacheLevel.L1
    compressed: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CacheStats:
    """Cache statistics."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    size: int = 0
    memory_usage: int = 0
    hit_rate: float = 0.0
    miss_rate: float = 0.0

class L1Cache:
    """
    Level 1 cache (in-memory).
    
    Features:
    - Fast access
    - Limited size
    - Multiple eviction policies
    - Compression support
    """
    
    def __init__(self, max_size: int = 10000, eviction_policy: EvictionPolicy = EvictionPolicy.LRU):
        self.max_size = max_size
        self.eviction_policy = eviction_policy
        self.cache = OrderedDict()
        self.access_times = {}
        self.access_counts = defaultdict(int)
        self.stats = CacheStats()
        self.lock = threading.RLock()
        self.compression_enabled = True
        
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self.lock:
            if key not in self.cache:
                self.stats.misses += 1
                return None
            
            entry = self.cache[key]
            
            # Check TTL
            if entry.ttl and time.time() - entry.created_at.timestamp() > entry.ttl:
                self._evict(key)
                self.stats.misses += 1
                return None
            
            # Update access info
            entry.accessed_at = datetime.utcnow()
            entry.access_count += 1
            self.access_times[key] = time.time()
            self.access_counts[key] += 1
            
            # Move to end (LRU)
            self.cache.move_to_end(key)
            
            self.stats.hits += 1
            return entry.value
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Set value in cache."""
        with self.lock:
            try:
                # Calculate size
                size = self._calculate_size(value)
                
                # Create entry
                entry = CacheEntry(
                    key=key,
                    value=value,
                    created_at=datetime.utcnow(),
                    accessed_at=datetime.utcnow(),
                    size=size,
                    ttl=ttl,
                    level=CacheLevel.L1,
                    metadata=metadata or {}
                )
                
                # Check if we need to evict
                if len(self.cache) >= self.max_size and key not in self.cache:
                    self._evict_entries()
                
                # Set entry
                self.cache[key] = entry
                self.access_times[key] = time.time()
                self.access_counts[key] = 1
                
                self.stats.size = len(self.cache)
                return True
                
            except Exception as e:
                logger.error(f"Failed to set cache entry {key}: {str(e)}")
                return False
    
    def delete(self, key: str) -> bool:
        """Delete value from cache."""
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                self.access_times.pop(key, None)
                self.access_counts.pop(key, None)
                self.stats.size = len(self.cache)
                return True
            return False
    
    def clear(self):
        """Clear all cache entries."""
        with self.lock:
            self.cache.clear()
            self.access_times.clear()
            self.access_counts.clear()
            self.stats.size = 0
    
    def _calculate_size(self, value: Any) -> int:
        """Calculate size of value."""
        try:
            return len(pickle.dumps(value))
        except:
            return sys.getsizeof(value)
    
    def _evict_entries(self):
        """Evict entries based on policy."""
        if not self.cache:
            return
        
        if self.eviction_policy == EvictionPolicy.LRU:
            # Remove least recently used
            oldest_key = next(iter(self.cache))
            self._evict(oldest_key)
        elif self.eviction_policy == EvictionPolicy.LFU:
            # Remove least frequently used
            least_frequent_key = min(self.access_counts.keys(), key=lambda k: self.access_counts[k])
            self._evict(least_frequent_key)
        elif self.eviction_policy == EvictionPolicy.SIZE:
            # Remove largest entries
            largest_key = max(self.cache.keys(), key=lambda k: self.cache[k].size)
            self._evict(largest_key)
    
    def _evict(self, key: str):
        """Evict specific key."""
        if key in self.cache:
            del self.cache[key]
            self.access_times.pop(key, None)
            self.access_counts.pop(key, None)
            self.stats.evictions += 1
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        with self.lock:
            total_requests = self.stats.hits + self.stats.misses
            if total_requests > 0:
                self.stats.hit_rate = self.stats.hits / total_requests
                self.stats.miss_rate = self.stats.misses / total_requests
            
            return self.stats

class L2Cache:
    """
    Level 2 cache (Redis).
    
    Features:
    - Distributed caching
    - Persistence
    - TTL support
    - Compression
    """
    
    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        self.redis_url = redis_url
        self.redis = None
        self.stats = CacheStats()
        self.compression_enabled = True
        
    async def initialize(self):
        """Initialize Redis connection."""
        try:
            self.redis = aioredis.from_url(self.redis_url)
            await self.redis.ping()
            logger.info("L2 Cache (Redis) initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize L2 Cache: {str(e)}")
            raise
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from Redis cache."""
        try:
            if not self.redis:
                return None
            
            data = await self.redis.get(key)
            if data is None:
                self.stats.misses += 1
                return None
            
            # Deserialize
            value = pickle.loads(data)
            self.stats.hits += 1
            return value
            
        except Exception as e:
            logger.error(f"Failed to get from L2 cache {key}: {str(e)}")
            self.stats.misses += 1
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in Redis cache."""
        try:
            if not self.redis:
                return False
            
            # Serialize
            data = pickle.dumps(value)
            
            # Set with TTL
            if ttl:
                await self.redis.setex(key, ttl, data)
            else:
                await self.redis.set(key, data)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to set L2 cache {key}: {str(e)}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete value from Redis cache."""
        try:
            if not self.redis:
                return False
            
            result = await self.redis.delete(key)
            return result > 0
            
        except Exception as e:
            logger.error(f"Failed to delete from L2 cache {key}: {str(e)}")
            return False
    
    async def clear(self):
        """Clear all cache entries."""
        try:
            if not self.redis:
                return
            
            await self.redis.flushdb()
            
        except Exception as e:
            logger.error(f"Failed to clear L2 cache: {str(e)}")
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        return self.stats

class L3Cache:
    """
    Level 3 cache (disk).
    
    Features:
    - Persistent storage
    - Large capacity
    - Compression
    - File-based storage
    """
    
    def __init__(self, cache_dir: str = "./cache"):
        self.cache_dir = cache_dir
        self.stats = CacheStats()
        self.compression_enabled = True
        
    async def initialize(self):
        """Initialize disk cache."""
        try:
            import os
            os.makedirs(self.cache_dir, exist_ok=True)
            logger.info("L3 Cache (Disk) initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize L3 Cache: {str(e)}")
            raise
    
    def _get_file_path(self, key: str) -> str:
        """Get file path for key."""
        # Create hash of key for filename
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return f"{self.cache_dir}/{key_hash}.cache"
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from disk cache."""
        try:
            file_path = self._get_file_path(key)
            
            if not os.path.exists(file_path):
                self.stats.misses += 1
                return None
            
            with open(file_path, 'rb') as f:
                data = f.read()
            
            # Deserialize
            value = pickle.loads(data)
            self.stats.hits += 1
            return value
            
        except Exception as e:
            logger.error(f"Failed to get from L3 cache {key}: {str(e)}")
            self.stats.misses += 1
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in disk cache."""
        try:
            file_path = self._get_file_path(key)
            
            # Serialize
            data = pickle.dumps(value)
            
            with open(file_path, 'wb') as f:
                f.write(data)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to set L3 cache {key}: {str(e)}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete value from disk cache."""
        try:
            file_path = self._get_file_path(key)
            
            if os.path.exists(file_path):
                os.remove(file_path)
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to delete from L3 cache {key}: {str(e)}")
            return False
    
    async def clear(self):
        """Clear all cache entries."""
        try:
            import shutil
            if os.path.exists(self.cache_dir):
                shutil.rmtree(self.cache_dir)
                os.makedirs(self.cache_dir, exist_ok=True)
                
        except Exception as e:
            logger.error(f"Failed to clear L3 cache: {str(e)}")
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        return self.stats

class AdvancedCache:
    """
    Multi-level cache system.
    
    Combines L1 (memory), L2 (Redis), and L3 (disk) caches
    with intelligent fallback and optimization.
    """
    
    def __init__(self, 
                 l1_max_size: int = 10000,
                 l1_eviction_policy: EvictionPolicy = EvictionPolicy.LRU,
                 l2_redis_url: str = "redis://localhost:6379/0",
                 l3_cache_dir: str = "./cache"):
        
        self.l1_cache = L1Cache(l1_max_size, l1_eviction_policy)
        self.l2_cache = L2Cache(l2_redis_url)
        self.l3_cache = L3Cache(l3_cache_dir)
        
        self.enabled_levels = [CacheLevel.L1, CacheLevel.L2, CacheLevel.L3]
        self.compression_enabled = True
        self.stats = CacheStats()
        
    async def initialize(self):
        """Initialize all cache levels."""
        try:
            await self.l2_cache.initialize()
            await self.l3_cache.initialize()
            logger.info("Advanced Cache initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Advanced Cache: {str(e)}")
            raise
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache with fallback."""
        # Try L1 first
        if CacheLevel.L1 in self.enabled_levels:
            value = self.l1_cache.get(key)
            if value is not None:
                return value
        
        # Try L2
        if CacheLevel.L2 in self.enabled_levels:
            value = await self.l2_cache.get(key)
            if value is not None:
                # Promote to L1
                self.l1_cache.set(key, value)
                return value
        
        # Try L3
        if CacheLevel.L3 in self.enabled_levels:
            value = await self.l3_cache.get(key)
            if value is not None:
                # Promote to L1 and L2
                self.l1_cache.set(key, value)
                await self.l2_cache.set(key, value)
                return value
        
        return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None, level: Optional[CacheLevel] = None) -> bool:
        """Set value in cache."""
        try:
            if level is None:
                # Set in all enabled levels
                success = True
                
                if CacheLevel.L1 in self.enabled_levels:
                    success &= self.l1_cache.set(key, value, ttl)
                
                if CacheLevel.L2 in self.enabled_levels:
                    success &= await self.l2_cache.set(key, value, ttl)
                
                if CacheLevel.L3 in self.enabled_levels:
                    success &= await self.l3_cache.set(key, value, ttl)
                
                return success
            else:
                # Set in specific level
                if level == CacheLevel.L1:
                    return self.l1_cache.set(key, value, ttl)
                elif level == CacheLevel.L2:
                    return await self.l2_cache.set(key, value, ttl)
                elif level == CacheLevel.L3:
                    return await self.l3_cache.set(key, value, ttl)
                
                return False
                
        except Exception as e:
            logger.error(f"Failed to set cache {key}: {str(e)}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete value from all cache levels."""
        try:
            success = True
            
            if CacheLevel.L1 in self.enabled_levels:
                success &= self.l1_cache.delete(key)
            
            if CacheLevel.L2 in self.enabled_levels:
                success &= await self.l2_cache.delete(key)
            
            if CacheLevel.L3 in self.enabled_levels:
                success &= await self.l3_cache.delete(key)
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to delete cache {key}: {str(e)}")
            return False
    
    async def clear(self):
        """Clear all cache levels."""
        try:
            if CacheLevel.L1 in self.enabled_levels:
                self.l1_cache.clear()
            
            if CacheLevel.L2 in self.enabled_levels:
                await self.l2_cache.clear()
            
            if CacheLevel.L3 in self.enabled_levels:
                await self.l3_cache.clear()
            
            logger.info("All cache levels cleared")
            
        except Exception as e:
            logger.error(f"Failed to clear cache: {str(e)}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        l1_stats = self.l1_cache.get_stats()
        l2_stats = self.l2_cache.get_stats()
        l3_stats = self.l3_cache.get_stats()
        
        return {
            'l1_cache': {
                'hits': l1_stats.hits,
                'misses': l1_stats.misses,
                'hit_rate': l1_stats.hit_rate,
                'size': l1_stats.size,
                'evictions': l1_stats.evictions
            },
            'l2_cache': {
                'hits': l2_stats.hits,
                'misses': l2_stats.misses,
                'hit_rate': l2_stats.hit_rate,
                'size': l2_stats.size,
                'evictions': l2_stats.evictions
            },
            'l3_cache': {
                'hits': l3_stats.hits,
                'misses': l3_stats.misses,
                'hit_rate': l3_stats.hit_rate,
                'size': l3_stats.size,
                'evictions': l3_stats.evictions
            },
            'enabled_levels': [level.value for level in self.enabled_levels],
            'compression_enabled': self.compression_enabled
        }
    
    def enable_level(self, level: CacheLevel):
        """Enable cache level."""
        if level not in self.enabled_levels:
            self.enabled_levels.append(level)
            logger.info(f"Enabled cache level: {level.value}")
    
    def disable_level(self, level: CacheLevel):
        """Disable cache level."""
        if level in self.enabled_levels:
            self.enabled_levels.remove(level)
            logger.info(f"Disabled cache level: {level.value}")

# Global cache instance
advanced_cache = AdvancedCache()

# Decorators for caching
def cache_result(ttl: Optional[int] = None, level: Optional[CacheLevel] = None):
    """Decorator to cache function results."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Create cache key
            key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"
            
            # Try to get from cache
            cached_result = await advanced_cache.get(key)
            if cached_result is not None:
                return cached_result
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Cache result
            await advanced_cache.set(key, result, ttl, level)
            
            return result
        
        return wrapper
    return decorator

def cache_invalidate(pattern: str):
    """Decorator to invalidate cache entries matching pattern."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            result = await func(*args, **kwargs)
            
            # Invalidate cache entries matching pattern
            # This would need to be implemented based on the cache backend
            
            return result
        
        return wrapper
    return decorator











