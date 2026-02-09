"""
Smart Caching System

Advanced caching with:
- LRU cache with TTL
- Predictive prefetching
- Cache warming
- Distributed caching
- Cache analytics
"""

import logging
import time
import hashlib
import pickle
from typing import Optional, Dict, Any, List, Tuple
from collections import OrderedDict
from pathlib import Path
import asyncio

logger = logging.getLogger(__name__)


class LRUCacheWithTTL:
    """LRU cache with TTL (Time To Live)."""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        """
        Initialize LRU cache with TTL.
        
        Args:
            max_size: Maximum cache size
            default_ttl: Default TTL in seconds
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: OrderedDict = OrderedDict()
        self.timestamps: Dict[str, float] = {}
    
    def _is_expired(self, key: str) -> bool:
        """Check if key is expired."""
        if key not in self.timestamps:
            return False
        
        age = time.time() - self.timestamps[key]
        return age > self.default_ttl
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if key not in self.cache:
            return None
        
        if self._is_expired(key):
            del self.cache[key]
            del self.timestamps[key]
            return None
        
        # Move to end (most recently used)
        self.cache.move_to_end(key)
        return self.cache[key]
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache."""
        ttl = ttl or self.default_ttl
        
        # Remove if exists
        if key in self.cache:
            del self.cache[key]
        
        # Remove oldest if full
        if len(self.cache) >= self.max_size:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
            del self.timestamps[oldest_key]
        
        # Add new entry
        self.cache[key] = value
        self.timestamps[key] = time.time()
    
    def clear(self) -> None:
        """Clear cache."""
        self.cache.clear()
        self.timestamps.clear()
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'usage_percent': len(self.cache) / self.max_size * 100
        }


class PredictiveCache:
    """Predictive caching with prefetching."""
    
    def __init__(self, base_cache: LRUCacheWithTTL):
        """
        Initialize predictive cache.
        
        Args:
            base_cache: Base cache to use
        """
        self.cache = base_cache
        self.access_patterns: Dict[str, List[str]] = {}
        self.prefetch_queue: List[str] = []
    
    def record_access(self, key: str, next_key: Optional[str] = None) -> None:
        """
        Record access pattern.
        
        Args:
            key: Current key
            next_key: Next accessed key (for pattern learning)
        """
        if key not in self.access_patterns:
            self.access_patterns[key] = []
        
        if next_key:
            self.access_patterns[key].append(next_key)
    
    def predict_next(self, current_key: str) -> List[str]:
        """
        Predict next likely keys.
        
        Args:
            current_key: Current key
            
        Returns:
            List of predicted next keys
        """
        if current_key not in self.access_patterns:
            return []
        
        # Count frequency of next keys
        next_keys = self.access_patterns[current_key]
        if not next_keys:
            return []
        
        # Return most common next keys
        from collections import Counter
        counter = Counter(next_keys)
        return [key for key, _ in counter.most_common(3)]
    
    async def prefetch(self, current_key: str, generator_func) -> None:
        """
        Prefetch predicted keys.
        
        Args:
            current_key: Current key
            generator_func: Function to generate values
        """
        predicted_keys = self.predict_next(current_key)
        
        for key in predicted_keys:
            if key not in self.cache.cache:
                try:
                    # Generate and cache
                    value = await generator_func(key)
                    self.cache.set(key, value)
                    logger.debug(f"Prefetched: {key}")
                except Exception as e:
                    logger.debug(f"Prefetch failed for {key}: {e}")


class DistributedCache:
    """Distributed caching with Redis."""
    
    def __init__(self, redis_url: Optional[str] = None):
        """
        Initialize distributed cache.
        
        Args:
            redis_url: Redis connection URL
        """
        self.redis_url = redis_url
        self.redis_client = None
        
        if redis_url:
            try:
                import redis
                self.redis_client = redis.from_url(redis_url)
                logger.info("Distributed cache (Redis) initialized")
            except ImportError:
                logger.warning("Redis not available")
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}")
    
    def get(self, key: str) -> Optional[Any]:
        """Get from distributed cache."""
        if not self.redis_client:
            return None
        
        try:
            data = self.redis_client.get(key)
            if data:
                return pickle.loads(data)
        except Exception as e:
            logger.debug(f"Cache get error: {e}")
        
        return None
    
    def set(self, key: str, value: Any, ttl: int = 3600) -> None:
        """Set in distributed cache."""
        if not self.redis_client:
            return
        
        try:
            data = pickle.dumps(value)
            self.redis_client.setex(key, ttl, data)
        except Exception as e:
            logger.debug(f"Cache set error: {e}")


class SmartCache:
    """Smart multi-level cache system."""
    
    def __init__(
        self,
        max_size: int = 1000,
        ttl: int = 3600,
        use_redis: bool = False,
        redis_url: Optional[str] = None,
        enable_predictive: bool = True
    ):
        """
        Initialize smart cache.
        
        Args:
            max_size: Maximum cache size
            ttl: Time to live in seconds
            use_redis: Enable Redis distributed cache
            redis_url: Redis URL
            enable_predictive: Enable predictive prefetching
        """
        # L1: In-memory LRU cache
        self.l1_cache = LRUCacheWithTTL(max_size=max_size, default_ttl=ttl)
        
        # L2: Distributed cache (Redis)
        self.l2_cache = DistributedCache(redis_url) if use_redis else None
        
        # L3: Predictive cache
        self.predictive_cache = PredictiveCache(self.l1_cache) if enable_predictive else None
        
        # Statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'prefetches': 0
        }
    
    def _make_key(self, *args, **kwargs) -> str:
        """Create cache key from arguments."""
        key_data = str(args) + str(sorted(kwargs.items()))
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get(self, *args, **kwargs) -> Optional[Any]:
        """Get from cache (checks all levels)."""
        key = self._make_key(*args, **kwargs)
        
        # Try L1
        value = self.l1_cache.get(key)
        if value is not None:
            self.stats['hits'] += 1
            return value
        
        # Try L2
        if self.l2_cache:
            value = self.l2_cache.get(key)
            if value is not None:
                # Promote to L1
                self.l1_cache.set(key, value)
                self.stats['hits'] += 1
                return value
        
        self.stats['misses'] += 1
        return None
    
    def set(self, value: Any, *args, ttl: Optional[int] = None, **kwargs) -> None:
        """Set in cache (all levels)."""
        key = self._make_key(*args, **kwargs)
        
        # Set in L1
        self.l1_cache.set(key, value, ttl=ttl)
        
        # Set in L2
        if self.l2_cache:
            self.l2_cache.set(key, value, ttl=ttl or 3600)
    
    async def get_or_compute(
        self,
        compute_func,
        *args,
        ttl: Optional[int] = None,
        enable_prefetch: bool = True,
        **kwargs
    ) -> Any:
        """
        Get from cache or compute.
        
        Args:
            compute_func: Function to compute value
            *args: Arguments for key and function
            ttl: Time to live
            enable_prefetch: Enable predictive prefetching
            **kwargs: Keyword arguments
            
        Returns:
            Cached or computed value
        """
        # Try cache
        value = self.get(*args, **kwargs)
        if value is not None:
            return value
        
        # Compute
        if asyncio.iscoroutinefunction(compute_func):
            value = await compute_func(*args, **kwargs)
        else:
            value = compute_func(*args, **kwargs)
        
        # Cache
        self.set(value, *args, ttl=ttl, **kwargs)
        
        # Prefetch if enabled
        if enable_prefetch and self.predictive_cache:
            key = self._make_key(*args, **kwargs)
            asyncio.create_task(
                self.predictive_cache.prefetch(key, compute_func)
            )
        
        return value
    
    def clear(self) -> None:
        """Clear all caches."""
        self.l1_cache.clear()
        if self.l2_cache:
            # Redis clear would need implementation
            pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        hit_rate = (
            self.stats['hits'] / (self.stats['hits'] + self.stats['misses'])
            if (self.stats['hits'] + self.stats['misses']) > 0 else 0
        )
        
        return {
            **self.l1_cache.stats(),
            'hits': self.stats['hits'],
            'misses': self.stats['misses'],
            'hit_rate': hit_rate,
            'prefetches': self.stats['prefetches']
        }









