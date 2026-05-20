"""
Performance Optimizer - Aggressive performance optimizations
"""

import asyncio
from typing import Dict, Any, Optional, Callable
from functools import lru_cache, wraps
import time
import logging
from collections import OrderedDict
import threading
import hashlib
import pickle

logger = logging.getLogger(__name__)


class LRUCache:
    """High-performance LRU cache with TTL"""
    
    def __init__(self, maxsize: int = 128, ttl: float = 3600.0):
        self.maxsize = maxsize
        self.ttl = ttl
        self.cache: OrderedDict = OrderedDict()
        self.timestamps: Dict[str, float] = {}
        self.lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache"""
        with self.lock:
            if key not in self.cache:
                return None
            
            # Check TTL
            if time.time() - self.timestamps[key] > self.ttl:
                del self.cache[key]
                del self.timestamps[key]
                return None
            
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            return self.cache[key]
    
    def set(self, key: str, value: Any):
        """Set item in cache"""
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
            else:
                if len(self.cache) >= self.maxsize:
                    # Remove oldest
                    oldest_key = next(iter(self.cache))
                    del self.cache[oldest_key]
                    del self.timestamps[oldest_key]
            
            self.cache[key] = value
            self.timestamps[key] = time.time()
    
    def clear(self):
        """Clear cache"""
        with self.lock:
            self.cache.clear()
            self.timestamps.clear()


class PerformanceOptimizer:
    """
    Aggressive performance optimizations:
    - LRU cache with TTL
    - Function memoization
    - Async prefetching
    - Connection pooling
    - Lazy loading
    """
    
    def __init__(self):
        self.caches: Dict[str, LRUCache] = {}
        self.prefetch_queue: asyncio.Queue = asyncio.Queue()
        self.prefetch_workers: list = []
    
    def get_cache(self, name: str, maxsize: int = 128, ttl: float = 3600.0) -> LRUCache:
        """Get or create a named cache"""
        if name not in self.caches:
            self.caches[name] = LRUCache(maxsize=maxsize, ttl=ttl)
        return self.caches[name]
    
    def cached(
        self,
        cache_name: str = "default",
        maxsize: int = 128,
        ttl: float = 3600.0,
        key_func: Optional[Callable] = None
    ):
        """Decorator for caching function results"""
        cache = self.get_cache(cache_name, maxsize, ttl)
        
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Generate cache key
                if key_func:
                    cache_key = key_func(*args, **kwargs)
                else:
                    try:
                        cache_key = hashlib.md5(
                            pickle.dumps((args, tuple(sorted(kwargs.items()))))
                        ).hexdigest()
                    except Exception:
                        cache_key = str((args, kwargs))
                
                # Check cache
                result = cache.get(cache_key)
                if result is not None:
                    return result
                
                # Compute result
                result = func(*args, **kwargs)
                cache.set(cache_key, result)
                return result
            
            return wrapper
        return decorator
    
    async def prefetch(self, func: Callable, *args, **kwargs):
        """Prefetch a function call"""
        await self.prefetch_queue.put((func, args, kwargs))
    
    async def start_prefetch_workers(self, num_workers: int = 2):
        """Start prefetch workers"""
        for i in range(num_workers):
            worker = asyncio.create_task(self._prefetch_worker(f"prefetch-{i}"))
            self.prefetch_workers.append(worker)
    
    async def _prefetch_worker(self, worker_id: str):
        """Prefetch worker coroutine"""
        while True:
            try:
                func, args, kwargs = await asyncio.wait_for(
                    self.prefetch_queue.get(),
                    timeout=1.0
                )
                
                # Execute function
                if asyncio.iscoroutinefunction(func):
                    await func(*args, **kwargs)
                else:
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(None, func, *args, **kwargs)
                
                self.prefetch_queue.task_done()
            
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Prefetch worker {worker_id} error: {str(e)}")


class LazyLoader:
    """Lazy loading for expensive resources"""
    
    def __init__(self, loader_func: Callable, *args, **kwargs):
        self.loader_func = loader_func
        self.args = args
        self.kwargs = kwargs
        self._resource: Optional[Any] = None
        self._lock = threading.Lock()
    
    @property
    def resource(self) -> Any:
        """Get resource, loading if necessary"""
        if self._resource is None:
            with self._lock:
                if self._resource is None:
                    self._resource = self.loader_func(*self.args, **self.kwargs)
        return self._resource
    
    def reload(self):
        """Force reload of resource"""
        with self._lock:
            self._resource = self.loader_func(*self.args, **self.kwargs)


def async_timing(func: Callable) -> Callable:
    """Async timing decorator"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start = time.time()
        result = await func(*args, **kwargs)
        elapsed = time.time() - start
        logger.debug(f"{func.__name__} took {elapsed:.4f}s")
        return result
    return wrapper


def batch_process(items: list, batch_size: int = 32, func: Optional[Callable] = None):
    """Process items in batches"""
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        if func:
            yield func(batch)
        else:
            yield batch


