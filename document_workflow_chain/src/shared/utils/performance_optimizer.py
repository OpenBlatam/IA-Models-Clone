"""
Performance Optimizer
=====================

Performance optimization utilities following FastAPI best practices.
"""

from __future__ import annotations
import asyncio
import functools
import time
import logging
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
import weakref
import gc

from .helpers import DateTimeHelpers
from .decorators import log_execution

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class PerformanceMetrics:
    """Performance metrics container"""
    function_name: str
    execution_time: float
    memory_usage: float
    cache_hits: int = 0
    cache_misses: int = 0
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = DateTimeHelpers.now_utc()


class LRUCache:
    """Thread-safe LRU cache implementation"""
    
    def __init__(self, max_size: int = 1000, ttl: Optional[float] = None):
        self.max_size = max_size
        self.ttl = ttl
        self._cache: Dict[str, tuple[Any, datetime]] = {}
        self._access_order: List[str] = []
        self._lock = asyncio.Lock()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        async with self._lock:
            if key not in self._cache:
                return None
            
            value, timestamp = self._cache[key]
            
            # Check TTL
            if self.ttl and DateTimeHelpers.now_utc() - timestamp > timedelta(seconds=self.ttl):
                del self._cache[key]
                if key in self._access_order:
                    self._access_order.remove(key)
                return None
            
            # Update access order
            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)
            
            return value
    
    async def set(self, key: str, value: Any) -> None:
        """Set value in cache"""
        async with self._lock:
            # Remove oldest if at capacity
            if len(self._cache) >= self.max_size and key not in self._cache:
                oldest_key = self._access_order.pop(0)
                del self._cache[oldest_key]
            
            self._cache[key] = (value, DateTimeHelpers.now_utc())
            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)
    
    async def delete(self, key: str) -> bool:
        """Delete key from cache"""
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                if key in self._access_order:
                    self._access_order.remove(key)
                return True
            return False
    
    async def clear(self) -> None:
        """Clear all cache entries"""
        async with self._lock:
            self._cache.clear()
            self._access_order.clear()
    
    async def size(self) -> int:
        """Get cache size"""
        async with self._lock:
            return len(self._cache)


class ConnectionPool:
    """Async connection pool for database connections"""
    
    def __init__(self, factory: Callable, min_connections: int = 5, max_connections: int = 20):
        self.factory = factory
        self.min_connections = min_connections
        self.max_connections = max_connections
        self._pool: List[Any] = []
        self._in_use: set = set()
        self._lock = asyncio.Lock()
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize connection pool"""
        if self._initialized:
            return
        
        async with self._lock:
            for _ in range(self.min_connections):
                connection = await self.factory()
                self._pool.append(connection)
            self._initialized = True
    
    async def acquire(self) -> Any:
        """Acquire connection from pool"""
        await self.initialize()
        
        async with self._lock:
            if self._pool:
                connection = self._pool.pop()
                self._in_use.add(connection)
                return connection
            
            if len(self._in_use) < self.max_connections:
                connection = await self.factory()
                self._in_use.add(connection)
                return connection
            
            # Wait for connection to be available
            while not self._pool:
                await asyncio.sleep(0.01)
            
            connection = self._pool.pop()
            self._in_use.add(connection)
            return connection
    
    async def release(self, connection: Any) -> None:
        """Release connection back to pool"""
        async with self._lock:
            if connection in self._in_use:
                self._in_use.remove(connection)
                self._pool.append(connection)
    
    async def close(self) -> None:
        """Close all connections"""
        async with self._lock:
            for connection in self._pool + list(self._in_use):
                if hasattr(connection, 'close'):
                    await connection.close()
            self._pool.clear()
            self._in_use.clear()


class BatchProcessor:
    """Batch processor for handling multiple operations efficiently"""
    
    def __init__(self, batch_size: int = 100, flush_interval: float = 1.0):
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self._batch: List[Any] = []
        self._lock = asyncio.Lock()
        self._last_flush = time.time()
        self._processor_task: Optional[asyncio.Task] = None
        self._is_running = False
    
    async def start(self) -> None:
        """Start batch processor"""
        if self._is_running:
            return
        
        self._is_running = True
        self._processor_task = asyncio.create_task(self._process_batches())
    
    async def stop(self) -> None:
        """Stop batch processor"""
        if not self._is_running:
            return
        
        self._is_running = False
        
        if self._processor_task:
            self._processor_task.cancel()
            try:
                await self._processor_task
            except asyncio.CancelledError:
                pass
        
        # Process remaining items
        await self._flush_batch()
    
    async def add_item(self, item: Any) -> None:
        """Add item to batch"""
        async with self._lock:
            self._batch.append(item)
            
            if len(self._batch) >= self.batch_size:
                await self._flush_batch()
    
    async def _process_batches(self) -> None:
        """Process batches periodically"""
        while self._is_running:
            try:
                current_time = time.time()
                if current_time - self._last_flush >= self.flush_interval:
                    await self._flush_batch()
                await asyncio.sleep(0.1)
            except Exception as e:
                logger.error(f"Batch processor error: {e}")
                await asyncio.sleep(1)
    
    async def _flush_batch(self) -> None:
        """Flush current batch"""
        async with self._lock:
            if not self._batch:
                return
            
            batch = self._batch.copy()
            self._batch.clear()
            self._last_flush = time.time()
        
        # Process batch (implement in subclass)
        await self._process_batch(batch)
    
    async def _process_batch(self, batch: List[Any]) -> None:
        """Process batch of items (to be implemented by subclass)"""
        pass


class MemoryOptimizer:
    """Memory optimization utilities"""
    
    @staticmethod
    def get_memory_usage() -> float:
        """Get current memory usage in MB"""
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    @staticmethod
    def force_garbage_collection() -> None:
        """Force garbage collection"""
        gc.collect()
    
    @staticmethod
    def optimize_memory() -> Dict[str, Any]:
        """Optimize memory usage"""
        before_memory = MemoryOptimizer.get_memory_usage()
        
        # Force garbage collection
        MemoryOptimizer.force_garbage_collection()
        
        after_memory = MemoryOptimizer.get_memory_usage()
        freed_memory = before_memory - after_memory
        
        return {
            "before_memory_mb": before_memory,
            "after_memory_mb": after_memory,
            "freed_memory_mb": freed_memory,
            "timestamp": DateTimeHelpers.now_utc().isoformat()
        }


class AsyncRateLimiter:
    """Async rate limiter for API endpoints"""
    
    def __init__(self, max_requests: int, time_window: float):
        self.max_requests = max_requests
        self.time_window = time_window
        self._requests: List[float] = []
        self._lock = asyncio.Lock()
    
    async def is_allowed(self) -> bool:
        """Check if request is allowed"""
        async with self._lock:
            now = time.time()
            
            # Remove old requests
            self._requests = [req_time for req_time in self._requests if now - req_time < self.time_window]
            
            if len(self._requests) < self.max_requests:
                self._requests.append(now)
                return True
            
            return False
    
    async def get_retry_after(self) -> float:
        """Get retry after time in seconds"""
        async with self._lock:
            if not self._requests:
                return 0.0
            
            oldest_request = min(self._requests)
            return max(0.0, self.time_window - (time.time() - oldest_request))


# Global instances
_lru_cache = LRUCache(max_size=1000, ttl=300)  # 5 minutes TTL
_memory_optimizer = MemoryOptimizer()


# Performance decorators
def cache_result(ttl: Optional[float] = None, key_func: Optional[Callable] = None):
    """Cache function result"""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"
            
            # Try to get from cache
            cached_result = await _lru_cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            # Cache result
            await _lru_cache.set(cache_key, result)
            
            return result
        
        return wrapper
    return decorator


def measure_performance(func: Callable[..., T]) -> Callable[..., T]:
    """Measure function performance"""
    @functools.wraps(func)
    async def wrapper(*args, **kwargs) -> T:
        start_time = time.time()
        start_memory = _memory_optimizer.get_memory_usage()
        
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            execution_time = time.time() - start_time
            end_memory = _memory_optimizer.get_memory_usage()
            memory_usage = end_memory - start_memory
            
            metrics = PerformanceMetrics(
                function_name=func.__name__,
                execution_time=execution_time,
                memory_usage=memory_usage
            )
            
            logger.info(f"Performance metrics: {metrics}")
            
            return result
        
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Function {func.__name__} failed after {execution_time:.2f}s: {e}")
            raise
    
    return wrapper


def optimize_memory_usage(func: Callable[..., T]) -> Callable[..., T]:
    """Optimize memory usage for function"""
    @functools.wraps(func)
    async def wrapper(*args, **kwargs) -> T:
        # Optimize memory before execution
        _memory_optimizer.optimize_memory()
        
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            # Optimize memory after execution
            _memory_optimizer.optimize_memory()
            
            return result
        
        except Exception as e:
            # Optimize memory on error
            _memory_optimizer.optimize_memory()
            raise
    
    return wrapper


# Utility functions
async def get_cache_stats() -> Dict[str, Any]:
    """Get cache statistics"""
    return {
        "cache_size": await _lru_cache.size(),
        "memory_usage_mb": _memory_optimizer.get_memory_usage(),
        "timestamp": DateTimeHelpers.now_utc().isoformat()
    }


async def clear_cache() -> None:
    """Clear all cache entries"""
    await _lru_cache.clear()


async def optimize_system() -> Dict[str, Any]:
    """Optimize system performance"""
    memory_stats = _memory_optimizer.optimize_memory()
    cache_stats = await get_cache_stats()
    
    return {
        "memory_optimization": memory_stats,
        "cache_stats": cache_stats,
        "timestamp": DateTimeHelpers.now_utc().isoformat()
    }


# FastAPI-specific utilities
def create_async_dependency(func: Callable[..., T]) -> Callable[..., T]:
    """Create async dependency for FastAPI"""
    @functools.wraps(func)
    async def dependency(*args, **kwargs) -> T:
        if asyncio.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        else:
            return func(*args, **kwargs)
    
    return dependency


def create_cached_dependency(func: Callable[..., T], ttl: float = 300) -> Callable[..., T]:
    """Create cached dependency for FastAPI"""
    cached_func = cache_result(ttl=ttl)(func)
    return create_async_dependency(cached_func)


# Performance monitoring
class PerformanceMonitor:
    """Performance monitoring for FastAPI applications"""
    
    def __init__(self):
        self.metrics: List[PerformanceMetrics] = []
        self._lock = asyncio.Lock()
    
    async def record_metric(self, metric: PerformanceMetrics) -> None:
        """Record performance metric"""
        async with self._lock:
            self.metrics.append(metric)
            
            # Keep only last 1000 metrics
            if len(self.metrics) > 1000:
                self.metrics = self.metrics[-500:]
    
    async def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        async with self._lock:
            if not self.metrics:
                return {"message": "No metrics available"}
            
            recent_metrics = self.metrics[-100:]  # Last 100 metrics
            
            execution_times = [m.execution_time for m in recent_metrics]
            memory_usages = [m.memory_usage for m in recent_metrics]
            
            return {
                "total_metrics": len(self.metrics),
                "recent_metrics": len(recent_metrics),
                "average_execution_time": sum(execution_times) / len(execution_times),
                "max_execution_time": max(execution_times),
                "min_execution_time": min(execution_times),
                "average_memory_usage": sum(memory_usages) / len(memory_usages),
                "max_memory_usage": max(memory_usages),
                "timestamp": DateTimeHelpers.now_utc().isoformat()
            }


# Global performance monitor
performance_monitor = PerformanceMonitor()




