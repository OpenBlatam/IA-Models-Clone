from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

# Constants
BUFFER_SIZE = 1024

import asyncio
import time
import logging
import json
import hashlib
from typing import Any, Dict, List, Optional, Union, Callable, TypeVar, Generic
from functools import lru_cache, wraps
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
import weakref
import threading
from concurrent.futures import ThreadPoolExecutor
import psutil
import gc
import redis.asyncio as redis
from sqlalchemy.ext.asyncio import AsyncSession
import torch
import numpy as np
        import sys
from typing import Any, List, Dict, Optional
"""
🚀 PERFORMANCE OPTIMIZATION - ASYNC, CACHING & LAZY LOADING
==========================================================

Comprehensive performance optimization system including:
- Async functions for I/O-bound tasks
- Intelligent caching strategies
- Lazy loading patterns
- Memory optimization
- Database query optimization
- Background task processing
"""



logger = logging.getLogger(__name__)

# Type variables for generic caching
T = TypeVar('T')
K = TypeVar('K')

# ============================================================================
# 1. ASYNC I/O OPTIMIZATION
# ============================================================================

class AsyncIOOptimizer:
    """Optimizer for I/O-bound operations using async patterns."""
    
    def __init__(self, max_concurrent: int = 10):
        
    """__init__ function."""
self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent)
    
    async def batch_process_async(self, items: List[Any], processor: Callable) -> List[Any]:
        """Process items concurrently with semaphore control."""
        async def process_with_semaphore(item: Any) -> Any:
            async with self.semaphore:
                return await processor(item)
        
        tasks = [process_with_semaphore(item) for item in items]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions
        return [result for result in results if not isinstance(result, Exception)]
    
    async def process_with_timeout(self, coro: Callable, timeout: float = 30.0) -> Any:
        """Process coroutine with timeout."""
        try:
            return await asyncio.wait_for(coro, timeout=timeout)
        except asyncio.TimeoutError:
            logger.error(f"Operation timed out after {timeout}s")
            raise
    
    async def retry_with_backoff(self, coro: Callable, max_retries: int = 3, 
                                base_delay: float = 1.0) -> Any:
        """Retry coroutine with exponential backoff."""
        for attempt in range(max_retries):
            try:
                return await coro()
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                
                delay = base_delay * (2 ** attempt)
                logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                await asyncio.sleep(delay)
    
    async def run_in_executor(self, func: Callable, *args, **kwargs) -> Any:
        """Run CPU-bound function in executor."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, func, *args, **kwargs)

# ============================================================================
# 2. INTELLIGENT CACHING STRATEGIES
# ============================================================================

@dataclass
class CacheConfig:
    """Configuration for caching strategies."""
    ttl: int = 3600  # Time to live in seconds
    max_size: int = 1000  # Maximum number of items
    enable_compression: bool = True
    enable_stats: bool = True
    eviction_policy: str = "lru"  # lru, lfu, ttl

class CacheStats:
    """Statistics for cache performance."""
    
    def __init__(self) -> Any:
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.size = 0
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    def reset(self) -> Any:
        """Reset statistics."""
        self.hits = 0
        self.misses = 0
        self.evictions = 0

class AsyncCache:
    """Async cache with multiple backends and strategies."""
    
    def __init__(self, redis_client: Optional[redis.Redis] = None, 
                 config: Optional[CacheConfig] = None):
        
    """__init__ function."""
self.redis_client = redis_client
        self.config = config or CacheConfig()
        self.memory_cache = {}
        self.stats = CacheStats() if self.config.enable_stats else None
        self._lock = asyncio.Lock()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        # Try memory cache first
        if key in self.memory_cache:
            if self.stats:
                self.stats.hits += 1
            return self.memory_cache[key]
        
        # Try Redis cache
        if self.redis_client:
            try:
                value = await self.redis_client.get(key)
                if value:
                    if self.stats:
                        self.stats.hits += 1
                    # Store in memory cache
                    await self._store_in_memory(key, json.loads(value))
                    return json.loads(value)
            except Exception as e:
                logger.error(f"Redis cache get failed: {e}")
        
        if self.stats:
            self.stats.misses += 1
        return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache."""
        ttl = ttl or self.config.ttl
        
        async with self._lock:
            # Store in memory cache
            await self._store_in_memory(key, value)
            
            # Store in Redis cache
            if self.redis_client:
                try:
                    await self.redis_client.setex(
                        key, 
                        ttl, 
                        json.dumps(value)
                    )
                    return True
                except Exception as e:
                    logger.error(f"Redis cache set failed: {e}")
                    return False
        
        return True
    
    async def _store_in_memory(self, key: str, value: Any):
        """Store value in memory cache with eviction."""
        if len(self.memory_cache) >= self.config.max_size:
            await self._evict_from_memory()
        
        self.memory_cache[key] = value
        if self.stats:
            self.stats.size = len(self.memory_cache)
    
    async def _evict_from_memory(self) -> Any:
        """Evict items from memory cache."""
        if self.config.eviction_policy == "lru":
            # Remove oldest item
            oldest_key = next(iter(self.memory_cache))
            del self.memory_cache[oldest_key]
        elif self.config.eviction_policy == "lfu":
            # Remove least frequently used (simplified)
            oldest_key = next(iter(self.memory_cache))
            del self.memory_cache[oldest_key]
        
        if self.stats:
            self.stats.evictions += 1
    
    async def invalidate(self, pattern: str = None):
        """Invalidate cache entries."""
        if pattern:
            # Invalidate by pattern
            keys_to_remove = [k for k in self.memory_cache.keys() if pattern in k]
            for key in keys_to_remove:
                del self.memory_cache[key]
            
            if self.redis_client:
                try:
                    # Use SCAN for pattern matching in Redis
                    async for key in self.redis_client.scan_iter(match=pattern):
                        await self.redis_client.delete(key)
                except Exception as e:
                    logger.error(f"Redis cache invalidation failed: {e}")
        else:
            # Clear all cache
            self.memory_cache.clear()
            if self.redis_client:
                try:
                    await self.redis_client.flushdb()
                except Exception as e:
                    logger.error(f"Redis cache clear failed: {e}")
    
    def get_stats(self) -> Optional[Dict[str, Any]]:
        """Get cache statistics."""
        if not self.stats:
            return None
        
        return {
            "hits": self.stats.hits,
            "misses": self.stats.misses,
            "hit_rate": self.stats.hit_rate,
            "evictions": self.stats.evictions,
            "size": self.stats.size,
            "max_size": self.config.max_size
        }

class ModelCache:
    """Specialized cache for AI models with lazy loading."""
    
    def __init__(self, cache: AsyncCache):
        
    """__init__ function."""
self.cache = cache
        self.loaded_models = {}
        self.model_loaders = {}
        self._lock = asyncio.Lock()
    
    def register_model(self, model_name: str, loader: Callable):
        """Register a model loader function."""
        self.model_loaders[model_name] = loader
    
    async def get_model(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get model with lazy loading."""
        # Check if model is already loaded
        if model_name in self.loaded_models:
            return self.loaded_models[model_name]
        
        async with self._lock:
            # Double-check after acquiring lock
            if model_name in self.loaded_models:
                return self.loaded_models[model_name]
            
            # Try to load from cache
            cached_model = await self.cache.get(f"model:{model_name}")
            if cached_model:
                self.loaded_models[model_name] = cached_model
                return cached_model
            
            # Load model using registered loader
            if model_name in self.model_loaders:
                logger.info(f"Loading model: {model_name}")
                model = await self.model_loaders[model_name]()
                
                # Store in memory and cache
                self.loaded_models[model_name] = model
                await self.cache.set(f"model:{model_name}", model)
                
                return model
            
            raise ValueError(f"Model {model_name} not found and no loader registered")
    
    async def preload_models(self, model_names: List[str]):
        """Preload multiple models concurrently."""
        tasks = [self.get_model(name) for name in model_names]
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def unload_model(self, model_name: str):
        """Unload model from memory."""
        if model_name in self.loaded_models:
            del self.loaded_models[model_name]
            await self.cache.invalidate(f"model:{model_name}")

# ============================================================================
# 3. LAZY LOADING PATTERNS
# ============================================================================

class LazyLoader:
    """Generic lazy loader for expensive resources."""
    
    def __init__(self, loader_func: Callable, cache_key: Optional[str] = None):
        
    """__init__ function."""
self.loader_func = loader_func
        self.cache_key = cache_key
        self._value = None
        self._loaded = False
        self._lock = asyncio.Lock()
    
    async def get(self) -> Optional[Dict[str, Any]]:
        """Get value, loading if necessary."""
        if self._loaded:
            return self._value
        
        async with self._lock:
            if self._loaded:
                return self._value
            
            self._value = await self.loader_func()
            self._loaded = True
            return self._value
    
    def reset(self) -> Any:
        """Reset lazy loader."""
        self._value = None
        self._loaded = False

class LazyDict:
    """Dictionary with lazy loading of values."""
    
    def __init__(self) -> Any:
        self._data = {}
        self._loaders = {}
        self._loaded = set()
        self._lock = asyncio.Lock()
    
    def register_loader(self, key: str, loader: Callable):
        """Register a loader function for a key."""
        self._loaders[key] = loader
    
    async def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get value, loading if necessary."""
        if key in self._data:
            return self._data[key]
        
        if key in self._loaders:
            async with self._lock:
                if key in self._data:
                    return self._data[key]
                
                if key not in self._loaded:
                    self._data[key] = await self._loaders[key]()
                    self._loaded.add(key)
                
                return self._data[key]
        
        raise KeyError(f"Key {key} not found and no loader registered")
    
    async def preload(self, keys: List[str]):
        """Preload multiple keys."""
        tasks = [self.get(key) for key in keys if key in self._loaders]
        await asyncio.gather(*tasks, return_exceptions=True)
    
    def clear(self) -> Any:
        """Clear all loaded data."""
        self._data.clear()
        self._loaded.clear()

# ============================================================================
# 4. DATABASE QUERY OPTIMIZATION
# ============================================================================

class QueryOptimizer:
    """Optimizer for database queries."""
    
    def __init__(self, cache: AsyncCache):
        
    """__init__ function."""
self.cache = cache
        self.query_stats = {}
    
    async def cached_query(self, query_key: str, query_func: Callable, 
                          ttl: int = 300) -> Any:
        """Execute query with caching."""
        # Try cache first
        cached_result = await self.cache.get(f"query:{query_key}")
        if cached_result:
            return cached_result
        
        # Execute query
        result = await query_func()
        
        # Cache result
        await self.cache.set(f"query:{query_key}", result, ttl)
        
        return result
    
    async def batch_query(self, queries: List[Dict[str, Any]]) -> List[Any]:
        """Execute multiple queries in batch."""
        # Group queries by type for optimization
        grouped_queries = self._group_queries(queries)
        
        results = []
        for query_type, query_list in grouped_queries.items():
            batch_results = await self._execute_batch(query_type, query_list)
            results.extend(batch_results)
        
        return results
    
    def _group_queries(self, queries: List[Dict[str, Any]]) -> Dict[str, List]:
        """Group queries by type for batch processing."""
        grouped = {}
        for query in queries:
            query_type = query.get('type', 'default')
            if query_type not in grouped:
                grouped[query_type] = []
            grouped[query_type].append(query)
        return grouped
    
    async def _execute_batch(self, query_type: str, queries: List[Dict]) -> List[Any]:
        """Execute batch of queries of the same type."""
        # This is a simplified implementation
        # In practice, you would optimize based on query type
        results = []
        for query in queries:
            result = await self.cached_query(
                query['key'],
                query['func'],
                query.get('ttl', 300)
            )
            results.append(result)
        return results

# ============================================================================
# 5. MEMORY OPTIMIZATION
# ============================================================================

class MemoryOptimizer:
    """Memory optimization utilities."""
    
    def __init__(self) -> Any:
        self.memory_threshold = 0.8  # 80% of available memory
        self.gc_threshold = 1000  # Number of objects before GC
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage."""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            "rss_mb": memory_info.rss / 1024 / 1024,
            "vms_mb": memory_info.vms / 1024 / 1024,
            "percent": process.memory_percent(),
            "available_mb": psutil.virtual_memory().available / 1024 / 1024
        }
    
    def should_optimize_memory(self) -> bool:
        """Check if memory optimization is needed."""
        memory_usage = self.get_memory_usage()
        return memory_usage["percent"] > (self.memory_threshold * 100)
    
    def optimize_memory(self) -> Any:
        """Perform memory optimization."""
        logger.info("Performing memory optimization...")
        
        # Force garbage collection
        collected = gc.collect()
        logger.info(f"Garbage collected {collected} objects")
        
        # Clear Python cache
        for module in list(sys.modules.keys()):
            if hasattr(sys.modules[module], '__file__'):
                try:
                    del sys.modules[module]
                except:
                    pass
        
        logger.info("Memory optimization completed")
    
    def monitor_memory(self, callback: Optional[Callable] = None):
        """Monitor memory usage and trigger optimization if needed."""
        if self.should_optimize_memory():
            self.optimize_memory()
            if callback:
                callback()

class WeakRefCache:
    """Cache using weak references to avoid memory leaks."""
    
    def __init__(self) -> Any:
        self._cache = weakref.WeakValueDictionary()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from weak reference cache."""
        return self._cache.get(key)
    
    def set(self, key: str, value: Any):
        """Set value in weak reference cache."""
        self._cache[key] = value
    
    def clear(self) -> Any:
        """Clear cache."""
        self._cache.clear()

# ============================================================================
# 6. BACKGROUND TASK PROCESSING
# ============================================================================

class BackgroundTaskProcessor:
    """Processor for background tasks with optimization."""
    
    def __init__(self, max_workers: int = 4):
        
    """__init__ function."""
self.max_workers = max_workers
        self.task_queue = asyncio.Queue()
        self.workers = []
        self.running = False
    
    async def start(self) -> Any:
        """Start background task processor."""
        self.running = True
        self.workers = [
            asyncio.create_task(self._worker(f"worker-{i}"))
            for i in range(self.max_workers)
        ]
        logger.info(f"Started {self.max_workers} background workers")
    
    async def stop(self) -> Any:
        """Stop background task processor."""
        self.running = False
        
        # Wait for all workers to complete
        if self.workers:
            await asyncio.gather(*self.workers, return_exceptions=True)
        
        logger.info("Background task processor stopped")
    
    async def add_task(self, task_func: Callable, *args, **kwargs):
        """Add task to processing queue."""
        await self.task_queue.put((task_func, args, kwargs))
    
    async def _worker(self, worker_name: str):
        """Background worker function."""
        logger.info(f"Worker {worker_name} started")
        
        while self.running:
            try:
                # Get task from queue with timeout
                task_func, args, kwargs = await asyncio.wait_for(
                    self.task_queue.get(), timeout=1.0
                )
                
                # Execute task
                try:
                    result = await task_func(*args, **kwargs)
                    logger.info(f"Worker {worker_name} completed task successfully")
                except Exception as e:
                    logger.error(f"Worker {worker_name} task failed: {e}")
                
                # Mark task as done
                self.task_queue.task_done()
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Worker {worker_name} error: {e}")
        
        logger.info(f"Worker {worker_name} stopped")

# ============================================================================
# 7. PERFORMANCE MONITORING
# ============================================================================

@dataclass
class PerformanceMetrics:
    """Performance metrics container."""
    operation: str
    duration: float
    memory_delta: float
    cache_hits: int = 0
    cache_misses: int = 0
    timestamp: float = field(default_factory=time.time)

class PerformanceMonitor:
    """Monitor and track performance metrics."""
    
    def __init__(self) -> Any:
        self.metrics = []
        self.memory_optimizer = MemoryOptimizer()
    
    async def track_operation(self, operation: str, coro: Callable) -> Any:
        """Track performance of an operation."""
        start_time = time.time()
        start_memory = self.memory_optimizer.get_memory_usage()["rss_mb"]
        
        try:
            result = await coro()
            
            duration = time.time() - start_time
            end_memory = self.memory_optimizer.get_memory_usage()["rss_mb"]
            memory_delta = end_memory - start_memory
            
            # Record metrics
            metric = PerformanceMetrics(
                operation=operation,
                duration=duration,
                memory_delta=memory_delta
            )
            self.metrics.append(metric)
            
            # Log slow operations
            if duration > 1.0:
                logger.warning(f"Slow operation: {operation} took {duration:.3f}s")
            
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Operation {operation} failed after {duration:.3f}s: {e}")
            raise
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate performance report."""
        if not self.metrics:
            return {"message": "No metrics available"}
        
        # Calculate statistics
        durations = [m.duration for m in self.metrics]
        memory_deltas = [m.memory_delta for m in self.metrics]
        
        return {
            "total_operations": len(self.metrics),
            "avg_duration": sum(durations) / len(durations),
            "max_duration": max(durations),
            "min_duration": min(durations),
            "avg_memory_delta": sum(memory_deltas) / len(memory_deltas),
            "memory_usage": self.memory_optimizer.get_memory_usage(),
            "recent_operations": [
                {
                    "operation": m.operation,
                    "duration": m.duration,
                    "memory_delta": m.memory_delta,
                    "timestamp": m.timestamp
                }
                for m in self.metrics[-10:]  # Last 10 operations
            ]
        }

# ============================================================================
# 8. USAGE EXAMPLES
# ============================================================================

async def example_async_io_optimization():
    """Example of async I/O optimization."""
    
    optimizer = AsyncIOOptimizer(max_concurrent=5)
    
    # Simulate I/O-bound operations
    async def fetch_data(item_id: int):
        
    """fetch_data function."""
await asyncio.sleep(0.1)  # Simulate I/O
        return f"data_{item_id}"
    
    # Process items concurrently
    items = list(range(100))
    results = await optimizer.batch_process_async(items, fetch_data)
    
    logger.info(f"Processed {len(results)} items concurrently")
    return results

async def example_caching_strategy():
    """Example of intelligent caching."""
    
    # Create cache
    cache = AsyncCache()
    
    # Simulate expensive operation
    async def expensive_operation(key: str):
        
    """expensive_operation function."""
await asyncio.sleep(1)  # Simulate expensive operation
        return f"result_for_{key}"
    
    # Use cache
    result1 = await cache.get("test_key")
    if result1 is None:
        result1 = await expensive_operation("test_key")
        await cache.set("test_key", result1)
    
    # Second call should be cached
    result2 = await cache.get("test_key")
    
    logger.info(f"Cache stats: {cache.get_stats()}")
    return result1, result2

async def example_lazy_loading():
    """Example of lazy loading patterns."""
    
    # Create lazy loader
    async def load_expensive_resource():
        
    """load_expensive_resource function."""
await asyncio.sleep(2)  # Simulate loading
        return {"data": "expensive_resource"}
    
    lazy_loader = LazyLoader(load_expensive_resource)
    
    # First call loads the resource
    resource1 = await lazy_loader.get()
    
    # Second call uses cached resource
    resource2 = await lazy_loader.get()
    
    logger.info("Lazy loading completed")
    return resource1, resource2

async def example_background_processing():
    """Example of background task processing."""
    
    processor = BackgroundTaskProcessor(max_workers=3)
    await processor.start()
    
    # Add tasks
    async def background_task(task_id: int):
        
    """background_task function."""
await asyncio.sleep(1)
        logger.info(f"Background task {task_id} completed")
    
    for i in range(10):
        await processor.add_task(background_task, i)
    
    # Wait for tasks to complete
    await processor.task_queue.join()
    
    await processor.stop()
    logger.info("Background processing completed")

def example_memory_optimization():
    """Example of memory optimization."""
    
    optimizer = MemoryOptimizer()
    
    # Check memory usage
    usage = optimizer.get_memory_usage()
    logger.info(f"Memory usage: {usage}")
    
    # Optimize if needed
    if optimizer.should_optimize_memory():
        optimizer.optimize_memory()
    
    return usage

# ============================================================================
# 9. INTEGRATED PERFORMANCE SYSTEM
# ============================================================================

class PerformanceOptimizationSystem:
    """Integrated performance optimization system."""
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        
    """__init__ function."""
self.io_optimizer = AsyncIOOptimizer()
        self.cache = AsyncCache(redis_client)
        self.model_cache = ModelCache(self.cache)
        self.query_optimizer = QueryOptimizer(self.cache)
        self.memory_optimizer = MemoryOptimizer()
        self.background_processor = BackgroundTaskProcessor()
        self.performance_monitor = PerformanceMonitor()
        
        # Start background processor
        asyncio.create_task(self.background_processor.start())
    
    async def optimize_operation(self, operation_name: str, coro: Callable) -> Any:
        """Optimize an operation with all available optimizations."""
        return await self.performance_monitor.track_operation(
            operation_name,
            coro
        )
    
    async def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        return {
            "memory": self.memory_optimizer.get_memory_usage(),
            "cache": self.cache.get_stats(),
            "performance": self.performance_monitor.get_performance_report(),
            "background_tasks": self.background_processor.task_queue.qsize()
        }
    
    async def cleanup(self) -> Any:
        """Cleanup system resources."""
        await self.background_processor.stop()
        await self.cache.invalidate()

if __name__ == "__main__":
    # Example usage
    async def main():
        
    """main function."""
# Create optimization system
        system = PerformanceOptimizationSystem()
        
        # Run examples
        await example_async_io_optimization()
        await example_caching_strategy()
        await example_lazy_loading()
        await example_background_processing()
        example_memory_optimization()
        
        # Get system stats
        stats = await system.get_system_stats()
        logger.info(f"System stats: {json.dumps(stats, indent=2)}")
        
        # Cleanup
        await system.cleanup()
    
    asyncio.run(main()) 