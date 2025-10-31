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
import gc
import time
import psutil
import threading
from typing import Dict, Any, Optional, List, Callable, Union, TypeVar, Generic
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from contextlib import asynccontextmanager, contextmanager
from functools import wraps, lru_cache
from collections import defaultdict, deque
import weakref
import tracemalloc
import logging
import json
import hashlib
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
try:
    import aioredis  # type: ignore
except Exception:  # pragma: no cover - optional in tests
    aioredis = None  # type: ignore[assignment]
import numpy as np
import torch
from cachetools import TTLCache, LRUCache
from prometheus_client import Counter, Histogram, Gauge, Summary
import orjson
import zstandard as zstd
import msgpack
from onyx.utils.logger import setup_logger
from onyx.server.features.ads.optimized_config import settings
from typing import Any, List, Dict, Optional
"""
Advanced Performance Optimization System for Onyx Ads Backend

This module provides comprehensive performance optimizations including:
- Multi-level caching with intelligent eviction
- Memory management and garbage collection
- Async processing with worker pools
- Performance monitoring and profiling
- Database query optimization
- Resource pooling and connection management
"""


logger = setup_logger()

T = TypeVar('T')

# Prometheus metrics
PERFORMANCE_METRICS = {
    'cache_hits': Counter('cache_hits_total', 'Total cache hits', ['cache_type', 'operation']),
    'cache_misses': Counter('cache_misses_total', 'Total cache misses', ['cache_type', 'operation']),
    'cache_size': Gauge('cache_size_bytes', 'Cache size in bytes', ['cache_type']),
    'memory_usage': Gauge('memory_usage_bytes', 'Memory usage in bytes', ['type']),
    'processing_time': Histogram('processing_time_seconds', 'Processing time in seconds', ['operation']),
    'async_operations': Counter('async_operations_total', 'Total async operations', ['operation', 'status']),
    'database_queries': Histogram('database_query_time_seconds', 'Database query time', ['query_type']),
    'gc_collections': Counter('gc_collections_total', 'Garbage collection events', ['generation']),
    'resource_usage': Gauge('resource_usage_percent', 'Resource usage percentage', ['resource_type']),
}

@dataclass
class PerformanceConfig:
    """Configuration for performance optimization."""
    
    # Cache settings
    cache_ttl: int = 3600
    cache_max_size: int = 10000
    cache_cleanup_interval: int = 300
    
    # Memory settings
    max_memory_usage: int = 1024 * 1024 * 1024  # 1GB
    memory_cleanup_threshold: float = 0.8
    gc_threshold: int = 1000
    
    # Async settings
    max_workers: int = 10
    max_processes: int = 4
    task_timeout: int = 30
    
    # Database settings
    query_cache_ttl: int = 1800
    connection_pool_size: int = 20
    query_timeout: int = 10
    
    # Monitoring settings
    metrics_interval: int = 60
    profiling_enabled: bool = True
    tracemalloc_enabled: bool = True
    
    # Performance thresholds
    slow_query_threshold: float = 1.0
    memory_warning_threshold: float = 0.7
    cache_hit_rate_threshold: float = 0.8

class MemoryManager:
    """Advanced memory management with monitoring and optimization."""
    
    def __init__(self, config: PerformanceConfig):
        
    """__init__ function."""
self.config = config
        self._memory_history = deque(maxlen=1000)
        self._gc_counter = 0
        self._last_gc_time = time.time()
        self._memory_warnings = 0
        self._lock = threading.Lock()
        
        if config.tracemalloc_enabled:
            tracemalloc.start()
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            'rss': memory_info.rss,
            'vms': memory_info.vms,
            'percent': process.memory_percent(),
            'available': psutil.virtual_memory().available,
            'total': psutil.virtual_memory().total,
        }
    
    def should_cleanup_memory(self) -> bool:
        """Check if memory cleanup is needed."""
        memory_usage = self.get_memory_usage()
        usage_ratio = memory_usage['rss'] / memory_usage['total']
        
        PERFORMANCE_METRICS['memory_usage'].labels(type='rss').set(memory_usage['rss'])
        PERFORMANCE_METRICS['memory_usage'].labels(type='vms').set(memory_usage['vms'])
        PERFORMANCE_METRICS['resource_usage'].labels(resource_type='memory').set(usage_ratio * 100)
        
        return usage_ratio > self.config.memory_cleanup_threshold
    
    def cleanup_memory(self, force: bool = False) -> Dict[str, Any]:
        """Perform memory cleanup and optimization."""
        with self._lock:
            start_time = time.time()
            initial_memory = self.get_memory_usage()
            
            # Force garbage collection
            if force or self._gc_counter >= self.config.gc_threshold:
                collected = gc.collect()
                self._gc_counter = 0
                self._last_gc_time = time.time()
                PERFORMANCE_METRICS['gc_collections'].labels(generation='all').inc()
                
                # Clear PyTorch cache if available
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                collected = 0
                self._gc_counter += 1
            
            # Clear weak references
            weakref._weakref._cleanup_dead_references()
            
            final_memory = self.get_memory_usage()
            cleanup_time = time.time() - start_time
            
            result = {
                'initial_memory': initial_memory,
                'final_memory': final_memory,
                'memory_freed': initial_memory['rss'] - final_memory['rss'],
                'gc_collected': collected,
                'cleanup_time': cleanup_time,
            }
            
            # Record memory history
            self._memory_history.append({
                'timestamp': datetime.now(),
                'memory_usage': final_memory,
                'cleanup_performed': collected > 0
            })
            
            logger.info(f"Memory cleanup completed: {result['memory_freed']} bytes freed in {cleanup_time:.3f}s")
            return result
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics."""
        current_memory = self.get_memory_usage()
        
        return {
            'current': current_memory,
            'history': list(self._memory_history),
            'gc_stats': {
                'counter': self._gc_counter,
                'last_gc': self._last_gc_time,
                'threshold': self.config.gc_threshold,
            },
            'warnings': self._memory_warnings,
            'tracemalloc': self._get_tracemalloc_stats() if self.config.tracemalloc_enabled else None,
        }
    
    def _get_tracemalloc_stats(self) -> Dict[str, Any]:
        """Get tracemalloc statistics."""
        try:
            snapshot = tracemalloc.take_snapshot()
            top_stats = snapshot.statistics('lineno')[:10]
            
            return {
                'total_memory': sum(stat.size for stat in top_stats),
                'top_allocations': [
                    {
                        'size': stat.size,
                        'count': stat.count,
                        'filename': stat.traceback.format()[-1]
                    }
                    for stat in top_stats
                ]
            }
        except Exception as e:
            logger.warning(f"Failed to get tracemalloc stats: {e}")
            return {}

class AdvancedCache:
    """Multi-level cache with intelligent eviction and compression."""
    
    def __init__(self, config: PerformanceConfig, redis_client: Optional[aioredis.Redis] = None):
        
    """__init__ function."""
self.config = config
        self.redis_client = redis_client
        
        # Memory caches
        self.l1_cache = TTLCache(maxsize=config.cache_max_size, ttl=config.cache_ttl)
        self.l2_cache = LRUCache(maxsize=config.cache_max_size * 2)
        
        # Statistics
        self._hits = defaultdict(int)
        self._misses = defaultdict(int)
        self._compression_stats = {'compressed': 0, 'decompressed': 0, 'bytes_saved': 0}
        
        # Compression
        self._compressor = zstd.ZstdCompressor(level=3)
        self._decompressor = zstd.ZstdDecompressor()
        
        # Serialization
        self._serializer = orjson
        self._packer = msgpack
    
    def _get_cache_key(self, key: str, prefix: str = "cache") -> str:
        """Generate cache key with prefix."""
        return f"{prefix}:{hashlib.md5(key.encode()).hexdigest()}"
    
    def _compress_data(self, data: Any) -> bytes:
        """Compress data using zstd."""
        try:
            serialized = self._serializer.dumps(data)
            compressed = self._compressor.compress(serialized)
            
            self._compression_stats['compressed'] += 1
            self._compression_stats['bytes_saved'] += len(serialized) - len(compressed)
            
            return compressed
        except Exception as e:
            logger.warning(f"Compression failed: {e}")
            return self._serializer.dumps(data)
    
    def _decompress_data(self, data: bytes) -> Any:
        """Decompress data using zstd."""
        try:
            decompressed = self._decompressor.decompress(data)
            result = self._serializer.loads(decompressed)
            
            self._compression_stats['decompressed'] += 1
            return result
        except Exception as e:
            logger.warning(f"Decompression failed: {e}")
            return self._serializer.loads(data)
    
    def get(self, key: str, cache_type: str = "l1") -> Optional[Any]:
        """Get value from cache with statistics."""
        cache_key = self._get_cache_key(key)
        start_time = time.time()
        
        try:
            # Try L1 cache first
            if cache_type in ["l1", "both"]:
                value = self.l1_cache.get(cache_key)
                if value is not None:
                    self._hits['l1'] += 1
                    PERFORMANCE_METRICS['cache_hits'].labels(cache_type='l1', operation='get').inc()
                    return self._decompress_data(value)
            
            # Try L2 cache
            if cache_type in ["l2", "both"]:
                value = self.l2_cache.get(cache_key)
                if value is not None:
                    self._hits['l2'] += 1
                    PERFORMANCE_METRICS['cache_hits'].labels(cache_type='l2', operation='get').inc()
                    # Promote to L1
                    self.l1_cache[cache_key] = value
                    return self._decompress_data(value)
            
            # Try Redis cache
            if self.redis_client and cache_type in ["redis", "both"]:
                # Note: This would need to be async in practice
                pass
            
            # Cache miss
            self._misses[cache_type] += 1
            PERFORMANCE_METRICS['cache_misses'].labels(cache_type=cache_type, operation='get').inc()
            
            processing_time = time.time() - start_time
            PERFORMANCE_METRICS['processing_time'].labels(operation='cache_get').observe(processing_time)
            
            return None
            
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None, cache_type: str = "both"):
        """Set value in cache with compression."""
        cache_key = self._get_cache_key(key)
        start_time = time.time()
        
        try:
            compressed_value = self._compress_data(value)
            
            if cache_type in ["l1", "both"]:
                self.l1_cache[cache_key] = compressed_value
            
            if cache_type in ["l2", "both"]:
                self.l2_cache[cache_key] = compressed_value
            
            # Update cache size metrics
            cache_size = len(compressed_value)
            PERFORMANCE_METRICS['cache_size'].labels(cache_type='l1').set(len(self.l1_cache))
            PERFORMANCE_METRICS['cache_size'].labels(cache_type='l2').set(len(self.l2_cache))
            
            processing_time = time.time() - start_time
            PERFORMANCE_METRICS['processing_time'].labels(operation='cache_set').observe(processing_time)
            
        except Exception as e:
            logger.error(f"Cache set error: {e}")
    
    def clear(self, cache_type: str = "all"):
        """Clear cache entries."""
        if cache_type in ["l1", "all"]:
            self.l1_cache.clear()
        if cache_type in ["l2", "all"]:
            self.l2_cache.clear()
        
        logger.info(f"Cache cleared: {cache_type}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_hits = sum(self._hits.values())
        total_misses = sum(self._misses.values())
        total_requests = total_hits + total_misses
        hit_rate = total_hits / total_requests if total_requests > 0 else 0
        
        return {
            'hits': dict(self._hits),
            'misses': dict(self._misses),
            'hit_rate': hit_rate,
            'compression_stats': self._compression_stats,
            'cache_sizes': {
                'l1': len(self.l1_cache),
                'l2': len(self.l2_cache),
            }
        }

class AsyncTaskManager:
    """Manages async tasks with worker pools and monitoring."""
    
    def __init__(self, config: PerformanceConfig):
        
    """__init__ function."""
self.config = config
        self._thread_pool = ThreadPoolExecutor(max_workers=config.max_workers)
        self._process_pool = ProcessPoolExecutor(max_workers=config.max_processes)
        self._task_queue = asyncio.Queue(maxsize=1000)
        self._running_tasks = set()
        self._task_stats = defaultdict(lambda: {'completed': 0, 'failed': 0, 'total_time': 0})
        self._lock = asyncio.Lock()
        
    async def submit_task(self, func: Callable, *args, **kwargs) -> asyncio.Task:
        """Submit a task for async execution."""
        task = asyncio.create_task(self._execute_task(func, *args, **kwargs))
        self._running_tasks.add(task)
        task.add_done_callback(self._running_tasks.discard)
        
        PERFORMANCE_METRICS['async_operations'].labels(operation='submit', status='success').inc()
        return task
    
    async def _execute_task(self, func: Callable, *args, **kwargs):
        """Execute a task with monitoring."""
        start_time = time.time()
        operation_name = func.__name__ if hasattr(func, '__name__') else 'unknown'
        
        try:
            if asyncio.iscoroutinefunction(func):
                result = await asyncio.wait_for(func(*args, **kwargs), timeout=self.config.task_timeout)
            else:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    self._thread_pool, 
                    func, 
                    *args, 
                    **kwargs
                )
            
            processing_time = time.time() - start_time
            self._task_stats[operation_name]['completed'] += 1
            self._task_stats[operation_name]['total_time'] += processing_time
            
            PERFORMANCE_METRICS['processing_time'].labels(operation=operation_name).observe(processing_time)
            PERFORMANCE_METRICS['async_operations'].labels(operation=operation_name, status='success').inc()
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            self._task_stats[operation_name]['failed'] += 1
            
            PERFORMANCE_METRICS['async_operations'].labels(operation=operation_name, status='error').inc()
            logger.error(f"Task execution failed: {e}")
            raise
    
    async def batch_submit(self, tasks: List[tuple]) -> List[asyncio.Task]:
        """Submit multiple tasks in batch."""
        submitted_tasks = []
        
        for task_data in tasks:
            if isinstance(task_data, tuple):
                func, args, kwargs = task_data[0], task_data[1:], {}
            else:
                func, args, kwargs = task_data, (), {}
            
            task = await self.submit_task(func, *args, **kwargs)
            submitted_tasks.append(task)
        
        return submitted_tasks
    
    async def wait_for_tasks(self, tasks: List[asyncio.Task], timeout: Optional[float] = None):
        """Wait for multiple tasks to complete."""
        try:
            results = await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=timeout)
            return results
        except asyncio.TimeoutError:
            logger.warning(f"Task batch timeout after {timeout}s")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Get task execution statistics."""
        return {
            'running_tasks': len(self._running_tasks),
            'task_stats': dict(self._task_stats),
            'pool_stats': {
                'thread_pool_size': self._thread_pool._max_workers,
                'process_pool_size': self._process_pool._max_workers,
            }
        }
    
    async def shutdown(self) -> Any:
        """Shutdown task manager."""
        self._thread_pool.shutdown(wait=True)
        self._process_pool.shutdown(wait=True)
        
        # Cancel running tasks
        for task in self._running_tasks:
            task.cancel()
        
        await asyncio.gather(*self._running_tasks, return_exceptions=True)

class DatabaseOptimizer:
    """Database query optimization and connection management."""
    
    def __init__(self, config: PerformanceConfig):
        
    """__init__ function."""
self.config = config
        self._query_cache = TTLCache(maxsize=1000, ttl=config.query_cache_ttl)
        self._slow_queries = deque(maxlen=100)
        self._query_stats = defaultdict(lambda: {'count': 0, 'total_time': 0, 'avg_time': 0})
    
    @contextmanager
    def query_timer(self, query_type: str = "unknown"):
        """Context manager for timing database queries."""
        start_time = time.time()
        try:
            yield
        finally:
            query_time = time.time() - start_time
            
            # Update statistics
            self._query_stats[query_type]['count'] += 1
            self._query_stats[query_type]['total_time'] += query_time
            self._query_stats[query_type]['avg_time'] = (
                self._query_stats[query_type]['total_time'] / 
                self._query_stats[query_type]['count']
            )
            
            # Record metrics
            PERFORMANCE_METRICS['database_queries'].labels(query_type=query_type).observe(query_time)
            
            # Track slow queries
            if query_time > self.config.slow_query_threshold:
                self._slow_queries.append({
                    'query_type': query_type,
                    'execution_time': query_time,
                    'timestamp': datetime.now()
                })
                logger.warning(f"Slow query detected: {query_type} took {query_time:.3f}s")
    
    def get_cached_query(self, query_key: str) -> Optional[Any]:
        """Get cached query result."""
        return self._query_cache.get(query_key)
    
    def cache_query_result(self, query_key: str, result: Any):
        """Cache query result."""
        self._query_cache[query_key] = result
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database optimization statistics."""
        return {
            'query_stats': dict(self._query_stats),
            'slow_queries': list(self._slow_queries),
            'cache_size': len(self._query_cache),
            'cache_hit_rate': self._calculate_cache_hit_rate(),
        }
    
    def _calculate_cache_hit_rate(self) -> float:
        """Calculate query cache hit rate."""
        # This would need actual hit/miss tracking in a real implementation
        return 0.0

class PerformanceOptimizer:
    """Main performance optimization orchestrator."""
    
    def __init__(self, config: Optional[PerformanceConfig] = None):
        
    """__init__ function."""
self.config = config or PerformanceConfig()
        self.memory_manager = MemoryManager(self.config)
        self.cache = AdvancedCache(self.config)
        self.task_manager = AsyncTaskManager(self.config)
        self.db_optimizer = DatabaseOptimizer(self.config)
        
        self._monitoring_task = None
        self._cleanup_task = None
        self._started = False
    
    async def start(self) -> Any:
        """Start the performance optimizer."""
        if self._started:
            return
        
        self._started = True
        
        # Start monitoring task
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        # Start cleanup task
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        logger.info("Performance optimizer started")
    
    async def stop(self) -> Any:
        """Stop the performance optimizer."""
        if not self._started:
            return
        
        self._started = False
        
        # Cancel monitoring tasks
        if self._monitoring_task:
            self._monitoring_task.cancel()
        if self._cleanup_task:
            self._cleanup_task.cancel()
        
        # Shutdown task manager
        await self.task_manager.shutdown()
        
        logger.info("Performance optimizer stopped")
    
    async def _monitoring_loop(self) -> Any:
        """Main monitoring loop."""
        while self._started:
            try:
                # Check memory usage
                if self.memory_manager.should_cleanup_memory():
                    await self.task_manager.submit_task(self.memory_manager.cleanup_memory)
                
                # Update resource usage metrics
                self._update_resource_metrics()
                
                await asyncio.sleep(self.config.metrics_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(5)
    
    async def _cleanup_loop(self) -> Any:
        """Periodic cleanup loop."""
        while self._started:
            try:
                # Memory cleanup
                await self.task_manager.submit_task(self.memory_manager.cleanup_memory)
                
                # Cache cleanup (if needed)
                # This would be implemented based on cache statistics
                
                await asyncio.sleep(self.config.cache_cleanup_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")
                await asyncio.sleep(30)
    
    def _update_resource_metrics(self) -> Any:
        """Update resource usage metrics."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            PERFORMANCE_METRICS['resource_usage'].labels(resource_type='cpu').set(cpu_percent)
            
            # Disk usage
            disk_usage = psutil.disk_usage('/')
            disk_percent = (disk_usage.used / disk_usage.total) * 100
            PERFORMANCE_METRICS['resource_usage'].labels(resource_type='disk').set(disk_percent)
            
        except Exception as e:
            logger.warning(f"Failed to update resource metrics: {e}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        return {
            'memory': self.memory_manager.get_memory_stats(),
            'cache': self.cache.get_stats(),
            'tasks': self.task_manager.get_stats(),
            'database': self.db_optimizer.get_stats(),
            'config': {
                'cache_ttl': self.config.cache_ttl,
                'max_workers': self.config.max_workers,
                'memory_threshold': self.config.memory_cleanup_threshold,
            }
        }

# Performance decorators and utilities
def performance_monitor(operation_name: str = None):
    """Decorator for monitoring function performance."""
    def decorator(func) -> Any:
        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            operation = operation_name or func.__name__
            
            try:
                result = await func(*args, **kwargs)
                processing_time = time.time() - start_time
                PERFORMANCE_METRICS['processing_time'].labels(operation=operation).observe(processing_time)
                PERFORMANCE_METRICS['async_operations'].labels(operation=operation, status='success').inc()
                return result
            except Exception as e:
                processing_time = time.time() - start_time
                PERFORMANCE_METRICS['processing_time'].labels(operation=operation).observe(processing_time)
                PERFORMANCE_METRICS['async_operations'].labels(operation=operation, status='error').inc()
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            operation = operation_name or func.__name__
            
            try:
                result = func(*args, **kwargs)
                processing_time = time.time() - start_time
                PERFORMANCE_METRICS['processing_time'].labels(operation=operation).observe(processing_time)
                return result
            except Exception as e:
                processing_time = time.time() - start_time
                PERFORMANCE_METRICS['processing_time'].labels(operation=operation).observe(processing_time)
                raise
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator

def cache_result(ttl: int = 3600, cache_type: str = "both"):
    """Decorator for caching function results."""
    def decorator(func) -> Any:
        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            # Generate cache key
            cache_key = f"{func.__name__}:{hash(str(args) + str(sorted(kwargs.items())))}"
            
            # Try to get from cache
            cached_result = optimizer.cache.get(cache_key, cache_type)
            if cached_result is not None:
                return cached_result
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Cache result
            optimizer.cache.set(cache_key, result, ttl, cache_type)
            
            return result
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            # Generate cache key
            cache_key = f"{func.__name__}:{hash(str(args) + str(sorted(kwargs.items())))}"
            
            # Try to get from cache
            cached_result = optimizer.cache.get(cache_key, cache_type)
            if cached_result is not None:
                return cached_result
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Cache result
            optimizer.cache.set(cache_key, result, ttl, cache_type)
            
            return result
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator

# Global optimizer instance
optimizer = PerformanceOptimizer()

# Performance monitoring context managers
@asynccontextmanager
async def performance_context(operation_name: str):
    """Context manager for performance monitoring."""
    start_time = time.time()
    try:
        yield
    finally:
        processing_time = time.time() - start_time
        PERFORMANCE_METRICS['processing_time'].labels(operation=operation_name).observe(processing_time)

@contextmanager
def memory_context():
    """Context manager for memory monitoring."""
    initial_memory = optimizer.memory_manager.get_memory_usage()
    try:
        yield
    finally:
        final_memory = optimizer.memory_manager.get_memory_usage()
        memory_diff = final_memory['rss'] - initial_memory['rss']
        if memory_diff > 1024 * 1024:  # 1MB threshold
            logger.info(f"Memory usage changed by {memory_diff / 1024 / 1024:.2f}MB") 