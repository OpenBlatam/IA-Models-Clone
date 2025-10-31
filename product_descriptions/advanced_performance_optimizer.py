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
import json
import logging
import psutil
import gc
from typing import Dict, Any, Optional, List, Callable, Union, TypeVar, Generic, Tuple
from functools import wraps, lru_cache
from pathlib import Path
import pickle
import hashlib
from datetime import datetime, timedelta
import aiofiles
import aiohttp
from contextlib import asynccontextmanager
import weakref
import threading
from collections import defaultdict, deque
import statistics
import tracemalloc
from dataclasses import dataclass, field
from enum import Enum
                import gzip
                import gzip
from typing import Any, List, Dict, Optional
#!/usr/bin/env python3
"""
Advanced Performance Optimizer
Product Descriptions Feature - Production-Grade Performance Optimization with Advanced Features
"""


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Type variables for generic caching
T = TypeVar('T')
K = TypeVar('K')

class CacheStrategy(Enum):
    """Cache strategy enumeration"""
    LRU = "lru"
    LFU = "lfu"
    FIFO = "fifo"
    TTL = "ttl"

class MemoryPolicy(Enum):
    """Memory management policy"""
    AGGRESSIVE = "aggressive"
    CONSERVATIVE = "conservative"
    ADAPTIVE = "adaptive"

@dataclass
class PerformanceMetrics:
    """Performance metrics data class"""
    operation_name: str
    duration_ms: float
    memory_usage_mb: float
    cpu_usage_percent: float
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CacheStats:
    """Cache statistics data class"""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    size: int = 0
    max_size: int = 0
    hit_rate: float = 0.0
    avg_load_time_ms: float = 0.0
    memory_usage_mb: float = 0.0

class AdvancedAsyncCache(Generic[K, T]):
    """Advanced async cache with multiple strategies and memory management"""
    
    def __init__(
        self, 
        ttl_seconds: int = 300, 
        max_size: int = 1000,
        strategy: CacheStrategy = CacheStrategy.LRU,
        memory_policy: MemoryPolicy = MemoryPolicy.ADAPTIVE,
        enable_compression: bool = True,
        enable_metrics: bool = True
    ):
        
    """__init__ function."""
self.ttl_seconds = ttl_seconds
        self.max_size = max_size
        self.strategy = strategy
        self.memory_policy = memory_policy
        self.enable_compression = enable_compression
        self.enable_metrics = enable_metrics
        
        # Cache storage
        self._cache: Dict[K, Dict[str, Any]] = {}
        self._access_times: Dict[K, float] = {}
        self._access_counts: Dict[K, int] = defaultdict(int)
        self._creation_times: Dict[K, float] = {}
        self._lock = asyncio.Lock()
        
        # Metrics
        self._stats = CacheStats(max_size=max_size)
        self._load_times: deque = deque(maxlen=100)
        
        # Memory management
        self._memory_threshold = 0.8  # 80% of available memory
        self._last_memory_check = 0
        self._memory_check_interval = 60  # seconds
        
        # Background cleanup
        self._cleanup_task: Optional[asyncio.Task] = None
        self._start_cleanup_task()
    
    def _start_cleanup_task(self) -> None:
        """Start background cleanup task"""
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
    
    async def _cleanup_loop(self) -> None:
        """Background cleanup loop"""
        while True:
            try:
                await asyncio.sleep(30)  # Run every 30 seconds
                await self._cleanup_expired()
                await self._check_memory_usage()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")
    
    async def _cleanup_expired(self) -> None:
        """Clean up expired entries"""
        async with self._lock:
            current_time = time.time()
            expired_keys = [
                key for key, entry in self._cache.items()
                if current_time - entry['timestamp'] > self.ttl_seconds
            ]
            
            for key in expired_keys:
                await self._remove(key)
                self._stats.evictions += 1
    
    async def _check_memory_usage(self) -> None:
        """Check memory usage and apply policy"""
        current_time = time.time()
        if current_time - self._last_memory_check < self._memory_check_interval:
            return
        
        self._last_memory_check = current_time
        memory_percent = psutil.virtual_memory().percent / 100
        
        if memory_percent > self._memory_threshold:
            await self._apply_memory_policy()
    
    async def _apply_memory_policy(self) -> None:
        """Apply memory management policy"""
        if self.memory_policy == MemoryPolicy.AGGRESSIVE:
            # Clear 50% of cache
            await self._clear_percentage(0.5)
        elif self.memory_policy == MemoryPolicy.CONSERVATIVE:
            # Clear 25% of cache
            await self._clear_percentage(0.25)
        elif self.memory_policy == MemoryPolicy.ADAPTIVE:
            # Adaptive clearing based on memory pressure
            memory_percent = psutil.virtual_memory().percent / 100
            clear_percentage = min(0.5, (memory_percent - self._memory_threshold) * 2)
            await self._clear_percentage(clear_percentage)
    
    async def _clear_percentage(self, percentage: float) -> None:
        """Clear a percentage of cache entries"""
        if not self._cache:
            return
        
        target_size = int(len(self._cache) * (1 - percentage))
        
        if self.strategy == CacheStrategy.LRU:
            # Remove least recently used
            sorted_keys = sorted(
                self._access_times.keys(),
                key=lambda k: self._access_times[k]
            )
        elif self.strategy == CacheStrategy.LFU:
            # Remove least frequently used
            sorted_keys = sorted(
                self._access_counts.keys(),
                key=lambda k: self._access_counts[k]
            )
        else:
            # FIFO - remove oldest
            sorted_keys = sorted(
                self._creation_times.keys(),
                key=lambda k: self._creation_times[k]
            )
        
        keys_to_remove = sorted_keys[:len(sorted_keys) - target_size]
        for key in keys_to_remove:
            await self._remove(key)
            self._stats.evictions += 1
    
    async def get(self, key: K) -> Optional[T]:
        """Get value from cache with metrics"""
        start_time = time.time()
        
        async with self._lock:
            if key not in self._cache:
                self._stats.misses += 1
                return None
            
            cache_entry = self._cache[key]
            if time.time() - cache_entry['timestamp'] > self.ttl_seconds:
                await self._remove(key)
                self._stats.misses += 1
                return None
            
            # Update access tracking
            self._access_times[key] = time.time()
            self._access_counts[key] += 1
            self._stats.hits += 1
            
            # Update hit rate
            total_requests = self._stats.hits + self._stats.misses
            self._stats.hit_rate = self._stats.hits / total_requests if total_requests > 0 else 0.0
            
            value = cache_entry['value']
        
        # Record load time
        load_time = (time.time() - start_time) * 1000
        self._load_times.append(load_time)
        self._stats.avg_load_time_ms = statistics.mean(self._load_times)
        
        return value
    
    async def set(self, key: K, value: T) -> None:
        """Set value in cache with compression and metrics"""
        start_time = time.time()
        
        # Compress value if enabled
        if self.enable_compression:
            value = await self._compress_value(value)
        
        async with self._lock:
            # Check if cache is full
            if len(self._cache) >= self.max_size:
                await self._evict_entry()
            
            current_time = time.time()
            self._cache[key] = {
                'value': value,
                'timestamp': current_time,
                'size': await self._estimate_size(value)
            }
            self._access_times[key] = current_time
            self._creation_times[key] = current_time
            self._access_counts[key] = 0
            
            self._stats.size = len(self._cache)
        
        # Update memory usage
        if self.enable_metrics:
            self._stats.memory_usage_mb = await self._calculate_memory_usage()
    
    async def _compress_value(self, value: T) -> T:
        """Compress value if it's large"""
        try:
            if isinstance(value, (str, bytes)) and len(value) > 1024:
                if isinstance(value, str):
                    value = value.encode('utf-8')
                compressed = gzip.compress(value)
                if len(compressed) < len(value):
                    return compressed
        except Exception as e:
            logger.warning(f"Compression failed: {e}")
        return value
    
    async def _decompress_value(self, value: T) -> T:
        """Decompress value if it was compressed"""
        try:
            if isinstance(value, bytes) and value.startswith(b'\x1f\x8b'):
                decompressed = gzip.decompress(value)
                try:
                    return decompressed.decode('utf-8')
                except UnicodeDecodeError:
                    return decompressed
        except Exception as e:
            logger.warning(f"Decompression failed: {e}")
        return value
    
    async def _estimate_size(self, value: T) -> int:
        """Estimate memory size of value"""
        try:
            if isinstance(value, (str, bytes)):
                return len(value)
            elif isinstance(value, (dict, list)):
                return len(str(value))
            else:
                return 1
        except Exception:
            return 1
    
    async def _calculate_memory_usage(self) -> float:
        """Calculate cache memory usage in MB"""
        try:
            total_size = sum(
                entry.get('size', 1) for entry in self._cache.values()
            )
            return total_size / (1024 * 1024)  # Convert to MB
        except Exception:
            return 0.0
    
    async def _evict_entry(self) -> None:
        """Evict entry based on strategy"""
        if not self._cache:
            return
        
        if self.strategy == CacheStrategy.LRU:
            lru_key = min(
                self._access_times.keys(),
                key=lambda k: self._access_times[k]
            )
        elif self.strategy == CacheStrategy.LFU:
            lfu_key = min(
                self._access_counts.keys(),
                key=lambda k: self._access_counts[k]
            )
        else:  # FIFO
            fifo_key = min(
                self._creation_times.keys(),
                key=lambda k: self._creation_times[k]
            )
        
        key_to_evict = lru_key if self.strategy == CacheStrategy.LRU else (
            lfu_key if self.strategy == CacheStrategy.LFU else fifo_key
        )
        await self._remove(key_to_evict)
        self._stats.evictions += 1
    
    async def _remove(self, key: K) -> None:
        """Remove key from cache"""
        self._cache.pop(key, None)
        self._access_times.pop(key, None)
        self._access_counts.pop(key, None)
        self._creation_times.pop(key, None)
    
    async def clear(self) -> None:
        """Clear all cache entries"""
        async with self._lock:
            self._cache.clear()
            self._access_times.clear()
            self._access_counts.clear()
            self._creation_times.clear()
            self._stats.size = 0
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get detailed cache statistics"""
        async with self._lock:
            return {
                'hits': self._stats.hits,
                'misses': self._stats.misses,
                'evictions': self._stats.evictions,
                'size': self._stats.size,
                'max_size': self._stats.max_size,
                'hit_rate': self._stats.hit_rate,
                'avg_load_time_ms': self._stats.avg_load_time_ms,
                'memory_usage_mb': self._stats.memory_usage_mb,
                'strategy': self.strategy.value,
                'memory_policy': self.memory_policy.value,
                'compression_enabled': self.enable_compression
            }
    
    async def close(self) -> None:
        """Close cache and cleanup resources"""
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

class AdvancedPerformanceMonitor:
    """Advanced performance monitoring with detailed metrics"""
    
    def __init__(self, enable_memory_tracking: bool = True, enable_cpu_tracking: bool = True):
        
    """__init__ function."""
self.enable_memory_tracking = enable_memory_tracking
        self.enable_cpu_tracking = enable_cpu_tracking
        
        # Metrics storage
        self._metrics: Dict[str, List[PerformanceMetrics]] = defaultdict(list)
        self._operation_stats: Dict[str, Dict[str, float]] = defaultdict(dict)
        self._lock = asyncio.Lock()
        
        # System monitoring
        self._system_metrics: Dict[str, List[float]] = defaultdict(list)
        self._last_system_check = 0
        self._system_check_interval = 5  # seconds
        
        # Memory tracking
        if self.enable_memory_tracking:
            tracemalloc.start()
        
        # Background monitoring
        self._monitor_task: Optional[asyncio.Task] = None
        self._start_monitoring()
    
    def _start_monitoring(self) -> None:
        """Start background monitoring"""
        if self._monitor_task is None or self._monitor_task.done():
            self._monitor_task = asyncio.create_task(self._monitor_loop())
    
    async def _monitor_loop(self) -> None:
        """Background monitoring loop"""
        while True:
            try:
                await asyncio.sleep(1)  # Monitor every second
                await self._collect_system_metrics()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
    
    async def _collect_system_metrics(self) -> None:
        """Collect system metrics"""
        current_time = time.time()
        if current_time - self._last_system_check < self._system_check_interval:
            return
        
        self._last_system_check = current_time
        
        # Memory metrics
        if self.enable_memory_tracking:
            memory = psutil.virtual_memory()
            self._system_metrics['memory_percent'].append(memory.percent)
            self._system_metrics['memory_available_mb'].append(memory.available / (1024 * 1024))
            
            # Keep only last 1000 measurements
            if len(self._system_metrics['memory_percent']) > 1000:
                self._system_metrics['memory_percent'] = self._system_metrics['memory_percent'][-1000:]
                self._system_metrics['memory_available_mb'] = self._system_metrics['memory_available_mb'][-1000:]
        
        # CPU metrics
        if self.enable_cpu_tracking:
            cpu_percent = psutil.cpu_percent(interval=1)
            self._system_metrics['cpu_percent'].append(cpu_percent)
            
            if len(self._system_metrics['cpu_percent']) > 1000:
                self._system_metrics['cpu_percent'] = self._system_metrics['cpu_percent'][-1000:]
    
    async def record_metric(self, operation_name: str, duration_ms: float, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Record performance metric"""
        async with self._lock:
            # Get current system metrics
            memory_usage_mb = 0.0
            cpu_usage_percent = 0.0
            
            if self.enable_memory_tracking:
                memory = psutil.virtual_memory()
                memory_usage_mb = memory.used / (1024 * 1024)
            
            if self.enable_cpu_tracking:
                cpu_usage_percent = psutil.cpu_percent()
            
            # Create metric
            metric = PerformanceMetrics(
                operation_name=operation_name,
                duration_ms=duration_ms,
                memory_usage_mb=memory_usage_mb,
                cpu_usage_percent=cpu_usage_percent,
                metadata=metadata or {}
            )
            
            # Store metric
            self._metrics[operation_name].append(metric)
            
            # Keep only last 1000 metrics per operation
            if len(self._metrics[operation_name]) > 1000:
                self._metrics[operation_name] = self._metrics[operation_name][-1000:]
            
            # Update operation statistics
            self._update_operation_stats(operation_name, metric)
    
    def _update_operation_stats(self, operation_name: str, metric: PerformanceMetrics) -> None:
        """Update operation statistics"""
        if operation_name not in self._operation_stats:
            self._operation_stats[operation_name] = {
                'count': 0,
                'total_duration': 0.0,
                'min_duration': float('inf'),
                'max_duration': 0.0,
                'avg_duration': 0.0,
                'total_memory': 0.0,
                'avg_memory': 0.0,
                'total_cpu': 0.0,
                'avg_cpu': 0.0
            }
        
        stats = self._operation_stats[operation_name]
        stats['count'] += 1
        stats['total_duration'] += metric.duration_ms
        stats['min_duration'] = min(stats['min_duration'], metric.duration_ms)
        stats['max_duration'] = max(stats['max_duration'], metric.duration_ms)
        stats['avg_duration'] = stats['total_duration'] / stats['count']
        
        stats['total_memory'] += metric.memory_usage_mb
        stats['avg_memory'] = stats['total_memory'] / stats['count']
        
        stats['total_cpu'] += metric.cpu_usage_percent
        stats['avg_cpu'] = stats['total_cpu'] / stats['count']
    
    async def get_metrics(self, operation_name: Optional[str] = None) -> Dict[str, Any]:
        """Get performance metrics"""
        async with self._lock:
            if operation_name:
                if operation_name not in self._metrics:
                    return {}
                
                metrics = self._metrics[operation_name]
                stats = self._operation_stats[operation_name]
                
                return {
                    'operation_name': operation_name,
                    'metrics_count': len(metrics),
                    'stats': stats,
                    'recent_metrics': [
                        {
                            'duration_ms': m.duration_ms,
                            'memory_usage_mb': m.memory_usage_mb,
                            'cpu_usage_percent': m.cpu_usage_percent,
                            'timestamp': m.timestamp.isoformat(),
                            'metadata': m.metadata
                        }
                        for m in metrics[-10:]  # Last 10 metrics
                    ]
                }
            else:
                # Return all metrics
                return {
                    'operations': list(self._metrics.keys()),
                    'operation_stats': dict(self._operation_stats),
                    'system_metrics': {
                        'memory_percent': {
                            'current': self._system_metrics['memory_percent'][-1] if self._system_metrics['memory_percent'] else 0,
                            'avg': statistics.mean(self._system_metrics['memory_percent']) if self._system_metrics['memory_percent'] else 0,
                            'max': max(self._system_metrics['memory_percent']) if self._system_metrics['memory_percent'] else 0
                        },
                        'cpu_percent': {
                            'current': self._system_metrics['cpu_percent'][-1] if self._system_metrics['cpu_percent'] else 0,
                            'avg': statistics.mean(self._system_metrics['cpu_percent']) if self._system_metrics['cpu_percent'] else 0,
                            'max': max(self._system_metrics['cpu_percent']) if self._system_metrics['cpu_percent'] else 0
                        }
                    }
                }
    
    async def get_memory_snapshot(self) -> Dict[str, Any]:
        """Get memory snapshot using tracemalloc"""
        if not self.enable_memory_tracking:
            return {}
        
        try:
            snapshot = tracemalloc.take_snapshot()
            top_stats = snapshot.statistics('lineno')
            
            return {
                'total_memory_mb': sum(stat.size for stat in top_stats) / (1024 * 1024),
                'top_allocations': [
                    {
                        'file': stat.traceback.format()[-1],
                        'size_mb': stat.size / (1024 * 1024),
                        'count': stat.count
                    }
                    for stat in top_stats[:10]
                ]
            }
        except Exception as e:
            logger.error(f"Memory snapshot error: {e}")
            return {}
    
    async def close(self) -> None:
        """Close monitor and cleanup resources"""
        if self._monitor_task and not self._monitor_task.done():
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        
        if self.enable_memory_tracking:
            tracemalloc.stop()

class AdvancedAsyncBatchProcessor:
    """Advanced batch processor with adaptive batching and error handling"""
    
    def __init__(
        self, 
        batch_size: int = 10, 
        max_concurrent: int = 5,
        adaptive_batching: bool = True,
        error_retry_attempts: int = 3,
        error_retry_delay: float = 1.0
    ):
        
    """__init__ function."""
self.batch_size = batch_size
        self.max_concurrent = max_concurrent
        self.adaptive_batching = adaptive_batching
        self.error_retry_attempts = error_retry_attempts
        self.error_retry_delay = error_retry_delay
        
        # Performance tracking
        self._batch_times: deque = deque(maxlen=100)
        self._error_counts: Dict[str, int] = defaultdict(int)
        self._success_counts: Dict[str, int] = defaultdict(int)
        
        # Adaptive batching
        self._current_batch_size = batch_size
        self._min_batch_size = max(1, batch_size // 4)
        self._max_batch_size = batch_size * 4
        
        # Semaphore for concurrency control
        self._semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_batch(
        self, 
        items: List[Any], 
        processor_func: Callable,
        batch_name: Optional[str] = None
    ) -> List[Any]:
        """Process items in batches with advanced features"""
        if not items:
            return []
        
        batch_name = batch_name or "default"
        start_time = time.time()
        
        try:
            # Split items into batches
            batches = [
                items[i:i + self._current_batch_size]
                for i in range(0, len(items), self._current_batch_size)
            ]
            
            # Process batches concurrently
            tasks = []
            for batch in batches:
                task = asyncio.create_task(
                    self._process_single_batch(batch, processor_func, batch_name)
                )
                tasks.append(task)
            
            # Wait for all batches to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Flatten results and handle exceptions
            processed_items = []
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Batch processing error: {result}")
                    self._error_counts[batch_name] += 1
                else:
                    processed_items.extend(result)
                    self._success_counts[batch_name] += 1
            
            # Update adaptive batching
            if self.adaptive_batching:
                await self._update_batch_size(batch_name, time.time() - start_time)
            
            return processed_items
            
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            self._error_counts[batch_name] += 1
            raise
    
    async def _process_single_batch(
        self, 
        batch: List[Any], 
        processor_func: Callable,
        batch_name: str
    ) -> List[Any]:
        """Process a single batch with retry logic"""
        async with self._semaphore:
            for attempt in range(self.error_retry_attempts + 1):
                try:
                    if asyncio.iscoroutinefunction(processor_func):
                        results = await processor_func(batch)
                    else:
                        # Run sync function in thread pool
                        loop = asyncio.get_event_loop()
                        results = await loop.run_in_executor(None, processor_func, batch)
                    
                    return results if isinstance(results, list) else list(results)
                    
                except Exception as e:
                    if attempt == self.error_retry_attempts:
                        logger.error(f"Batch processing failed after {self.error_retry_attempts} attempts: {e}")
                        raise
                    
                    logger.warning(f"Batch processing attempt {attempt + 1} failed: {e}")
                    await asyncio.sleep(self.error_retry_delay * (attempt + 1))
    
    async def _update_batch_size(self, batch_name: str, processing_time: float) -> None:
        """Update batch size based on performance"""
        self._batch_times.append(processing_time)
        
        if len(self._batch_times) < 10:
            return
        
        # Calculate average processing time
        avg_time = statistics.mean(self._batch_times)
        
        # Adjust batch size based on performance
        if avg_time < 0.1:  # Very fast processing
            self._current_batch_size = min(
                self._max_batch_size,
                self._current_batch_size + 2
            )
        elif avg_time > 1.0:  # Slow processing
            self._current_batch_size = max(
                self._min_batch_size,
                self._current_batch_size - 1
            )
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get batch processing statistics"""
        return {
            'current_batch_size': self._current_batch_size,
            'min_batch_size': self._min_batch_size,
            'max_batch_size': self._max_batch_size,
            'adaptive_batching': self.adaptive_batching,
            'error_counts': dict(self._error_counts),
            'success_counts': dict(self._success_counts),
            'avg_batch_time': statistics.mean(self._batch_times) if self._batch_times else 0,
            'total_batches': len(self._batch_times)
        }

# Global instances
advanced_cache = AdvancedAsyncCache[str, Any](
    ttl_seconds=600,
    max_size=2000,
    strategy=CacheStrategy.LRU,
    memory_policy=MemoryPolicy.ADAPTIVE,
    enable_compression=True,
    enable_metrics=True
)

advanced_monitor = AdvancedPerformanceMonitor(
    enable_memory_tracking=True,
    enable_cpu_tracking=True
)

advanced_batch_processor = AdvancedAsyncBatchProcessor(
    batch_size=20,
    max_concurrent=10,
    adaptive_batching=True,
    error_retry_attempts=3,
    error_retry_delay=1.0
)

# Decorators
def advanced_async_timed(metric_name: Optional[str] = None):
    """Advanced async timing decorator with detailed metrics"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            operation_name = metric_name or f"{func.__module__}.{func.__name__}"
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                
                # Record success metric
                duration_ms = (time.time() - start_time) * 1000
                await advanced_monitor.record_metric(
                    operation_name,
                    duration_ms,
                    {
                        'status': 'success',
                        'args_count': len(args),
                        'kwargs_count': len(kwargs)
                    }
                )
                
                return result
                
            except Exception as e:
                # Record error metric
                duration_ms = (time.time() - start_time) * 1000
                await advanced_monitor.record_metric(
                    operation_name,
                    duration_ms,
                    {
                        'status': 'error',
                        'error_type': type(e).__name__,
                        'error_message': str(e)
                    }
                )
                raise
        
        return wrapper
    return decorator

def advanced_cached_async(
    ttl_seconds: int = 300,
    key_func: Optional[Callable] = None,
    cache_instance: Optional[AdvancedAsyncCache] = None
):
    """Advanced async caching decorator"""
    cache = cache_instance or advanced_cache
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                # Default key generation
                key_parts = [func.__name__]
                key_parts.extend(str(arg) for arg in args)
                key_parts.extend(f"{k}:{v}" for k, v in sorted(kwargs.items()))
                cache_key = hashlib.md5("|".join(key_parts).encode()).hexdigest()
            
            # Try to get from cache
            cached_result = await cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Cache result
            await cache.set(cache_key, result)
            
            return result
        
        return wrapper
    return decorator

# Utility functions
@advanced_async_timed("performance.get_stats")
async def get_advanced_performance_stats() -> Dict[str, Any]:
    """Get comprehensive performance statistics"""
    cache_stats = await advanced_cache.get_stats()
    monitor_metrics = await advanced_monitor.get_metrics()
    batch_stats = await advanced_batch_processor.get_stats()
    memory_snapshot = await advanced_monitor.get_memory_snapshot()
    
    return {
        'cache': cache_stats,
        'monitor': monitor_metrics,
        'batch_processor': batch_stats,
        'memory_snapshot': memory_snapshot,
        'system': {
            'memory_percent': psutil.virtual_memory().percent,
            'cpu_percent': psutil.cpu_percent(),
            'disk_usage_percent': psutil.disk_usage('/').percent
        }
    }

@advanced_async_timed("performance.clear_all")
async def clear_all_advanced_caches() -> None:
    """Clear all advanced caches"""
    await advanced_cache.clear()
    logger.info("All advanced caches cleared")

@advanced_async_timed("performance.cleanup")
async def perform_advanced_cleanup() -> None:
    """Perform advanced cleanup operations"""
    # Force garbage collection
    gc.collect()
    
    # Clear caches
    await clear_all_advanced_caches()
    
    # Log cleanup results
    memory_snapshot = await advanced_monitor.get_memory_snapshot()
    logger.info(f"Advanced cleanup completed. Memory usage: {memory_snapshot.get('total_memory_mb', 0):.2f} MB")

# Example usage
async def example_advanced_optimized_operations():
    """Example of advanced optimized operations"""
    
    # Example 1: Advanced caching
    @advanced_cached_async(ttl_seconds=600)
    @advanced_async_timed("data.fetch")
    async async def fetch_data(data_id: str) -> Dict[str, Any]:
        # Simulate data fetching
        await asyncio.sleep(0.1)
        return {"id": data_id, "data": f"data_{data_id}", "timestamp": time.time()}
    
    # Example 2: Advanced batch processing
    async def process_item(item: int) -> int:
        # Simulate item processing
        await asyncio.sleep(0.01)
        return item * 2
    
    # Example 3: Performance monitoring
    @advanced_async_timed("complex_operation")
    async def complex_operation(items: List[int]) -> List[int]:
        # Simulate complex operation
        await asyncio.sleep(0.5)
        return [item * 2 for item in items]
    
    # Execute examples
    print("Running advanced optimized operations...")
    
    # Test caching
    data1 = await fetch_data("test1")
    data2 = await fetch_data("test1")  # Should be cached
    print(f"Cached data: {data1 == data2}")
    
    # Test batch processing
    items = list(range(100))
    processed_items = await advanced_batch_processor.process_batch(
        items, process_item, "test_batch"
    )
    print(f"Processed {len(processed_items)} items")
    
    # Test complex operation
    result = await complex_operation(items[:10])
    print(f"Complex operation result: {len(result)} items")
    
    # Get performance stats
    stats = await get_advanced_performance_stats()
    print(f"Performance stats: {json.dumps(stats, indent=2)}")

match __name__:
    case "__main__":
    asyncio.run(example_advanced_optimized_operations()) 