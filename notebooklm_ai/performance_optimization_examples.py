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
import functools
import inspect
import json
import logging
import time
import tracemalloc
import weakref
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable, Set, Tuple, Type, TypeVar, Awaitable
from enum import Enum
import threading
from contextlib import contextmanager, asynccontextmanager
from collections import defaultdict, deque, OrderedDict
import uuid
import hashlib
import base64
from datetime import datetime, timedelta
import statistics
import gc
import sys
import os
import pickle
import gzip
import lzma
    import redis
    import psutil
    import cProfile
    import pstats
    import line_profiler
    import memory_profiler
from typing import Any, List, Dict, Optional
"""
Performance Optimization Examples
================================

This module provides comprehensive performance optimization techniques and
strategies for Python applications.

Features:
- Caching strategies (memory, Redis, file-based)
- Profiling and benchmarking tools
- Memory optimization techniques
- Async performance optimization
- Database query optimization
- Algorithm optimization
- Resource management
- Performance monitoring
- Bottleneck detection
- Optimization recommendations

Author: AI Assistant
License: MIT
"""


try:
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    PROFILE_AVAILABLE = True
except ImportError:
    PROFILE_AVAILABLE = False

try:
    LINE_PROFILER_AVAILABLE = True
except ImportError:
    LINE_PROFILER_AVAILABLE = False

try:
    MEMORY_PROFILER_AVAILABLE = True
except ImportError:
    MEMORY_PROFILER_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

T = TypeVar('T')
F = TypeVar('F', bound=Callable[..., Any])


class CacheStrategy(Enum):
    """Cache strategies."""
    MEMORY = "memory"
    REDIS = "redis"
    FILE = "file"
    HYBRID = "hybrid"


class OptimizationLevel(Enum):
    """Optimization levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"


class ProfilerType(Enum):
    """Profiler types."""
    CPROFILE = "cprofile"
    LINE_PROFILER = "line_profiler"
    MEMORY_PROFILER = "memory_profiler"
    CUSTOM = "custom"


@dataclass
class PerformanceMetrics:
    """Performance metrics."""
    function_name: str
    execution_time: float
    memory_usage: Optional[float] = None
    cpu_usage: Optional[float] = None
    call_count: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    input_size: Optional[int] = None
    output_size: Optional[int] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class CacheEntry:
    """Cache entry."""
    key: str
    value: Any
    timestamp: datetime = field(default_factory=datetime.now)
    ttl: Optional[float] = None
    access_count: int = 0
    size: Optional[int] = None


@dataclass
class OptimizationConfig:
    """Optimization configuration."""
    enable_caching: bool = True
    enable_profiling: bool = True
    enable_memory_optimization: bool = True
    enable_async_optimization: bool = True
    cache_strategy: CacheStrategy = CacheStrategy.MEMORY
    cache_ttl: float = 3600.0  # 1 hour
    max_cache_size: int = 1000
    memory_threshold: float = 0.8  # 80% of available memory
    cpu_threshold: float = 0.9  # 90% of available CPU
    profiling_enabled: bool = False
    optimization_level: OptimizationLevel = OptimizationLevel.MEDIUM


@dataclass
class ProfilingResult:
    """Profiling result."""
    function_name: str
    total_time: float
    call_count: int
    average_time: float
    min_time: float
    max_time: float
    memory_usage: Optional[float] = None
    cpu_usage: Optional[float] = None
    bottlenecks: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


class PerformanceOptimizer:
    """Main performance optimizer."""
    
    def __init__(self, config: OptimizationConfig):
        """Initialize performance optimizer."""
        self.config = config
        self.cache_manager = CacheManager(config)
        self.profiler = Profiler(config)
        self.memory_optimizer = MemoryOptimizer(config)
        self.async_optimizer = AsyncOptimizer(config)
        self.metrics: Dict[str, PerformanceMetrics] = {}
        self._lock = threading.Lock()
    
    def optimize_function(self, func: F) -> F:
        """Apply optimizations to a function."""
        if not self.config.enable_caching and not self.config.enable_profiling:
            return func
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Get function info
            function_name = func.__name__
            
            # Check cache first
            if self.config.enable_caching:
                cache_key = self._generate_cache_key(function_name, args, kwargs)
                cached_result = self.cache_manager.get(cache_key)
                if cached_result is not None:
                    self._record_metrics(function_name, 0.0, cache_hits=1)
                    return cached_result
            
            # Profile execution
            start_time = time.time()
            start_memory = self._get_memory_usage()
            
            try:
                # Execute function
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                end_memory = self._get_memory_usage()
                memory_usage = end_memory - start_memory if end_memory and start_memory else None
                
                # Cache result
                if self.config.enable_caching:
                    cache_key = self._generate_cache_key(function_name, args, kwargs)
                    self.cache_manager.set(cache_key, result)
                
                # Record metrics
                self._record_metrics(function_name, execution_time, memory_usage=memory_usage)
                
                return result
            
            except Exception as e:
                execution_time = time.time() - start_time
                self._record_metrics(function_name, execution_time, error=True)
                raise
        
        return wrapper
    
    def _generate_cache_key(self, function_name: str, args: tuple, kwargs: dict) -> str:
        """Generate cache key for function call."""
        # Create a hash of the function signature and arguments
        key_data = {
            'function': function_name,
            'args': args,
            'kwargs': kwargs
        }
        
        key_string = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _get_memory_usage(self) -> Optional[float]:
        """Get current memory usage."""
        if not PSUTIL_AVAILABLE:
            return None
        
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # MB
        except Exception:
            return None
    
    def _record_metrics(self, function_name: str, execution_time: float, 
                       memory_usage: Optional[float] = None, cache_hits: int = 0,
                       cache_misses: int = 0, error: bool = False):
        """Record performance metrics."""
        with self._lock:
            if function_name not in self.metrics:
                self.metrics[function_name] = PerformanceMetrics(
                    function_name=function_name,
                    execution_time=execution_time,
                    memory_usage=memory_usage,
                    call_count=1,
                    cache_hits=cache_hits,
                    cache_misses=cache_misses
                )
            else:
                metrics = self.metrics[function_name]
                metrics.execution_time += execution_time
                metrics.call_count += 1
                metrics.cache_hits += cache_hits
                metrics.cache_misses += cache_misses
                
                if memory_usage:
                    if metrics.memory_usage:
                        metrics.memory_usage = (metrics.memory_usage + memory_usage) / 2
                    else:
                        metrics.memory_usage = memory_usage
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Get comprehensive optimization report."""
        with self._lock:
            report = {
                'metrics': {},
                'cache_stats': self.cache_manager.get_stats(),
                'memory_stats': self.memory_optimizer.get_stats(),
                'recommendations': self._generate_recommendations()
            }
            
            # Calculate metrics
            for function_name, metrics in self.metrics.items():
                avg_time = metrics.execution_time / metrics.call_count if metrics.call_count > 0 else 0
                cache_hit_rate = metrics.cache_hits / (metrics.cache_hits + metrics.cache_misses) if (metrics.cache_hits + metrics.cache_misses) > 0 else 0
                
                report['metrics'][function_name] = {
                    'total_time': metrics.execution_time,
                    'call_count': metrics.call_count,
                    'average_time': avg_time,
                    'memory_usage': metrics.memory_usage,
                    'cache_hit_rate': cache_hit_rate,
                    'cache_hits': metrics.cache_hits,
                    'cache_misses': metrics.cache_misses
                }
            
            return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []
        
        # Analyze metrics
        for function_name, metrics in self.metrics.items():
            avg_time = metrics.execution_time / metrics.call_count if metrics.call_count > 0 else 0
            
            # Slow function recommendations
            if avg_time > 1.0:
                recommendations.append(f"Function '{function_name}' is slow ({avg_time:.2f}s avg). Consider optimization.")
            
            # Memory usage recommendations
            if metrics.memory_usage and metrics.memory_usage > 100:  # 100MB
                recommendations.append(f"Function '{function_name}' uses high memory ({metrics.memory_usage:.1f}MB). Consider memory optimization.")
            
            # Cache recommendations
            cache_hit_rate = metrics.cache_hits / (metrics.cache_hits + metrics.cache_misses) if (metrics.cache_hits + metrics.cache_misses) > 0 else 0
            if cache_hit_rate < 0.5 and metrics.call_count > 10:
                recommendations.append(f"Function '{function_name}' has low cache hit rate ({cache_hit_rate:.1%}). Consider cache key optimization.")
        
        return recommendations


class CacheManager:
    """Cache management system."""
    
    def __init__(self, config: OptimizationConfig):
        """Initialize cache manager."""
        self.config = config
        self.memory_cache: OrderedDict = OrderedDict()
        self.redis_client = None
        self.cache_dir = Path("cache")
        self._lock = threading.Lock()
        
        # Setup cache backend
        if config.cache_strategy == CacheStrategy.REDIS and REDIS_AVAILABLE:
            self._setup_redis_cache()
        elif config.cache_strategy == CacheStrategy.FILE:
            self._setup_file_cache()
        
        # Create cache directory
        if config.cache_strategy in (CacheStrategy.FILE, CacheStrategy.HYBRID):
            self.cache_dir.mkdir(exist_ok=True)
    
    def _setup_redis_cache(self) -> Any:
        """Setup Redis cache."""
        try:
            self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
            self.redis_client.ping()  # Test connection
            logger.info("Redis cache initialized successfully")
        except Exception as e:
            logger.warning(f"Redis cache initialization failed: {e}")
            self.redis_client = None
    
    def _setup_file_cache(self) -> Any:
        """Setup file cache."""
        self.cache_dir.mkdir(exist_ok=True)
        logger.info(f"File cache initialized at {self.cache_dir}")
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if not self.config.enable_caching:
            return None
        
        # Try memory cache first
        with self._lock:
            if key in self.memory_cache:
                entry = self.memory_cache[key]
                if not self._is_expired(entry):
                    entry.access_count += 1
                    # Move to end (LRU)
                    self.memory_cache.move_to_end(key)
                    return entry.value
                else:
                    del self.memory_cache[key]
        
        # Try Redis cache
        if self.redis_client:
            try:
                value = self.redis_client.get(key)
                if value:
                    return pickle.loads(value)
            except Exception as e:
                logger.warning(f"Redis cache get failed: {e}")
        
        # Try file cache
        if self.config.cache_strategy in (CacheStrategy.FILE, CacheStrategy.HYBRID):
            file_path = self.cache_dir / f"{key}.cache"
            if file_path.exists():
                try:
                    with open(file_path, 'rb') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                        entry = pickle.load(f)
                    if not self._is_expired(entry):
                        return entry.value
                    else:
                        file_path.unlink()  # Remove expired cache
                except Exception as e:
                    logger.warning(f"File cache get failed: {e}")
        
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[float] = None):
        """Set value in cache."""
        if not self.config.enable_caching:
            return
        
        ttl = ttl or self.config.cache_ttl
        entry = CacheEntry(
            key=key,
            value=value,
            ttl=ttl,
            size=self._estimate_size(value)
        )
        
        # Set in memory cache
        with self._lock:
            if key in self.memory_cache:
                del self.memory_cache[key]
            
            self.memory_cache[key] = entry
            
            # Enforce max cache size
            while len(self.memory_cache) > self.config.max_cache_size:
                self.memory_cache.popitem(last=False)
        
        # Set in Redis cache
        if self.redis_client:
            try:
                serialized_value = pickle.dumps(entry)
                self.redis_client.setex(key, int(ttl), serialized_value)
            except Exception as e:
                logger.warning(f"Redis cache set failed: {e}")
        
        # Set in file cache
        if self.config.cache_strategy in (CacheStrategy.FILE, CacheStrategy.HYBRID):
            file_path = self.cache_dir / f"{key}.cache"
            try:
                with open(file_path, 'wb') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                    pickle.dump(entry, f)
            except Exception as e:
                logger.warning(f"File cache set failed: {e}")
    
    def _is_expired(self, entry: CacheEntry) -> bool:
        """Check if cache entry is expired."""
        if entry.ttl is None:
            return False
        
        age = (datetime.now() - entry.timestamp).total_seconds()
        return age > entry.ttl
    
    def _estimate_size(self, value: Any) -> Optional[int]:
        """Estimate size of value in bytes."""
        try:
            return len(pickle.dumps(value))
        except Exception:
            return None
    
    def clear(self) -> Any:
        """Clear all caches."""
        with self._lock:
            self.memory_cache.clear()
        
        if self.redis_client:
            try:
                self.redis_client.flushdb()
            except Exception as e:
                logger.warning(f"Redis cache clear failed: {e}")
        
        if self.config.cache_strategy in (CacheStrategy.FILE, CacheStrategy.HYBRID):
            try:
                for file_path in self.cache_dir.glob("*.cache"):
                    file_path.unlink()
            except Exception as e:
                logger.warning(f"File cache clear failed: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            stats = {
                'memory_cache_size': len(self.memory_cache),
                'memory_cache_keys': list(self.memory_cache.keys()),
                'strategy': self.config.cache_strategy.value
            }
            
            if self.redis_client:
                try:
                    stats['redis_keys'] = self.redis_client.dbsize()
                except Exception:
                    stats['redis_keys'] = 0
            
            if self.config.cache_strategy in (CacheStrategy.FILE, CacheStrategy.HYBRID):
                try:
                    file_count = len(list(self.cache_dir.glob("*.cache")))
                    stats['file_cache_count'] = file_count
                except Exception:
                    stats['file_cache_count'] = 0
            
            return stats


class Profiler:
    """Profiling system."""
    
    def __init__(self, config: OptimizationConfig):
        """Initialize profiler."""
        self.config = config
        self.profiling_results: Dict[str, ProfilingResult] = {}
        self._lock = threading.Lock()
    
    def profile_function(self, func: F) -> F:
        """Profile a function."""
        if not self.config.enable_profiling:
            return func
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            function_name = func.__name__
            start_time = time.time()
            start_memory = self._get_memory_usage()
            
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                end_memory = self._get_memory_usage()
                memory_usage = end_memory - start_memory if end_memory and start_memory else None
                
                self._record_profiling_result(function_name, execution_time, memory_usage)
                return result
            
            except Exception as e:
                execution_time = time.time() - start_time
                self._record_profiling_result(function_name, execution_time, error=True)
                raise
        
        return wrapper
    
    def _record_profiling_result(self, function_name: str, execution_time: float,
                               memory_usage: Optional[float] = None, error: bool = False):
        """Record profiling result."""
        with self._lock:
            if function_name not in self.profiling_results:
                self.profiling_results[function_name] = ProfilingResult(
                    function_name=function_name,
                    total_time=execution_time,
                    call_count=1,
                    average_time=execution_time,
                    min_time=execution_time,
                    max_time=execution_time,
                    memory_usage=memory_usage
                )
            else:
                result = self.profiling_results[function_name]
                result.total_time += execution_time
                result.call_count += 1
                result.average_time = result.total_time / result.call_count
                result.min_time = min(result.min_time, execution_time)
                result.max_time = max(result.max_time, execution_time)
                
                if memory_usage:
                    if result.memory_usage:
                        result.memory_usage = (result.memory_usage + memory_usage) / 2
                    else:
                        result.memory_usage = memory_usage
    
    def _get_memory_usage(self) -> Optional[float]:
        """Get current memory usage."""
        if not PSUTIL_AVAILABLE:
            return None
        
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # MB
        except Exception:
            return None
    
    def get_profiling_report(self) -> Dict[str, ProfilingResult]:
        """Get profiling report."""
        with self._lock:
            return dict(self.profiling_results)


class MemoryOptimizer:
    """Memory optimization system."""
    
    def __init__(self, config: OptimizationConfig):
        """Initialize memory optimizer."""
        self.config = config
        self.memory_usage_history: deque = deque(maxlen=1000)
        self._lock = threading.Lock()
    
    def optimize_memory(self) -> Any:
        """Perform memory optimization."""
        if not self.config.enable_memory_optimization:
            return
        
        current_memory = self._get_memory_usage()
        if current_memory:
            with self._lock:
                self.memory_usage_history.append({
                    'timestamp': datetime.now(),
                    'memory_usage': current_memory
                })
        
        # Check memory threshold
        if self._is_memory_high():
            self._perform_memory_cleanup()
    
    def _get_memory_usage(self) -> Optional[float]:
        """Get current memory usage."""
        if not PSUTIL_AVAILABLE:
            return None
        
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # MB
        except Exception:
            return None
    
    def _is_memory_high(self) -> bool:
        """Check if memory usage is high."""
        if not PSUTIL_AVAILABLE:
            return False
        
        try:
            memory = psutil.virtual_memory()
            return memory.percent > (self.config.memory_threshold * 100)
        except Exception:
            return False
    
    def _perform_memory_cleanup(self) -> Any:
        """Perform memory cleanup."""
        logger.info("Performing memory cleanup")
        
        # Force garbage collection
        collected = gc.collect()
        logger.info(f"Garbage collection freed {collected} objects")
        
        # Clear memory cache if available
        # This would be called from the cache manager
        
        # Clear weak references
        weakref._remove_dead_weakrefs()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        with self._lock:
            if not self.memory_usage_history:
                return {}
            
            memory_values = [entry['memory_usage'] for entry in self.memory_usage_history]
            
            return {
                'current_memory': memory_values[-1] if memory_values else None,
                'average_memory': statistics.mean(memory_values),
                'max_memory': max(memory_values),
                'min_memory': min(memory_values),
                'memory_trend': self._calculate_memory_trend()
            }
    
    def _calculate_memory_trend(self) -> str:
        """Calculate memory usage trend."""
        if len(self.memory_usage_history) < 10:
            return "insufficient_data"
        
        recent_values = [entry['memory_usage'] for entry in list(self.memory_usage_history)[-10:]]
        first_half = recent_values[:5]
        second_half = recent_values[5:]
        
        first_avg = statistics.mean(first_half)
        second_avg = statistics.mean(second_half)
        
        if second_avg > first_avg * 1.1:
            return "increasing"
        elif second_avg < first_avg * 0.9:
            return "decreasing"
        else:
            return "stable"


class AsyncOptimizer:
    """Async performance optimization."""
    
    def __init__(self, config: OptimizationConfig):
        """Initialize async optimizer."""
        self.config = config
        self.task_pool = asyncio.Queue(maxsize=100)
        self.active_tasks: Set[asyncio.Task] = set()
        self._lock = threading.Lock()
    
    async def optimize_async_function(self, func: Callable) -> Callable:
        """Optimize async function."""
        if not self.config.enable_async_optimization:
            return func
        
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            # Check if we should use task pooling
            if self._should_use_task_pool():
                return await self._execute_with_task_pool(func, args, kwargs)
            else:
                return await func(*args, **kwargs)
        
        return wrapper
    
    def _should_use_task_pool(self) -> bool:
        """Check if task pooling should be used."""
        return len(self.active_tasks) < 50  # Limit concurrent tasks
    
    async def _execute_with_task_pool(self, func: Callable, args: tuple, kwargs: dict):
        """Execute function with task pooling."""
        task = asyncio.create_task(func(*args, **kwargs))
        
        with self._lock:
            self.active_tasks.add(task)
        
        try:
            result = await task
            return result
        finally:
            with self._lock:
                self.active_tasks.discard(task)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get async optimization stats."""
        with self._lock:
            return {
                'active_tasks': len(self.active_tasks),
                'task_pool_size': self.task_pool.qsize(),
                'max_task_pool_size': self.task_pool.maxsize
            }


# Decorators
def optimize_performance(optimizer: PerformanceOptimizer):
    """Decorator to optimize function performance."""
    def decorator(func: F) -> F:
        return optimizer.optimize_function(func)
    return decorator


def profile_function(profiler: Profiler):
    """Decorator to profile function."""
    def decorator(func: F) -> F:
        return profiler.profile_function(func)
    return decorator


def cache_result(ttl: Optional[float] = None, key_func: Optional[Callable] = None):
    """Decorator to cache function results."""
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = f"{func.__name__}:{hash(args)}:{hash(frozenset(kwargs.items()))}"
            
            # Check cache
            # This would integrate with the cache manager
            # For now, we'll use a simple in-memory cache
            if hasattr(wrapper, '_cache') and cache_key in wrapper._cache:
                entry = wrapper._cache[cache_key]
                if time.time() - entry['timestamp'] < (ttl or 3600):
                    return entry['value']
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Cache result
            if not hasattr(wrapper, '_cache'):
                wrapper._cache = {}
            
            wrapper._cache[cache_key] = {
                'value': result,
                'timestamp': time.time()
            }
            
            return result
        
        return wrapper
    return decorator


def memory_efficient(max_memory_mb: float = 100):
    """Decorator to ensure memory efficiency."""
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            start_memory = _get_memory_usage()
            
            try:
                result = func(*args, **kwargs)
                
                end_memory = _get_memory_usage()
                if start_memory and end_memory:
                    memory_used = end_memory - start_memory
                    if memory_used > max_memory_mb:
                        logger.warning(f"Function {func.__name__} used {memory_used:.1f}MB memory")
                
                return result
            
            except Exception as e:
                logger.error(f"Function {func.__name__} failed: {e}")
                raise
        
        return wrapper
    return decorator


def _get_memory_usage() -> Optional[float]:
    """Get current memory usage."""
    if not PSUTIL_AVAILABLE:
        return None
    
    try:
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024  # MB
    except Exception:
        return None


# Context managers
@contextmanager
def performance_context(operation_name: str, optimizer: PerformanceOptimizer):
    """Context manager for performance monitoring."""
    start_time = time.time()
    start_memory = _get_memory_usage()
    
    try:
        yield
    finally:
        execution_time = time.time() - start_time
        end_memory = _get_memory_usage()
        memory_usage = end_memory - start_memory if end_memory and start_memory else None
        
        optimizer._record_metrics(operation_name, execution_time, memory_usage)


@asynccontextmanager
async def async_performance_context(operation_name: str, optimizer: PerformanceOptimizer):
    """Async context manager for performance monitoring."""
    start_time = time.time()
    start_memory = _get_memory_usage()
    
    try:
        yield
    finally:
        execution_time = time.time() - start_time
        end_memory = _get_memory_usage()
        memory_usage = end_memory - start_memory if end_memory and start_memory else None
        
        optimizer._record_metrics(operation_name, execution_time, memory_usage)


@contextmanager
def memory_tracking():
    """Context manager for memory tracking."""
    if PSUTIL_AVAILABLE:
        tracemalloc.start()
        start_snapshot = tracemalloc.take_snapshot()
    
    try:
        yield
    finally:
        if PSUTIL_AVAILABLE:
            end_snapshot = tracemalloc.take_snapshot()
            top_stats = end_snapshot.compare_to(start_snapshot, 'lineno')
            
            logger.info("Memory usage top changes:")
            for stat in top_stats[:3]:
                logger.info(stat)
            
            tracemalloc.stop()


# Example usage functions
def demonstrate_caching():
    """Demonstrate caching optimization."""
    config = OptimizationConfig(
        enable_caching=True,
        cache_strategy=CacheStrategy.MEMORY,
        cache_ttl=300.0  # 5 minutes
    )
    
    optimizer = PerformanceOptimizer(config)
    
    @optimize_performance(optimizer)
    def expensive_calculation(n: int) -> int:
        """Simulate expensive calculation."""
        time.sleep(0.1)  # Simulate work
        return n * n
    
    print("Testing caching optimization...")
    
    # First call - should be slow
    start_time = time.time()
    result1 = expensive_calculation(5)
    time1 = time.time() - start_time
    print(f"First call: {result1} (took {time1:.3f}s)")
    
    # Second call - should be fast (cached)
    start_time = time.time()
    result2 = expensive_calculation(5)
    time2 = time.time() - start_time
    print(f"Second call: {result2} (took {time2:.3f}s)")
    
    # Different input - should be slow again
    start_time = time.time()
    result3 = expensive_calculation(10)
    time3 = time.time() - start_time
    print(f"Different input: {result3} (took {time3:.3f}s)")
    
    # Get optimization report
    report = optimizer.get_optimization_report()
    print(f"\nOptimization Report:")
    print(json.dumps(report, indent=2, default=str))


def demonstrate_profiling():
    """Demonstrate profiling."""
    config = OptimizationConfig(enable_profiling=True)
    profiler = Profiler(config)
    
    @profile_function(profiler)
    def slow_function():
        """Function with intentional delays."""
        time.sleep(0.1)
        return "done"
    
    @profile_function(profiler)
    def fast_function():
        """Fast function."""
        return "fast"
    
    print("Testing profiling...")
    
    # Run functions multiple times
    for _ in range(5):
        slow_function()
        fast_function()
    
    # Get profiling report
    report = profiler.get_profiling_report()
    print(f"\nProfiling Report:")
    for function_name, result in report.items():
        print(f"{function_name}:")
        print(f"  Total time: {result.total_time:.3f}s")
        print(f"  Call count: {result.call_count}")
        print(f"  Average time: {result.average_time:.3f}s")
        print(f"  Min time: {result.min_time:.3f}s")
        print(f"  Max time: {result.max_time:.3f}s")


def demonstrate_memory_optimization():
    """Demonstrate memory optimization."""
    config = OptimizationConfig(enable_memory_optimization=True)
    memory_optimizer = MemoryOptimizer(config)
    
    @memory_efficient(max_memory_mb=50)
    def memory_intensive_function(size: int):
        """Function that uses significant memory."""
        # Create large data structure
        data = [i for i in range(size)]
        return sum(data)
    
    print("Testing memory optimization...")
    
    # Track memory usage
    with memory_tracking():
        result = memory_intensive_function(1000000)
        print(f"Result: {result}")
    
    # Get memory stats
    stats = memory_optimizer.get_stats()
    print(f"\nMemory Stats:")
    print(json.dumps(stats, indent=2, default=str))


def demonstrate_async_optimization():
    """Demonstrate async optimization."""
    config = OptimizationConfig(enable_async_optimization=True)
    async_optimizer = AsyncOptimizer(config)
    
    @asyncio.coroutine
    async def async_operation(delay: float):
        """Simulate async operation."""
        await asyncio.sleep(delay)
        return f"completed after {delay}s"
    
    async def test_async_optimization():
        
    """test_async_optimization function."""
print("Testing async optimization...")
        
        # Run multiple async operations
        tasks = []
        for i in range(10):
            task = async_optimizer.optimize_async_function(async_operation)(0.1)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        print(f"Completed {len(results)} async operations")
        
        # Get async stats
        stats = async_optimizer.get_stats()
        print(f"\nAsync Stats:")
        print(json.dumps(stats, indent=2, default=str))
    
    asyncio.run(test_async_optimization())


def demonstrate_comprehensive_optimization():
    """Demonstrate comprehensive optimization."""
    config = OptimizationConfig(
        enable_caching=True,
        enable_profiling=True,
        enable_memory_optimization=True,
        enable_async_optimization=True,
        optimization_level=OptimizationLevel.HIGH
    )
    
    optimizer = PerformanceOptimizer(config)
    
    @optimize_performance(optimizer)
    @cache_result(ttl=60.0)
    @memory_efficient(max_memory_mb=100)
    def comprehensive_function(n: int, use_cache: bool = True):
        """Function with comprehensive optimization."""
        if use_cache:
            time.sleep(0.05)  # Simulate work
        else:
            time.sleep(0.1)  # Simulate more work
        
        # Create some data
        data = [i * i for i in range(n)]
        return sum(data)
    
    print("Testing comprehensive optimization...")
    
    # Test with caching
    with performance_context("cached_operation", optimizer):
        result1 = comprehensive_function(1000, use_cache=True)
        result2 = comprehensive_function(1000, use_cache=True)  # Should use cache
    
    # Test without caching
    with performance_context("non_cached_operation", optimizer):
        result3 = comprehensive_function(1000, use_cache=False)
    
    print(f"Results: {result1}, {result2}, {result3}")
    
    # Get comprehensive report
    report = optimizer.get_optimization_report()
    print(f"\nComprehensive Optimization Report:")
    print(json.dumps(report, indent=2, default=str))


def main():
    """Main function demonstrating performance optimization."""
    logger.info("Starting performance optimization examples")
    
    # Demonstrate caching
    try:
        demonstrate_caching()
    except Exception as e:
        logger.error(f"Caching demonstration failed: {e}")
    
    # Demonstrate profiling
    try:
        demonstrate_profiling()
    except Exception as e:
        logger.error(f"Profiling demonstration failed: {e}")
    
    # Demonstrate memory optimization
    try:
        demonstrate_memory_optimization()
    except Exception as e:
        logger.error(f"Memory optimization demonstration failed: {e}")
    
    # Demonstrate async optimization
    try:
        demonstrate_async_optimization()
    except Exception as e:
        logger.error(f"Async optimization demonstration failed: {e}")
    
    # Demonstrate comprehensive optimization
    try:
        demonstrate_comprehensive_optimization()
    except Exception as e:
        logger.error(f"Comprehensive optimization demonstration failed: {e}")
    
    logger.info("Performance optimization examples completed")


match __name__:
    case "__main__":
    main() 