"""
Advanced optimization engine with performance tuning and resource management.
"""

import asyncio
import time
import psutil
import gc
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import wraps, lru_cache
import threading
from collections import deque
import weakref

from ..core.logging import get_logger
from ..core.config import get_settings

logger = get_logger(__name__)


@dataclass
class PerformanceProfile:
    """Performance profile for optimization."""
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    disk_io: float = 0.0
    network_io: float = 0.0
    response_time: float = 0.0
    throughput: float = 0.0
    error_rate: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class OptimizationConfig:
    """Configuration for optimization engine."""
    max_cpu_usage: float = 80.0
    max_memory_usage: float = 85.0
    max_response_time: float = 1.0
    min_throughput: float = 100.0
    max_error_rate: float = 5.0
    optimization_interval: float = 30.0
    auto_optimize: bool = True
    enable_gc_optimization: bool = True
    enable_memory_pooling: bool = True


class ResourceManager:
    """Advanced resource management with auto-scaling."""
    
    def __init__(self):
        self.settings = get_settings()
        self.config = OptimizationConfig()
        self.performance_history: deque = deque(maxlen=1000)
        self.resource_pools: Dict[str, Any] = {}
        self.optimization_lock = threading.Lock()
        self._running = False
        self._optimization_task: Optional[asyncio.Task] = None
        
        # Initialize resource pools
        self._init_resource_pools()
    
    def _init_resource_pools(self):
        """Initialize resource pools for optimization."""
        # Thread pool for CPU-bound tasks
        self.resource_pools["thread_pool"] = ThreadPoolExecutor(
            max_workers=self.settings.max_workers,
            thread_name_prefix="optimized_worker"
        )
        
        # Process pool for heavy computations
        self.resource_pools["process_pool"] = ProcessPoolExecutor(
            max_workers=min(4, psutil.cpu_count())
        )
        
        # Connection pool for database operations
        self.resource_pools["connection_pool"] = {}
        
        # Memory pool for object reuse
        self.resource_pools["memory_pool"] = {}
    
    async def start_optimization(self):
        """Start automatic optimization."""
        if self._running:
            return
        
        self._running = True
        self._optimization_task = asyncio.create_task(self._optimization_loop())
        logger.info("Resource optimization started")
    
    async def stop_optimization(self):
        """Stop automatic optimization."""
        self._running = False
        if self._optimization_task:
            self._optimization_task.cancel()
            try:
                await self._optimization_task
            except asyncio.CancelledError:
                pass
        
        # Cleanup resource pools
        for pool_name, pool in self.resource_pools.items():
            if hasattr(pool, 'shutdown'):
                pool.shutdown(wait=True)
        
        logger.info("Resource optimization stopped")
    
    async def _optimization_loop(self):
        """Main optimization loop."""
        while self._running:
            try:
                await self._collect_performance_metrics()
                await self._analyze_performance()
                await self._apply_optimizations()
                
                await asyncio.sleep(self.config.optimization_interval)
                
            except Exception as e:
                logger.error(f"Optimization loop error: {e}")
                await asyncio.sleep(5)
    
    async def _collect_performance_metrics(self):
        """Collect system performance metrics."""
        try:
            # CPU usage
            cpu_usage = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_usage = memory.percent
            
            # Disk I/O
            disk_io = psutil.disk_io_counters()
            disk_io_rate = disk_io.read_bytes + disk_io.write_bytes if disk_io else 0
            
            # Network I/O
            network_io = psutil.net_io_counters()
            network_io_rate = network_io.bytes_sent + network_io.bytes_recv if network_io else 0
            
            profile = PerformanceProfile(
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                disk_io=disk_io_rate,
                network_io=network_io_rate
            )
            
            with self.optimization_lock:
                self.performance_history.append(profile)
            
        except Exception as e:
            logger.error(f"Error collecting performance metrics: {e}")
    
    async def _analyze_performance(self):
        """Analyze performance and identify optimization opportunities."""
        if len(self.performance_history) < 10:
            return
        
        recent_profiles = list(self.performance_history)[-10:]
        
        # Calculate averages
        avg_cpu = sum(p.cpu_usage for p in recent_profiles) / len(recent_profiles)
        avg_memory = sum(p.memory_usage for p in recent_profiles) / len(recent_profiles)
        avg_response_time = sum(p.response_time for p in recent_profiles) / len(recent_profiles)
        avg_throughput = sum(p.throughput for p in recent_profiles) / len(recent_profiles)
        avg_error_rate = sum(p.error_rate for p in recent_profiles) / len(recent_profiles)
        
        # Identify optimization needs
        optimizations = []
        
        if avg_cpu > self.config.max_cpu_usage:
            optimizations.append("cpu_optimization")
        
        if avg_memory > self.config.max_memory_usage:
            optimizations.append("memory_optimization")
        
        if avg_response_time > self.config.max_response_time:
            optimizations.append("response_time_optimization")
        
        if avg_throughput < self.config.min_throughput:
            optimizations.append("throughput_optimization")
        
        if avg_error_rate > self.config.max_error_rate:
            optimizations.append("error_rate_optimization")
        
        # Store optimization recommendations
        self._pending_optimizations = optimizations
        
        if optimizations:
            logger.info(f"Performance analysis complete. Optimizations needed: {optimizations}")
    
    async def _apply_optimizations(self):
        """Apply identified optimizations."""
        if not hasattr(self, '_pending_optimizations'):
            return
        
        for optimization in self._pending_optimizations:
            try:
                if optimization == "cpu_optimization":
                    await self._optimize_cpu()
                elif optimization == "memory_optimization":
                    await self._optimize_memory()
                elif optimization == "response_time_optimization":
                    await self._optimize_response_time()
                elif optimization == "throughput_optimization":
                    await self._optimize_throughput()
                elif optimization == "error_rate_optimization":
                    await self._optimize_error_rate()
                
            except Exception as e:
                logger.error(f"Error applying optimization {optimization}: {e}")
        
        self._pending_optimizations = []
    
    async def _optimize_cpu(self):
        """Optimize CPU usage."""
        logger.info("Applying CPU optimizations")
        
        # Adjust thread pool size
        current_workers = self.resource_pools["thread_pool"]._max_workers
        if current_workers > 2:
            new_workers = max(2, current_workers - 1)
            self._resize_thread_pool(new_workers)
        
        # Force garbage collection
        if self.config.enable_gc_optimization:
            gc.collect()
    
    async def _optimize_memory(self):
        """Optimize memory usage."""
        logger.info("Applying memory optimizations")
        
        # Force garbage collection
        if self.config.enable_gc_optimization:
            gc.collect()
        
        # Clear memory pools
        if self.config.enable_memory_pooling:
            self._clear_memory_pools()
    
    async def _optimize_response_time(self):
        """Optimize response time."""
        logger.info("Applying response time optimizations")
        
        # Increase thread pool size for better concurrency
        current_workers = self.resource_pools["thread_pool"]._max_workers
        max_workers = min(psutil.cpu_count() * 2, 20)
        if current_workers < max_workers:
            new_workers = min(max_workers, current_workers + 2)
            self._resize_thread_pool(new_workers)
    
    async def _optimize_throughput(self):
        """Optimize throughput."""
        logger.info("Applying throughput optimizations")
        
        # Optimize connection pools
        await self._optimize_connection_pools()
        
        # Enable memory pooling
        if not self.config.enable_memory_pooling:
            self.config.enable_memory_pooling = True
    
    async def _optimize_error_rate(self):
        """Optimize error rate."""
        logger.info("Applying error rate optimizations")
        
        # Increase timeouts
        # Add retry mechanisms
        # Improve error handling
    
    def _resize_thread_pool(self, new_size: int):
        """Resize thread pool."""
        try:
            old_pool = self.resource_pools["thread_pool"]
            old_pool.shutdown(wait=False)
            
            new_pool = ThreadPoolExecutor(
                max_workers=new_size,
                thread_name_prefix="optimized_worker"
            )
            self.resource_pools["thread_pool"] = new_pool
            
            logger.info(f"Thread pool resized from {old_pool._max_workers} to {new_size}")
            
        except Exception as e:
            logger.error(f"Error resizing thread pool: {e}")
    
    def _clear_memory_pools(self):
        """Clear memory pools."""
        for pool_name, pool in self.resource_pools["memory_pool"].items():
            if isinstance(pool, list):
                pool.clear()
            elif isinstance(pool, dict):
                pool.clear()
    
    async def _optimize_connection_pools(self):
        """Optimize connection pools."""
        # Implementation would depend on specific database/connection types
        pass
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        if not self.performance_history:
            return {"status": "no_data"}
        
        recent_profiles = list(self.performance_history)[-10:]
        
        return {
            "cpu_usage": {
                "current": recent_profiles[-1].cpu_usage,
                "average": sum(p.cpu_usage for p in recent_profiles) / len(recent_profiles),
                "max": max(p.cpu_usage for p in recent_profiles)
            },
            "memory_usage": {
                "current": recent_profiles[-1].memory_usage,
                "average": sum(p.memory_usage for p in recent_profiles) / len(recent_profiles),
                "max": max(p.memory_usage for p in recent_profiles)
            },
            "optimization_status": {
                "auto_optimize": self.config.auto_optimize,
                "running": self._running,
                "pending_optimizations": getattr(self, '_pending_optimizations', [])
            }
        }


class PerformanceOptimizer:
    """Performance optimization decorators and utilities."""
    
    @staticmethod
    def optimize_cpu(func: Callable) -> Callable:
        """Optimize CPU-bound functions."""
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Use process pool for CPU-intensive tasks
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,  # Use default executor
                func, *args, **kwargs
            )
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    @staticmethod
    def optimize_memory(func: Callable) -> Callable:
        """Optimize memory usage."""
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                result = await func(*args, **kwargs)
                # Force garbage collection after memory-intensive operations
                gc.collect()
                return result
            except Exception as e:
                gc.collect()
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                result = func(*args, **kwargs)
                gc.collect()
                return result
            except Exception as e:
                gc.collect()
                raise
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    @staticmethod
    def optimize_io(func: Callable) -> Callable:
        """Optimize I/O operations."""
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Use thread pool for I/O operations
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,  # Use default executor
                func, *args, **kwargs
            )
        
        return async_wrapper
    
    @staticmethod
    def cache_result(ttl: int = 300, maxsize: int = 128):
        """Cache function results with TTL."""
        def decorator(func: Callable) -> Callable:
            cache = {}
            cache_times = {}
            
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                # Create cache key
                key = str(hash(str(args) + str(sorted(kwargs.items()))))
                current_time = time.time()
                
                # Check cache
                if key in cache and current_time - cache_times[key] < ttl:
                    return cache[key]
                
                # Execute function
                result = await func(*args, **kwargs)
                
                # Store in cache
                cache[key] = result
                cache_times[key] = current_time
                
                # Cleanup old entries
                if len(cache) > maxsize:
                    oldest_key = min(cache_times.keys(), key=lambda k: cache_times[k])
                    del cache[oldest_key]
                    del cache_times[oldest_key]
                
                return result
            
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                key = str(hash(str(args) + str(sorted(kwargs.items()))))
                current_time = time.time()
                
                if key in cache and current_time - cache_times[key] < ttl:
                    return cache[key]
                
                result = func(*args, **kwargs)
                cache[key] = result
                cache_times[key] = current_time
                
                if len(cache) > maxsize:
                    oldest_key = min(cache_times.keys(), key=lambda k: cache_times[k])
                    del cache[oldest_key]
                    del cache_times[oldest_key]
                
                return result
            
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper
        
        return decorator


class MemoryPool:
    """Memory pool for object reuse."""
    
    def __init__(self, object_type: type, initial_size: int = 100):
        self.object_type = object_type
        self.pool = deque()
        self.initial_size = initial_size
        self._initialize_pool()
    
    def _initialize_pool(self):
        """Initialize the object pool."""
        for _ in range(self.initial_size):
            obj = self.object_type()
            self.pool.append(obj)
    
    def get(self):
        """Get an object from the pool."""
        if self.pool:
            return self.pool.popleft()
        else:
            return self.object_type()
    
    def put(self, obj):
        """Return an object to the pool."""
        if len(self.pool) < self.initial_size * 2:
            self.pool.append(obj)
    
    def clear(self):
        """Clear the pool."""
        self.pool.clear()


class ConnectionPool:
    """Optimized connection pool."""
    
    def __init__(self, max_connections: int = 10):
        self.max_connections = max_connections
        self.connections = deque()
        self.active_connections = set()
        self.lock = threading.Lock()
    
    async def get_connection(self):
        """Get a connection from the pool."""
        with self.lock:
            if self.connections:
                conn = self.connections.popleft()
                self.active_connections.add(conn)
                return conn
            elif len(self.active_connections) < self.max_connections:
                # Create new connection
                conn = await self._create_connection()
                self.active_connections.add(conn)
                return conn
            else:
                # Wait for available connection
                while not self.connections:
                    await asyncio.sleep(0.01)
                conn = self.connections.popleft()
                self.active_connections.add(conn)
                return conn
    
    async def return_connection(self, conn):
        """Return a connection to the pool."""
        with self.lock:
            if conn in self.active_connections:
                self.active_connections.remove(conn)
                self.connections.append(conn)
    
    async def _create_connection(self):
        """Create a new connection."""
        # Implementation would depend on specific connection type
        pass


# Global instances
_resource_manager: Optional[ResourceManager] = None
_performance_optimizer = PerformanceOptimizer()


def get_resource_manager() -> ResourceManager:
    """Get global resource manager instance."""
    global _resource_manager
    if _resource_manager is None:
        _resource_manager = ResourceManager()
    return _resource_manager


def get_performance_optimizer() -> PerformanceOptimizer:
    """Get global performance optimizer instance."""
    return _performance_optimizer


# Optimization decorators
def optimize_cpu(func: Callable) -> Callable:
    """Optimize CPU-bound functions."""
    return _performance_optimizer.optimize_cpu(func)


def optimize_memory(func: Callable) -> Callable:
    """Optimize memory usage."""
    return _performance_optimizer.optimize_memory(func)


def optimize_io(func: Callable) -> Callable:
    """Optimize I/O operations."""
    return _performance_optimizer.optimize_io(func)


def cache_result(ttl: int = 300, maxsize: int = 128):
    """Cache function results with TTL."""
    return _performance_optimizer.cache_result(ttl, maxsize)


# Utility functions
async def start_optimization():
    """Start system optimization."""
    resource_manager = get_resource_manager()
    await resource_manager.start_optimization()


async def stop_optimization():
    """Stop system optimization."""
    resource_manager = get_resource_manager()
    await resource_manager.stop_optimization()


async def get_performance_summary() -> Dict[str, Any]:
    """Get performance summary."""
    resource_manager = get_resource_manager()
    return resource_manager.get_performance_summary()


async def force_optimization():
    """Force immediate optimization."""
    resource_manager = get_resource_manager()
    await resource_manager._collect_performance_metrics()
    await resource_manager._analyze_performance()
    await resource_manager._apply_optimizations()


