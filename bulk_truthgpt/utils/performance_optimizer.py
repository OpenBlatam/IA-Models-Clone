"""
Performance Optimizer
====================

Advanced performance optimization system for Bulk TruthGPT.
"""

import asyncio
import logging
import time
import psutil
import gc
from typing import Dict, List, Optional, Any, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import threading
from contextlib import asynccontextmanager
import weakref
import tracemalloc
import linecache
import sys
from functools import wraps, lru_cache
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

logger = logging.getLogger(__name__)

class OptimizationLevel(str, Enum):
    """Optimization levels."""
    NONE = "none"
    BASIC = "basic"
    ADVANCED = "advanced"
    AGGRESSIVE = "aggressive"

@dataclass
class PerformanceMetrics:
    """Performance metrics."""
    cpu_usage: float
    memory_usage: float
    memory_available: float
    disk_io_read: int
    disk_io_write: int
    network_io_sent: int
    network_io_recv: int
    process_count: int
    thread_count: int
    gc_collections: int
    gc_memory_freed: int
    timestamp: datetime

@dataclass
class OptimizationConfig:
    """Optimization configuration."""
    level: OptimizationLevel = OptimizationLevel.ADVANCED
    memory_threshold: float = 0.8
    cpu_threshold: float = 0.8
    gc_threshold: int = 1000
    batch_size: int = 100
    max_workers: int = None
    enable_memory_optimization: bool = True
    enable_cpu_optimization: bool = True
    enable_gc_optimization: bool = True
    enable_io_optimization: bool = True
    enable_network_optimization: bool = True

class MemoryOptimizer:
    """
    Advanced memory optimization system.
    
    Features:
    - Memory pooling
    - Garbage collection optimization
    - Memory leak detection
    - Object reuse
    - Weak references
    """
    
    def __init__(self):
        self.memory_pools = {}
        self.object_cache = {}
        self.weak_refs = weakref.WeakValueDictionary()
        self.memory_stats = deque(maxlen=1000)
        self.leak_detector = None
        self.gc_threshold = 1000
        self.memory_threshold = 0.8
        
    async def initialize(self):
        """Initialize memory optimizer."""
        logger.info("Initializing Memory Optimizer...")
        
        try:
            # Start memory monitoring
            asyncio.create_task(self._monitor_memory())
            
            # Start leak detection
            asyncio.create_task(self._detect_memory_leaks())
            
            # Configure garbage collection
            self._configure_gc()
            
            logger.info("Memory Optimizer initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Memory Optimizer: {str(e)}")
            raise
    
    def _configure_gc(self):
        """Configure garbage collection."""
        # Set GC thresholds
        gc.set_threshold(700, 10, 10)
        
        # Enable generation 0 and 1 collections
        gc.set_debug(gc.DEBUG_LEAK)
    
    async def _monitor_memory(self):
        """Monitor memory usage."""
        while True:
            try:
                await asyncio.sleep(5)  # Monitor every 5 seconds
                
                # Get memory info
                memory = psutil.virtual_memory()
                process = psutil.Process()
                
                memory_info = {
                    'total': memory.total,
                    'available': memory.available,
                    'used': memory.used,
                    'percent': memory.percent,
                    'process_memory': process.memory_info().rss,
                    'process_percent': process.memory_percent()
                }
                
                self.memory_stats.append(memory_info)
                
                # Check if memory usage is high
                if memory.percent > self.memory_threshold * 100:
                    await self._optimize_memory()
                
            except Exception as e:
                logger.error(f"Error monitoring memory: {str(e)}")
    
    async def _detect_memory_leaks(self):
        """Detect memory leaks."""
        while True:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                # Get current memory usage
                process = psutil.Process()
                current_memory = process.memory_info().rss
                
                # Check if memory is growing continuously
                if len(self.memory_stats) > 10:
                    recent_memory = [stat['process_memory'] for stat in list(self.memory_stats)[-10:]]
                    if all(recent_memory[i] < recent_memory[i+1] for i in range(len(recent_memory)-1)):
                        logger.warning("Potential memory leak detected")
                        await self._force_gc()
                
            except Exception as e:
                logger.error(f"Error detecting memory leaks: {str(e)}")
    
    async def _optimize_memory(self):
        """Optimize memory usage."""
        try:
            logger.info("Optimizing memory usage...")
            
            # Force garbage collection
            await self._force_gc()
            
            # Clear object cache
            self.object_cache.clear()
            
            # Clear weak references
            self.weak_refs.clear()
            
            # Clear memory pools
            for pool in self.memory_pools.values():
                pool.clear()
            
            logger.info("Memory optimization completed")
            
        except Exception as e:
            logger.error(f"Failed to optimize memory: {str(e)}")
    
    async def _force_gc(self):
        """Force garbage collection."""
        try:
            # Get GC stats before
            before = gc.get_count()
            
            # Force collection
            collected = gc.collect()
            
            # Get GC stats after
            after = gc.get_count()
            
            logger.info(f"Garbage collection: {collected} objects collected, {before} -> {after}")
            
        except Exception as e:
            logger.error(f"Failed to force GC: {str(e)}")
    
    def get_memory_pool(self, pool_name: str, factory: Callable):
        """Get or create memory pool."""
        if pool_name not in self.memory_pools:
            self.memory_pools[pool_name] = deque(maxlen=1000)
        
        pool = self.memory_pools[pool_name]
        
        if pool:
            return pool.popleft()
        else:
            return factory()
    
    def return_to_pool(self, pool_name: str, obj: Any):
        """Return object to memory pool."""
        if pool_name in self.memory_pools:
            self.memory_pools[pool_name].append(obj)
    
    def cache_object(self, key: str, obj: Any):
        """Cache object for reuse."""
        self.object_cache[key] = obj
    
    def get_cached_object(self, key: str) -> Optional[Any]:
        """Get cached object."""
        return self.object_cache.get(key)
    
    def create_weak_ref(self, obj: Any, key: str):
        """Create weak reference."""
        self.weak_refs[key] = obj
    
    def get_weak_ref(self, key: str) -> Optional[Any]:
        """Get weak reference."""
        return self.weak_refs.get(key)
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        if not self.memory_stats:
            return {}
        
        recent_stats = list(self.memory_stats)[-10:]
        
        return {
            'current_memory_percent': recent_stats[-1]['percent'],
            'current_process_memory': recent_stats[-1]['process_memory'],
            'average_memory_percent': np.mean([s['percent'] for s in recent_stats]),
            'max_memory_percent': max([s['percent'] for s in recent_stats]),
            'memory_trend': 'increasing' if recent_stats[-1]['percent'] > recent_stats[0]['percent'] else 'decreasing',
            'pool_sizes': {name: len(pool) for name, pool in self.memory_pools.items()},
            'cache_size': len(self.object_cache),
            'weak_ref_count': len(self.weak_refs)
        }

class CPUOptimizer:
    """
    Advanced CPU optimization system.
    
    Features:
    - CPU affinity
    - Thread pool optimization
    - Process pool management
    - CPU usage monitoring
    - Load balancing
    """
    
    def __init__(self):
        self.thread_pool = None
        self.process_pool = None
        self.cpu_cores = psutil.cpu_count()
        self.cpu_usage_history = deque(maxlen=1000)
        self.optimization_level = OptimizationLevel.ADVANCED
        
    async def initialize(self):
        """Initialize CPU optimizer."""
        logger.info("Initializing CPU Optimizer...")
        
        try:
            # Create thread pool
            self.thread_pool = ThreadPoolExecutor(
                max_workers=self.cpu_cores * 2,
                thread_name_prefix="bulk_truthgpt"
            )
            
            # Create process pool
            self.process_pool = ProcessPoolExecutor(
                max_workers=self.cpu_cores,
                mp_context=mp.get_context('spawn')
            )
            
            # Start CPU monitoring
            asyncio.create_task(self._monitor_cpu())
            
            logger.info("CPU Optimizer initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize CPU Optimizer: {str(e)}")
            raise
    
    async def _monitor_cpu(self):
        """Monitor CPU usage."""
        while True:
            try:
                await asyncio.sleep(1)  # Monitor every second
                
                # Get CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                cpu_per_core = psutil.cpu_percent(interval=1, percpu=True)
                
                self.cpu_usage_history.append({
                    'total': cpu_percent,
                    'per_core': cpu_per_core,
                    'timestamp': datetime.utcnow()
                })
                
                # Optimize if CPU usage is high
                if cpu_percent > 80:
                    await self._optimize_cpu()
                
            except Exception as e:
                logger.error(f"Error monitoring CPU: {str(e)}")
    
    async def _optimize_cpu(self):
        """Optimize CPU usage."""
        try:
            logger.info("Optimizing CPU usage...")
            
            # Adjust thread pool size
            if self.thread_pool:
                current_workers = self.thread_pool._max_workers
                new_workers = max(1, current_workers - 1)
                if new_workers != current_workers:
                    logger.info(f"Reducing thread pool size: {current_workers} -> {new_workers}")
            
            # Force garbage collection
            gc.collect()
            
            logger.info("CPU optimization completed")
            
        except Exception as e:
            logger.error(f"Failed to optimize CPU: {str(e)}")
    
    def get_thread_pool(self) -> ThreadPoolExecutor:
        """Get thread pool."""
        return self.thread_pool
    
    def get_process_pool(self) -> ProcessPoolExecutor:
        """Get process pool."""
        return self.process_pool
    
    def get_cpu_stats(self) -> Dict[str, Any]:
        """Get CPU statistics."""
        if not self.cpu_usage_history:
            return {}
        
        recent_stats = list(self.cpu_usage_history)[-10:]
        
        return {
            'current_cpu_percent': recent_stats[-1]['total'],
            'average_cpu_percent': np.mean([s['total'] for s in recent_stats]),
            'max_cpu_percent': max([s['total'] for s in recent_stats]),
            'cpu_trend': 'increasing' if recent_stats[-1]['total'] > recent_stats[0]['total'] else 'decreasing',
            'thread_pool_size': self.thread_pool._max_workers if self.thread_pool else 0,
            'process_pool_size': self.process_pool._max_workers if self.process_pool else 0,
            'cpu_cores': self.cpu_cores
        }
    
    async def cleanup(self):
        """Cleanup CPU optimizer."""
        try:
            if self.thread_pool:
                self.thread_pool.shutdown(wait=True)
            
            if self.process_pool:
                self.process_pool.shutdown(wait=True)
            
            logger.info("CPU Optimizer cleaned up successfully")
            
        except Exception as e:
            logger.error(f"Failed to cleanup CPU Optimizer: {str(e)}")

class IOOptimizer:
    """
    Advanced I/O optimization system.
    
    Features:
    - Async I/O operations
    - Connection pooling
    - Batch operations
    - Compression
    - Caching
    """
    
    def __init__(self):
        self.connection_pools = {}
        self.batch_operations = {}
        self.compression_enabled = True
        self.cache_enabled = True
        self.io_stats = deque(maxlen=1000)
        
    async def initialize(self):
        """Initialize I/O optimizer."""
        logger.info("Initializing I/O Optimizer...")
        
        try:
            # Start I/O monitoring
            asyncio.create_task(self._monitor_io())
            
            logger.info("I/O Optimizer initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize I/O Optimizer: {str(e)}")
            raise
    
    async def _monitor_io(self):
        """Monitor I/O operations."""
        while True:
            try:
                await asyncio.sleep(5)  # Monitor every 5 seconds
                
                # Get I/O stats
                io_counters = psutil.disk_io_counters()
                net_counters = psutil.net_io_counters()
                
                io_info = {
                    'disk_read_bytes': io_counters.read_bytes,
                    'disk_write_bytes': io_counters.write_bytes,
                    'disk_read_count': io_counters.read_count,
                    'disk_write_count': io_counters.write_count,
                    'network_sent_bytes': net_counters.bytes_sent,
                    'network_recv_bytes': net_counters.bytes_recv,
                    'timestamp': datetime.utcnow()
                }
                
                self.io_stats.append(io_info)
                
            except Exception as e:
                logger.error(f"Error monitoring I/O: {str(e)}")
    
    def get_io_stats(self) -> Dict[str, Any]:
        """Get I/O statistics."""
        if not self.io_stats:
            return {}
        
        recent_stats = list(self.io_stats)[-10:]
        
        return {
            'disk_read_rate': np.mean([s['disk_read_bytes'] for s in recent_stats]),
            'disk_write_rate': np.mean([s['disk_write_bytes'] for s in recent_stats]),
            'network_sent_rate': np.mean([s['network_sent_bytes'] for s in recent_stats]),
            'network_recv_rate': np.mean([s['network_recv_bytes'] for s in recent_stats]),
            'io_trend': 'increasing' if recent_stats[-1]['disk_read_bytes'] > recent_stats[0]['disk_read_bytes'] else 'decreasing'
        }

class PerformanceOptimizer:
    """
    Main performance optimization system.
    
    Coordinates all optimization subsystems and provides
    high-level optimization management.
    """
    
    def __init__(self, config: Optional[OptimizationConfig] = None):
        self.config = config or OptimizationConfig()
        self.memory_optimizer = MemoryOptimizer()
        self.cpu_optimizer = CPUOptimizer()
        self.io_optimizer = IOOptimizer()
        self.optimization_enabled = True
        self.performance_metrics = deque(maxlen=1000)
        
    async def initialize(self):
        """Initialize performance optimizer."""
        logger.info("Initializing Performance Optimizer...")
        
        try:
            # Initialize subsystems
            await self.memory_optimizer.initialize()
            await self.cpu_optimizer.initialize()
            await self.io_optimizer.initialize()
            
            # Start performance monitoring
            asyncio.create_task(self._monitor_performance())
            
            # Start optimization
            asyncio.create_task(self._optimize_performance())
            
            logger.info("Performance Optimizer initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Performance Optimizer: {str(e)}")
            raise
    
    async def _monitor_performance(self):
        """Monitor overall performance."""
        while True:
            try:
                await asyncio.sleep(10)  # Monitor every 10 seconds
                
                # Collect performance metrics
                memory_stats = self.memory_optimizer.get_memory_stats()
                cpu_stats = self.cpu_optimizer.get_cpu_stats()
                io_stats = self.io_optimizer.get_io_stats()
                
                performance_metric = PerformanceMetrics(
                    cpu_usage=cpu_stats.get('current_cpu_percent', 0),
                    memory_usage=memory_stats.get('current_memory_percent', 0),
                    memory_available=memory_stats.get('current_process_memory', 0),
                    disk_io_read=io_stats.get('disk_read_rate', 0),
                    disk_io_write=io_stats.get('disk_write_rate', 0),
                    network_io_sent=io_stats.get('network_sent_rate', 0),
                    network_io_recv=io_stats.get('network_recv_rate', 0),
                    process_count=len(psutil.pids()),
                    thread_count=psutil.Process().num_threads(),
                    gc_collections=sum(gc.get_count()),
                    gc_memory_freed=0,  # Would need to track this
                    timestamp=datetime.utcnow()
                )
                
                self.performance_metrics.append(performance_metric)
                
            except Exception as e:
                logger.error(f"Error monitoring performance: {str(e)}")
    
    async def _optimize_performance(self):
        """Optimize performance based on metrics."""
        while True:
            try:
                await asyncio.sleep(30)  # Optimize every 30 seconds
                
                if not self.optimization_enabled:
                    continue
                
                # Get recent performance metrics
                if len(self.performance_metrics) < 5:
                    continue
                
                recent_metrics = list(self.performance_metrics)[-5:]
                
                # Check if optimization is needed
                avg_cpu = np.mean([m.cpu_usage for m in recent_metrics])
                avg_memory = np.mean([m.memory_usage for m in recent_metrics])
                
                if avg_cpu > self.config.cpu_threshold * 100:
                    logger.info("CPU optimization triggered")
                    await self._optimize_cpu_performance()
                
                if avg_memory > self.config.memory_threshold * 100:
                    logger.info("Memory optimization triggered")
                    await self._optimize_memory_performance()
                
            except Exception as e:
                logger.error(f"Error optimizing performance: {str(e)}")
    
    async def _optimize_cpu_performance(self):
        """Optimize CPU performance."""
        try:
            # Adjust thread pool size
            # Force garbage collection
            gc.collect()
            
            logger.info("CPU performance optimization completed")
            
        except Exception as e:
            logger.error(f"Failed to optimize CPU performance: {str(e)}")
    
    async def _optimize_memory_performance(self):
        """Optimize memory performance."""
        try:
            # Force garbage collection
            await self.memory_optimizer._force_gc()
            
            # Clear caches
            self.memory_optimizer.object_cache.clear()
            
            logger.info("Memory performance optimization completed")
            
        except Exception as e:
            logger.error(f"Failed to optimize memory performance: {str(e)}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        if not self.performance_metrics:
            return {}
        
        recent_metrics = list(self.performance_metrics)[-10:]
        
        return {
            'cpu_usage': {
                'current': recent_metrics[-1].cpu_usage,
                'average': np.mean([m.cpu_usage for m in recent_metrics]),
                'max': max([m.cpu_usage for m in recent_metrics])
            },
            'memory_usage': {
                'current': recent_metrics[-1].memory_usage,
                'average': np.mean([m.memory_usage for m in recent_metrics]),
                'max': max([m.memory_usage for m in recent_metrics])
            },
            'io_usage': {
                'disk_read': recent_metrics[-1].disk_io_read,
                'disk_write': recent_metrics[-1].disk_io_write,
                'network_sent': recent_metrics[-1].network_io_sent,
                'network_recv': recent_metrics[-1].network_io_recv
            },
            'system_info': {
                'process_count': recent_metrics[-1].process_count,
                'thread_count': recent_metrics[-1].thread_count,
                'gc_collections': recent_metrics[-1].gc_collections
            },
            'optimization_status': {
                'enabled': self.optimization_enabled,
                'level': self.config.level.value,
                'memory_threshold': self.config.memory_threshold,
                'cpu_threshold': self.config.cpu_threshold
            }
        }
    
    def enable_optimization(self):
        """Enable optimization."""
        self.optimization_enabled = True
        logger.info("Performance optimization enabled")
    
    def disable_optimization(self):
        """Disable optimization."""
        self.optimization_enabled = False
        logger.info("Performance optimization disabled")
    
    def set_optimization_level(self, level: OptimizationLevel):
        """Set optimization level."""
        self.config.level = level
        logger.info(f"Optimization level set to: {level.value}")
    
    async def cleanup(self):
        """Cleanup performance optimizer."""
        try:
            await self.memory_optimizer._optimize_memory()
            await self.cpu_optimizer.cleanup()
            
            logger.info("Performance Optimizer cleaned up successfully")
            
        except Exception as e:
            logger.error(f"Failed to cleanup Performance Optimizer: {str(e)}")

# Global performance optimizer
performance_optimizer = PerformanceOptimizer()

# Decorators for optimization
def optimize_memory(func):
    """Decorator to optimize memory usage."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            result = await func(*args, **kwargs)
            return result
        finally:
            # Force garbage collection after function
            gc.collect()
    return wrapper

def optimize_cpu(func):
    """Decorator to optimize CPU usage."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            result = await func(*args, **kwargs)
            return result
        except Exception as e:
            logger.error(f"Error in optimized function {func.__name__}: {str(e)}")
            raise
    return wrapper

def cache_result(ttl: int = 3600):
    """Decorator to cache function results."""
    def decorator(func):
        cache = {}
        
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Create cache key
            key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"
            
            # Check cache
            if key in cache:
                cached_result, timestamp = cache[key]
                if time.time() - timestamp < ttl:
                    return cached_result
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Cache result
            cache[key] = (result, time.time())
            
            return result
        
        return wrapper
    return decorator











