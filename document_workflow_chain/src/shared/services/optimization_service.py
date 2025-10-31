"""
Optimization Service
====================

Advanced optimization service for performance tuning and resource management.
"""

from __future__ import annotations
import asyncio
import logging
import psutil
import gc
import time
from typing import Dict, List, Optional, Any, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing
from functools import wraps
import weakref
import tracemalloc

from ..utils.helpers import DateTimeHelpers
from ..utils.decorators import log_execution, measure_performance


logger = logging.getLogger(__name__)


class OptimizationType(str, Enum):
    """Optimization type enumeration"""
    MEMORY = "memory"
    CPU = "cpu"
    NETWORK = "network"
    DISK = "disk"
    DATABASE = "database"
    CACHE = "cache"
    CONNECTION = "connection"
    QUERY = "query"


class OptimizationLevel(str, Enum):
    """Optimization level enumeration"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    MAXIMUM = "maximum"


@dataclass
class SystemMetrics:
    """System metrics representation"""
    cpu_percent: float
    memory_percent: float
    memory_available: int
    memory_used: int
    disk_usage_percent: float
    disk_free: int
    network_io: Dict[str, int]
    process_count: int
    thread_count: int
    timestamp: datetime = field(default_factory=DateTimeHelpers.now_utc)


@dataclass
class OptimizationConfig:
    """Optimization configuration"""
    enable_memory_optimization: bool = True
    enable_cpu_optimization: bool = True
    enable_network_optimization: bool = True
    enable_disk_optimization: bool = True
    memory_threshold: float = 80.0  # Percentage
    cpu_threshold: float = 80.0  # Percentage
    disk_threshold: float = 90.0  # Percentage
    optimization_interval: int = 60  # Seconds
    gc_threshold: int = 1000  # Objects
    connection_pool_size: int = 20
    max_workers: int = multiprocessing.cpu_count()
    enable_profiling: bool = False
    enable_memory_tracing: bool = False


@dataclass
class OptimizationResult:
    """Optimization result representation"""
    optimization_type: OptimizationType
    level: OptimizationLevel
    before_metrics: SystemMetrics
    after_metrics: SystemMetrics
    improvement_percent: float
    duration: float
    timestamp: datetime = field(default_factory=DateTimeHelpers.now_utc)


class OptimizationService:
    """Advanced optimization service for system performance tuning"""
    
    def __init__(self, config: Optional[OptimizationConfig] = None):
        self.config = config or OptimizationConfig()
        self._is_running = False
        self._optimization_task: Optional[asyncio.Task] = None
        self._metrics_history: List[SystemMetrics] = []
        self._optimization_results: List[OptimizationResult] = []
        self._thread_pool: Optional[ThreadPoolExecutor] = None
        self._process_pool: Optional[ProcessPoolExecutor] = None
        self._weak_refs: List[weakref.ref] = []
        self._memory_tracer = None
        self._profiler = None
        self._lock = asyncio.Lock()
    
    async def start(self):
        """Start the optimization service"""
        if self._is_running:
            return
        
        try:
            # Initialize thread and process pools
            self._thread_pool = ThreadPoolExecutor(max_workers=self.config.max_workers)
            self._process_pool = ProcessPoolExecutor(max_workers=self.config.max_workers)
            
            # Initialize memory tracing
            if self.config.enable_memory_tracing:
                tracemalloc.start()
                self._memory_tracer = tracemalloc
            
            # Initialize profiler
            if self.config.enable_profiling:
                import cProfile
                self._profiler = cProfile.Profile()
            
            self._is_running = True
            
            # Start optimization task
            self._optimization_task = asyncio.create_task(self._optimization_worker())
            
            logger.info("Optimization service started successfully")
        
        except Exception as e:
            logger.error(f"Failed to start optimization service: {e}")
            raise
    
    async def stop(self):
        """Stop the optimization service"""
        if not self._is_running:
            return
        
        self._is_running = False
        
        # Stop optimization task
        if self._optimization_task:
            self._optimization_task.cancel()
            try:
                await self._optimization_task
            except asyncio.CancelledError:
                pass
        
        # Shutdown thread and process pools
        if self._thread_pool:
            self._thread_pool.shutdown(wait=True)
        
        if self._process_pool:
            self._process_pool.shutdown(wait=True)
        
        # Stop memory tracing
        if self._memory_tracer:
            tracemalloc.stop()
        
        # Stop profiler
        if self._profiler:
            self._profiler.disable()
        
        logger.info("Optimization service stopped")
    
    async def _optimization_worker(self):
        """Optimization worker task"""
        while self._is_running:
            try:
                # Collect system metrics
                metrics = await self._collect_system_metrics()
                self._metrics_history.append(metrics)
                
                # Keep only last 100 metrics
                if len(self._metrics_history) > 100:
                    self._metrics_history = self._metrics_history[-50:]
                
                # Check if optimization is needed
                if await self._needs_optimization(metrics):
                    await self._perform_optimization(metrics)
                
                await asyncio.sleep(self.config.optimization_interval)
            
            except Exception as e:
                logger.error(f"Optimization worker error: {e}")
                await asyncio.sleep(10)
    
    async def _collect_system_metrics(self) -> SystemMetrics:
        """Collect system metrics"""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_available = memory.available
            memory_used = memory.used
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_usage_percent = (disk.used / disk.total) * 100
            disk_free = disk.free
            
            # Network metrics
            network_io = psutil.net_io_counters()._asdict()
            
            # Process metrics
            process_count = len(psutil.pids())
            thread_count = threading.active_count()
            
            return SystemMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                memory_available=memory_available,
                memory_used=memory_used,
                disk_usage_percent=disk_usage_percent,
                disk_free=disk_free,
                network_io=network_io,
                process_count=process_count,
                thread_count=thread_count
            )
        
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
            return SystemMetrics(0, 0, 0, 0, 0, 0, {}, 0, 0)
    
    async def _needs_optimization(self, metrics: SystemMetrics) -> bool:
        """Check if optimization is needed"""
        return (
            metrics.memory_percent > self.config.memory_threshold or
            metrics.cpu_percent > self.config.cpu_threshold or
            metrics.disk_usage_percent > self.config.disk_threshold
        )
    
    async def _perform_optimization(self, metrics: SystemMetrics):
        """Perform system optimization"""
        try:
            optimization_tasks = []
            
            # Memory optimization
            if self.config.enable_memory_optimization and metrics.memory_percent > self.config.memory_threshold:
                optimization_tasks.append(self._optimize_memory(metrics))
            
            # CPU optimization
            if self.config.enable_cpu_optimization and metrics.cpu_percent > self.config.cpu_threshold:
                optimization_tasks.append(self._optimize_cpu(metrics))
            
            # Disk optimization
            if self.config.enable_disk_optimization and metrics.disk_usage_percent > self.config.disk_threshold:
                optimization_tasks.append(self._optimize_disk(metrics))
            
            # Network optimization
            if self.config.enable_network_optimization:
                optimization_tasks.append(self._optimize_network(metrics))
            
            # Execute optimizations
            if optimization_tasks:
                await asyncio.gather(*optimization_tasks, return_exceptions=True)
        
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
    
    async def _optimize_memory(self, metrics: SystemMetrics) -> OptimizationResult:
        """Optimize memory usage"""
        start_time = time.time()
        before_metrics = metrics
        
        try:
            # Force garbage collection
            collected = gc.collect()
            
            # Clear weak references
            self._weak_refs = [ref for ref in self._weak_refs if ref() is not None]
            
            # Clear unused caches (if any)
            # This would be implemented based on specific cache implementations
            
            # Get new metrics
            after_metrics = await self._collect_system_metrics()
            
            # Calculate improvement
            improvement = ((before_metrics.memory_percent - after_metrics.memory_percent) / 
                          before_metrics.memory_percent) * 100 if before_metrics.memory_percent > 0 else 0
            
            result = OptimizationResult(
                optimization_type=OptimizationType.MEMORY,
                level=OptimizationLevel.HIGH if improvement > 10 else OptimizationLevel.MEDIUM,
                before_metrics=before_metrics,
                after_metrics=after_metrics,
                improvement_percent=improvement,
                duration=time.time() - start_time
            )
            
            self._optimization_results.append(result)
            logger.info(f"Memory optimization completed: {improvement:.2f}% improvement")
            
            return result
        
        except Exception as e:
            logger.error(f"Memory optimization failed: {e}")
            raise
    
    async def _optimize_cpu(self, metrics: SystemMetrics) -> OptimizationResult:
        """Optimize CPU usage"""
        start_time = time.time()
        before_metrics = metrics
        
        try:
            # Adjust thread pool size based on CPU usage
            if metrics.cpu_percent > 90:
                # Reduce thread pool size
                new_size = max(1, self.config.max_workers // 2)
                if self._thread_pool:
                    self._thread_pool.shutdown(wait=False)
                    self._thread_pool = ThreadPoolExecutor(max_workers=new_size)
            elif metrics.cpu_percent < 50:
                # Increase thread pool size
                new_size = min(multiprocessing.cpu_count() * 2, self.config.max_workers * 2)
                if self._thread_pool:
                    self._thread_pool.shutdown(wait=False)
                    self._thread_pool = ThreadPoolExecutor(max_workers=new_size)
            
            # Get new metrics
            after_metrics = await self._collect_system_metrics()
            
            # Calculate improvement
            improvement = ((before_metrics.cpu_percent - after_metrics.cpu_percent) / 
                          before_metrics.cpu_percent) * 100 if before_metrics.cpu_percent > 0 else 0
            
            result = OptimizationResult(
                optimization_type=OptimizationType.CPU,
                level=OptimizationLevel.HIGH if improvement > 10 else OptimizationLevel.MEDIUM,
                before_metrics=before_metrics,
                after_metrics=after_metrics,
                improvement_percent=improvement,
                duration=time.time() - start_time
            )
            
            self._optimization_results.append(result)
            logger.info(f"CPU optimization completed: {improvement:.2f}% improvement")
            
            return result
        
        except Exception as e:
            logger.error(f"CPU optimization failed: {e}")
            raise
    
    async def _optimize_disk(self, metrics: SystemMetrics) -> OptimizationResult:
        """Optimize disk usage"""
        start_time = time.time()
        before_metrics = metrics
        
        try:
            # Clean temporary files
            import tempfile
            import os
            
            temp_dir = tempfile.gettempdir()
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        # Delete files older than 1 hour
                        if os.path.getmtime(file_path) < time.time() - 3600:
                            os.remove(file_path)
                    except (OSError, PermissionError):
                        pass
            
            # Get new metrics
            after_metrics = await self._collect_system_metrics()
            
            # Calculate improvement
            improvement = ((before_metrics.disk_usage_percent - after_metrics.disk_usage_percent) / 
                          before_metrics.disk_usage_percent) * 100 if before_metrics.disk_usage_percent > 0 else 0
            
            result = OptimizationResult(
                optimization_type=OptimizationType.DISK,
                level=OptimizationLevel.MEDIUM,
                before_metrics=before_metrics,
                after_metrics=after_metrics,
                improvement_percent=improvement,
                duration=time.time() - start_time
            )
            
            self._optimization_results.append(result)
            logger.info(f"Disk optimization completed: {improvement:.2f}% improvement")
            
            return result
        
        except Exception as e:
            logger.error(f"Disk optimization failed: {e}")
            raise
    
    async def _optimize_network(self, metrics: SystemMetrics) -> OptimizationResult:
        """Optimize network usage"""
        start_time = time.time()
        before_metrics = metrics
        
        try:
            # Network optimization would include:
            # - Connection pooling
            # - Request batching
            # - Compression
            # - Caching
            
            # For now, just return a placeholder result
            after_metrics = await self._collect_system_metrics()
            
            result = OptimizationResult(
                optimization_type=OptimizationType.NETWORK,
                level=OptimizationLevel.LOW,
                before_metrics=before_metrics,
                after_metrics=after_metrics,
                improvement_percent=0.0,
                duration=time.time() - start_time
            )
            
            self._optimization_results.append(result)
            logger.info("Network optimization completed")
            
            return result
        
        except Exception as e:
            logger.error(f"Network optimization failed: {e}")
            raise
    
    async def optimize_database_queries(self, queries: List[str]) -> List[str]:
        """Optimize database queries"""
        try:
            optimized_queries = []
            
            for query in queries:
                # Basic query optimization
                optimized_query = query.strip()
                
                # Remove unnecessary whitespace
                optimized_query = ' '.join(optimized_query.split())
                
                # Add query hints if needed
                if 'SELECT' in optimized_query.upper():
                    # Add index hints for common patterns
                    if 'WHERE' in optimized_query.upper():
                        # This would be more sophisticated in a real implementation
                        pass
                
                optimized_queries.append(optimized_query)
            
            logger.info(f"Optimized {len(queries)} database queries")
            return optimized_queries
        
        except Exception as e:
            logger.error(f"Database query optimization failed: {e}")
            return queries
    
    async def optimize_cache_strategy(self, cache_stats: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize cache strategy based on statistics"""
        try:
            recommendations = {}
            
            hit_rate = cache_stats.get('hit_rate', 0)
            miss_rate = cache_stats.get('miss_rate', 0)
            
            if hit_rate < 0.7:  # Low hit rate
                recommendations['increase_cache_size'] = True
                recommendations['extend_ttl'] = True
                recommendations['add_more_keys'] = True
            
            if miss_rate > 0.3:  # High miss rate
                recommendations['preload_frequent_data'] = True
                recommendations['implement_prediction'] = True
            
            # Memory usage optimization
            memory_usage = cache_stats.get('memory_usage', 0)
            if memory_usage > 0.8:  # High memory usage
                recommendations['reduce_cache_size'] = True
                recommendations['implement_lru_eviction'] = True
            
            logger.info(f"Cache optimization recommendations: {recommendations}")
            return recommendations
        
        except Exception as e:
            logger.error(f"Cache optimization failed: {e}")
            return {}
    
    def get_thread_pool(self) -> ThreadPoolExecutor:
        """Get thread pool executor"""
        return self._thread_pool
    
    def get_process_pool(self) -> ProcessPoolExecutor:
        """Get process pool executor"""
        return self._process_pool
    
    async def run_in_thread_pool(self, func: Callable, *args, **kwargs) -> Any:
        """Run function in thread pool"""
        if not self._thread_pool:
            raise RuntimeError("Thread pool not initialized")
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._thread_pool, func, *args, **kwargs)
    
    async def run_in_process_pool(self, func: Callable, *args, **kwargs) -> Any:
        """Run function in process pool"""
        if not self._process_pool:
            raise RuntimeError("Process pool not initialized")
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._process_pool, func, *args, **kwargs)
    
    def get_memory_snapshot(self) -> Dict[str, Any]:
        """Get memory snapshot"""
        try:
            if self._memory_tracer:
                snapshot = tracemalloc.take_snapshot()
                top_stats = snapshot.statistics('lineno')
                
                return {
                    "total_memory": sum(stat.size for stat in top_stats),
                    "top_allocations": [
                        {
                            "filename": stat.traceback.format()[0],
                            "size": stat.size,
                            "count": stat.count
                        }
                        for stat in top_stats[:10]
                    ],
                    "timestamp": DateTimeHelpers.now_utc().isoformat()
                }
            else:
                return {"error": "Memory tracing not enabled"}
        
        except Exception as e:
            logger.error(f"Failed to get memory snapshot: {e}")
            return {"error": str(e)}
    
    def get_profiler_stats(self) -> Dict[str, Any]:
        """Get profiler statistics"""
        try:
            if self._profiler:
                import pstats
                import io
                
                s = io.StringIO()
                ps = pstats.Stats(self._profiler, stream=s)
                ps.sort_stats('cumulative')
                ps.print_stats(20)  # Top 20 functions
                
                return {
                    "profile_data": s.getvalue(),
                    "timestamp": DateTimeHelpers.now_utc().isoformat()
                }
            else:
                return {"error": "Profiling not enabled"}
        
        except Exception as e:
            logger.error(f"Failed to get profiler stats: {e}")
            return {"error": str(e)}
    
    def get_optimization_history(self) -> List[OptimizationResult]:
        """Get optimization history"""
        return self._optimization_results[-50:]  # Last 50 optimizations
    
    def get_system_metrics_history(self) -> List[SystemMetrics]:
        """Get system metrics history"""
        return self._metrics_history[-100:]  # Last 100 metrics
    
    async def get_optimization_summary(self) -> Dict[str, Any]:
        """Get optimization summary"""
        try:
            recent_results = self._optimization_results[-10:]  # Last 10 optimizations
            
            if not recent_results:
                return {"message": "No optimizations performed yet"}
            
            # Calculate average improvements
            avg_improvements = {}
            for opt_type in OptimizationType:
                type_results = [r for r in recent_results if r.optimization_type == opt_type]
                if type_results:
                    avg_improvements[opt_type.value] = sum(r.improvement_percent for r in type_results) / len(type_results)
            
            # Get current system metrics
            current_metrics = await self._collect_system_metrics()
            
            return {
                "total_optimizations": len(self._optimization_results),
                "recent_optimizations": len(recent_results),
                "average_improvements": avg_improvements,
                "current_metrics": {
                    "cpu_percent": current_metrics.cpu_percent,
                    "memory_percent": current_metrics.memory_percent,
                    "disk_usage_percent": current_metrics.disk_usage_percent
                },
                "optimization_config": {
                    "memory_threshold": self.config.memory_threshold,
                    "cpu_threshold": self.config.cpu_threshold,
                    "disk_threshold": self.config.disk_threshold,
                    "optimization_interval": self.config.optimization_interval
                },
                "timestamp": DateTimeHelpers.now_utc().isoformat()
            }
        
        except Exception as e:
            logger.error(f"Failed to get optimization summary: {e}")
            return {"error": str(e)}


# Global optimization service
optimization_service = OptimizationService()


# Utility functions
async def start_optimization_service():
    """Start the optimization service"""
    await optimization_service.start()


async def stop_optimization_service():
    """Stop the optimization service"""
    await optimization_service.stop()


async def optimize_system() -> Dict[str, Any]:
    """Perform system optimization"""
    metrics = await optimization_service._collect_system_metrics()
    await optimization_service._perform_optimization(metrics)
    return await optimization_service.get_optimization_summary()


async def get_system_metrics() -> SystemMetrics:
    """Get current system metrics"""
    return await optimization_service._collect_system_metrics()


async def get_optimization_summary() -> Dict[str, Any]:
    """Get optimization summary"""
    return await optimization_service.get_optimization_summary()


def get_thread_pool() -> ThreadPoolExecutor:
    """Get thread pool executor"""
    return optimization_service.get_thread_pool()


def get_process_pool() -> ProcessPoolExecutor:
    """Get process pool executor"""
    return optimization_service.get_process_pool()


async def run_cpu_intensive_task(func: Callable, *args, **kwargs) -> Any:
    """Run CPU-intensive task in process pool"""
    return await optimization_service.run_in_process_pool(func, *args, **kwargs)


async def run_io_intensive_task(func: Callable, *args, **kwargs) -> Any:
    """Run I/O-intensive task in thread pool"""
    return await optimization_service.run_in_thread_pool(func, *args, **kwargs)


# Optimization decorators
def optimize_performance(func: Callable) -> Callable:
    """Decorator to optimize function performance"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        # Check if function is CPU or I/O intensive
        if 'cpu_intensive' in func.__name__.lower() or 'compute' in func.__name__.lower():
            return await run_cpu_intensive_task(func, *args, **kwargs)
        else:
            return await run_io_intensive_task(func, *args, **kwargs)
    
    return wrapper


def memory_efficient(func: Callable) -> Callable:
    """Decorator to make function memory efficient"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        # Force garbage collection before and after
        gc.collect()
        
        try:
            result = await func(*args, **kwargs)
            return result
        finally:
            gc.collect()
    
    return wrapper


def resource_monitored(func: Callable) -> Callable:
    """Decorator to monitor resource usage"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss
        
        try:
            result = await func(*args, **kwargs)
            return result
        finally:
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss
            
            logger.info(f"Function {func.__name__} - "
                       f"Duration: {end_time - start_time:.2f}s, "
                       f"Memory: {(end_memory - start_memory) / 1024 / 1024:.2f}MB")
    
    return wrapper


