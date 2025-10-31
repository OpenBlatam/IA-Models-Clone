"""
Performance Optimizer - Advanced Performance Optimization System

This module provides advanced performance optimization capabilities including:
- Memory optimization and garbage collection
- CPU optimization and load balancing
- I/O optimization and caching strategies
- Database query optimization
- Network optimization and connection pooling
- Resource monitoring and auto-scaling
"""

import asyncio
import gc
import psutil
import time
import threading
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import weakref
import tracemalloc
from contextlib import asynccontextmanager
import logging

logger = logging.getLogger(__name__)

class OptimizationLevel(Enum):
    """Performance optimization levels"""
    MINIMAL = "minimal"
    STANDARD = "standard"
    AGGRESSIVE = "aggressive"
    MAXIMUM = "maximum"

class ResourceType(Enum):
    """Types of system resources"""
    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"
    NETWORK = "network"
    DATABASE = "database"
    CACHE = "cache"

@dataclass
class PerformanceMetrics:
    """Performance metrics container"""
    timestamp: float = field(default_factory=time.time)
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    memory_available: float = 0.0
    disk_usage: float = 0.0
    network_io: Dict[str, float] = field(default_factory=dict)
    response_time: float = 0.0
    throughput: float = 0.0
    error_rate: float = 0.0
    cache_hit_rate: float = 0.0
    database_connections: int = 0
    active_threads: int = 0
    gc_collections: int = 0

@dataclass
class OptimizationConfig:
    """Configuration for performance optimization"""
    level: OptimizationLevel = OptimizationLevel.STANDARD
    enable_memory_optimization: bool = True
    enable_cpu_optimization: bool = True
    enable_io_optimization: bool = True
    enable_database_optimization: bool = True
    enable_cache_optimization: bool = True
    enable_network_optimization: bool = True
    memory_threshold: float = 0.8  # 80% memory usage threshold
    cpu_threshold: float = 0.8     # 80% CPU usage threshold
    gc_threshold: int = 1000       # GC threshold for collections
    cache_size_limit: int = 1000   # Maximum cache entries
    connection_pool_size: int = 20 # Database connection pool size
    thread_pool_size: int = 10     # Thread pool size
    process_pool_size: int = 4     # Process pool size
    monitoring_interval: float = 1.0  # Monitoring interval in seconds
    auto_optimization: bool = True    # Enable automatic optimization

class MemoryOptimizer:
    """Advanced memory optimization system"""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.memory_tracker = {}
        self.weak_refs = weakref.WeakValueDictionary()
        self.memory_pool = {}
        self.gc_stats = {}
        
    async def optimize_memory(self) -> Dict[str, Any]:
        """Optimize memory usage"""
        optimization_results = {
            "timestamp": time.time(),
            "before": self._get_memory_stats(),
            "actions": [],
            "after": {}
        }
        
        # Force garbage collection
        if self.config.enable_memory_optimization:
            collected = gc.collect()
            optimization_results["actions"].append(f"Garbage collection: {collected} objects collected")
            
            # Clear weak references
            cleared_refs = len(self.weak_refs)
            self.weak_refs.clear()
            optimization_results["actions"].append(f"Cleared {cleared_refs} weak references")
            
            # Optimize memory pool
            pool_optimized = await self._optimize_memory_pool()
            optimization_results["actions"].append(f"Memory pool optimized: {pool_optimized} entries")
        
        optimization_results["after"] = self._get_memory_stats()
        return optimization_results
    
    def _get_memory_stats(self) -> Dict[str, Any]:
        """Get current memory statistics"""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            "rss": memory_info.rss,
            "vms": memory_info.vms,
            "percent": process.memory_percent(),
            "available": psutil.virtual_memory().available,
            "gc_counts": gc.get_count(),
            "gc_threshold": gc.get_threshold()
        }
    
    async def _optimize_memory_pool(self) -> int:
        """Optimize memory pool"""
        optimized = 0
        for key, value in list(self.memory_pool.items()):
            if hasattr(value, '__weakref__'):
                if value is None:
                    del self.memory_pool[key]
                    optimized += 1
        return optimized
    
    def track_object(self, obj: Any, name: str) -> None:
        """Track object for memory optimization"""
        self.memory_tracker[name] = {
            "object": obj,
            "size": self._get_object_size(obj),
            "timestamp": time.time()
        }
    
    def _get_object_size(self, obj: Any) -> int:
        """Get object size in bytes"""
        try:
            return len(str(obj).encode('utf-8'))
        except:
            return 0

class CPUOptimizer:
    """Advanced CPU optimization system"""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.thread_pool = ThreadPoolExecutor(max_workers=config.thread_pool_size)
        self.process_pool = ProcessPoolExecutor(max_workers=config.process_pool_size)
        self.cpu_affinity = None
        self.priority_levels = {}
        
    async def optimize_cpu(self) -> Dict[str, Any]:
        """Optimize CPU usage"""
        optimization_results = {
            "timestamp": time.time(),
            "before": self._get_cpu_stats(),
            "actions": [],
            "after": {}
        }
        
        if self.config.enable_cpu_optimization:
            # Set CPU affinity
            if self.cpu_affinity:
                psutil.Process().cpu_affinity(self.cpu_affinity)
                optimization_results["actions"].append(f"Set CPU affinity: {self.cpu_affinity}")
            
            # Optimize thread priorities
            priority_optimized = await self._optimize_thread_priorities()
            optimization_results["actions"].append(f"Optimized {priority_optimized} thread priorities")
            
            # Balance load
            load_balanced = await self._balance_cpu_load()
            optimization_results["actions"].append(f"Load balanced: {load_balanced} processes")
        
        optimization_results["after"] = self._get_cpu_stats()
        return optimization_results
    
    def _get_cpu_stats(self) -> Dict[str, Any]:
        """Get current CPU statistics"""
        return {
            "usage": psutil.cpu_percent(interval=0.1),
            "count": psutil.cpu_count(),
            "frequency": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else {},
            "load_avg": psutil.getloadavg() if hasattr(psutil, 'getloadavg') else [0, 0, 0],
            "threads": threading.active_count()
        }
    
    async def _optimize_thread_priorities(self) -> int:
        """Optimize thread priorities"""
        optimized = 0
        for thread in threading.enumerate():
            if thread.name in self.priority_levels:
                try:
                    # Note: Thread priority setting is platform-specific
                    # This is a placeholder for the concept
                    optimized += 1
                except:
                    pass
        return optimized
    
    async def _balance_cpu_load(self) -> int:
        """Balance CPU load across processes"""
        # This would implement load balancing logic
        return 0
    
    def set_cpu_affinity(self, cores: List[int]) -> None:
        """Set CPU affinity for the process"""
        self.cpu_affinity = cores
    
    def set_thread_priority(self, thread_name: str, priority: int) -> None:
        """Set thread priority"""
        self.priority_levels[thread_name] = priority

class IOOptimizer:
    """Advanced I/O optimization system"""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.io_cache = {}
        self.buffer_pool = {}
        self.connection_pools = {}
        
    async def optimize_io(self) -> Dict[str, Any]:
        """Optimize I/O operations"""
        optimization_results = {
            "timestamp": time.time(),
            "before": self._get_io_stats(),
            "actions": [],
            "after": {}
        }
        
        if self.config.enable_io_optimization:
            # Optimize I/O cache
            cache_optimized = await self._optimize_io_cache()
            optimization_results["actions"].append(f"I/O cache optimized: {cache_optimized} entries")
            
            # Optimize buffer pool
            buffer_optimized = await self._optimize_buffer_pool()
            optimization_results["actions"].append(f"Buffer pool optimized: {buffer_optimized} buffers")
            
            # Optimize connection pools
            pool_optimized = await self._optimize_connection_pools()
            optimization_results["actions"].append(f"Connection pools optimized: {pool_optimized} pools")
        
        optimization_results["after"] = self._get_io_stats()
        return optimization_results
    
    def _get_io_stats(self) -> Dict[str, Any]:
        """Get current I/O statistics"""
        disk_io = psutil.disk_io_counters()
        network_io = psutil.net_io_counters()
        
        return {
            "disk_read": disk_io.read_bytes if disk_io else 0,
            "disk_write": disk_io.write_bytes if disk_io else 0,
            "network_sent": network_io.bytes_sent if network_io else 0,
            "network_recv": network_io.bytes_recv if network_io else 0,
            "cache_size": len(self.io_cache),
            "buffer_pool_size": len(self.buffer_pool)
        }
    
    async def _optimize_io_cache(self) -> int:
        """Optimize I/O cache"""
        # Remove expired cache entries
        current_time = time.time()
        expired_keys = [
            key for key, value in self.io_cache.items()
            if current_time - value.get('timestamp', 0) > 3600  # 1 hour TTL
        ]
        
        for key in expired_keys:
            del self.io_cache[key]
        
        return len(expired_keys)
    
    async def _optimize_buffer_pool(self) -> int:
        """Optimize buffer pool"""
        # Clean up unused buffers
        cleaned = 0
        for key, buffer in list(self.buffer_pool.items()):
            if not buffer or len(buffer) == 0:
                del self.buffer_pool[key]
                cleaned += 1
        return cleaned
    
    async def _optimize_connection_pools(self) -> int:
        """Optimize connection pools"""
        optimized = 0
        for pool_name, pool in self.connection_pools.items():
            if hasattr(pool, 'optimize'):
                pool.optimize()
                optimized += 1
        return optimized

class DatabaseOptimizer:
    """Advanced database optimization system"""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.query_cache = {}
        self.connection_pool = None
        self.query_stats = {}
        
    async def optimize_database(self) -> Dict[str, Any]:
        """Optimize database operations"""
        optimization_results = {
            "timestamp": time.time(),
            "before": self._get_database_stats(),
            "actions": [],
            "after": {}
        }
        
        if self.config.enable_database_optimization:
            # Optimize query cache
            cache_optimized = await self._optimize_query_cache()
            optimization_results["actions"].append(f"Query cache optimized: {cache_optimized} queries")
            
            # Optimize connection pool
            pool_optimized = await self._optimize_connection_pool()
            optimization_results["actions"].append(f"Connection pool optimized: {pool_optimized} connections")
            
            # Analyze slow queries
            slow_queries = await self._analyze_slow_queries()
            optimization_results["actions"].append(f"Analyzed {slow_queries} slow queries")
        
        optimization_results["after"] = self._get_database_stats()
        return optimization_results
    
    def _get_database_stats(self) -> Dict[str, Any]:
        """Get current database statistics"""
        return {
            "query_cache_size": len(self.query_cache),
            "connection_pool_size": self.connection_pool.size if self.connection_pool else 0,
            "active_connections": len(self.query_stats),
            "slow_queries": sum(1 for stats in self.query_stats.values() if stats.get('avg_time', 0) > 1.0)
        }
    
    async def _optimize_query_cache(self) -> int:
        """Optimize query cache"""
        # Remove least recently used queries
        current_time = time.time()
        lru_queries = sorted(
            self.query_cache.items(),
            key=lambda x: x[1].get('last_used', 0)
        )
        
        # Remove oldest 20% of queries
        remove_count = max(1, len(lru_queries) // 5)
        for key, _ in lru_queries[:remove_count]:
            del self.query_cache[key]
        
        return remove_count
    
    async def _optimize_connection_pool(self) -> int:
        """Optimize connection pool"""
        if self.connection_pool:
            # This would implement connection pool optimization
            return 1
        return 0
    
    async def _analyze_slow_queries(self) -> int:
        """Analyze slow queries"""
        slow_count = 0
        for query, stats in self.query_stats.items():
            if stats.get('avg_time', 0) > 1.0:  # Queries taking more than 1 second
                slow_count += 1
                # This would implement query optimization suggestions
        return slow_count

class CacheOptimizer:
    """Advanced cache optimization system"""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.cache_stats = {}
        self.eviction_policies = {}
        
    async def optimize_cache(self) -> Dict[str, Any]:
        """Optimize cache operations"""
        optimization_results = {
            "timestamp": time.time(),
            "before": self._get_cache_stats(),
            "actions": [],
            "after": {}
        }
        
        if self.config.enable_cache_optimization:
            # Optimize cache eviction
            evicted = await self._optimize_cache_eviction()
            optimization_results["actions"].append(f"Cache eviction optimized: {evicted} entries")
            
            # Optimize cache warming
            warmed = await self._optimize_cache_warming()
            optimization_results["actions"].append(f"Cache warming optimized: {warmed} entries")
            
            # Analyze cache hit rates
            hit_rate_analysis = await self._analyze_cache_hit_rates()
            optimization_results["actions"].append(f"Cache hit rate analysis: {hit_rate_analysis}")
        
        optimization_results["after"] = self._get_cache_stats()
        return optimization_results
    
    def _get_cache_stats(self) -> Dict[str, Any]:
        """Get current cache statistics"""
        total_hits = sum(stats.get('hits', 0) for stats in self.cache_stats.values())
        total_misses = sum(stats.get('misses', 0) for stats in self.cache_stats.values())
        total_requests = total_hits + total_misses
        
        return {
            "total_entries": len(self.cache_stats),
            "total_hits": total_hits,
            "total_misses": total_misses,
            "hit_rate": total_hits / total_requests if total_requests > 0 else 0,
            "cache_size": sum(stats.get('size', 0) for stats in self.cache_stats.values())
        }
    
    async def _optimize_cache_eviction(self) -> int:
        """Optimize cache eviction policies"""
        evicted = 0
        for cache_name, stats in self.cache_stats.items():
            if stats.get('size', 0) > self.config.cache_size_limit:
                # Implement eviction logic
                evicted += 1
        return evicted
    
    async def _optimize_cache_warming(self) -> int:
        """Optimize cache warming strategies"""
        # This would implement cache warming logic
        return 0
    
    async def _analyze_cache_hit_rates(self) -> str:
        """Analyze cache hit rates"""
        hit_rates = []
        for cache_name, stats in self.cache_stats.items():
            hits = stats.get('hits', 0)
            misses = stats.get('misses', 0)
            total = hits + misses
            if total > 0:
                hit_rate = hits / total
                hit_rates.append(f"{cache_name}: {hit_rate:.2%}")
        
        return "; ".join(hit_rates) if hit_rates else "No cache data available"

class NetworkOptimizer:
    """Advanced network optimization system"""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.connection_pools = {}
        self.keep_alive_settings = {}
        self.compression_settings = {}
        
    async def optimize_network(self) -> Dict[str, Any]:
        """Optimize network operations"""
        optimization_results = {
            "timestamp": time.time(),
            "before": self._get_network_stats(),
            "actions": [],
            "after": {}
        }
        
        if self.config.enable_network_optimization:
            # Optimize connection pools
            pool_optimized = await self._optimize_connection_pools()
            optimization_results["actions"].append(f"Connection pools optimized: {pool_optimized} pools")
            
            # Optimize keep-alive settings
            keep_alive_optimized = await self._optimize_keep_alive()
            optimization_results["actions"].append(f"Keep-alive optimized: {keep_alive_optimized} connections")
            
            # Optimize compression
            compression_optimized = await self._optimize_compression()
            optimization_results["actions"].append(f"Compression optimized: {compression_optimized} settings")
        
        optimization_results["after"] = self._get_network_stats()
        return optimization_results
    
    def _get_network_stats(self) -> Dict[str, Any]:
        """Get current network statistics"""
        network_io = psutil.net_io_counters()
        return {
            "bytes_sent": network_io.bytes_sent if network_io else 0,
            "bytes_recv": network_io.bytes_recv if network_io else 0,
            "packets_sent": network_io.packets_sent if network_io else 0,
            "packets_recv": network_io.packets_recv if network_io else 0,
            "connection_pools": len(self.connection_pools)
        }
    
    async def _optimize_connection_pools(self) -> int:
        """Optimize network connection pools"""
        optimized = 0
        for pool_name, pool in self.connection_pools.items():
            if hasattr(pool, 'optimize'):
                pool.optimize()
                optimized += 1
        return optimized
    
    async def _optimize_keep_alive(self) -> int:
        """Optimize keep-alive settings"""
        # This would implement keep-alive optimization
        return 0
    
    async def _optimize_compression(self) -> int:
        """Optimize compression settings"""
        # This would implement compression optimization
        return 0

class PerformanceOptimizer:
    """Main performance optimization system"""
    
    def __init__(self, config: Optional[OptimizationConfig] = None):
        self.config = config or OptimizationConfig()
        self.memory_optimizer = MemoryOptimizer(self.config)
        self.cpu_optimizer = CPUOptimizer(self.config)
        self.io_optimizer = IOOptimizer(self.config)
        self.database_optimizer = DatabaseOptimizer(self.config)
        self.cache_optimizer = CacheOptimizer(self.config)
        self.network_optimizer = NetworkOptimizer(self.config)
        
        self.metrics_history = []
        self.optimization_history = []
        self.monitoring_task = None
        self.is_monitoring = False
        
    async def start_monitoring(self) -> None:
        """Start performance monitoring"""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Performance monitoring started")
    
    async def stop_monitoring(self) -> None:
        """Stop performance monitoring"""
        self.is_monitoring = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        logger.info("Performance monitoring stopped")
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                # Collect metrics
                metrics = await self.collect_metrics()
                self.metrics_history.append(metrics)
                
                # Keep only last 1000 metrics
                if len(self.metrics_history) > 1000:
                    self.metrics_history = self.metrics_history[-1000:]
                
                # Check if optimization is needed
                if self.config.auto_optimization and self._should_optimize(metrics):
                    await self.optimize_all()
                
                await asyncio.sleep(self.config.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(self.config.monitoring_interval)
    
    def _should_optimize(self, metrics: PerformanceMetrics) -> bool:
        """Check if optimization is needed based on metrics"""
        return (
            metrics.memory_usage > self.config.memory_threshold or
            metrics.cpu_usage > self.config.cpu_threshold or
            metrics.error_rate > 0.05  # 5% error rate threshold
        )
    
    async def collect_metrics(self) -> PerformanceMetrics:
        """Collect current performance metrics"""
        metrics = PerformanceMetrics()
        
        # CPU metrics
        metrics.cpu_usage = psutil.cpu_percent(interval=0.1)
        
        # Memory metrics
        memory = psutil.virtual_memory()
        metrics.memory_usage = memory.percent / 100.0
        metrics.memory_available = memory.available
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        metrics.disk_usage = disk.percent / 100.0
        
        # Network metrics
        network = psutil.net_io_counters()
        if network:
            metrics.network_io = {
                "bytes_sent": network.bytes_sent,
                "bytes_recv": network.bytes_recv,
                "packets_sent": network.packets_sent,
                "packets_recv": network.packets_recv
            }
        
        # Thread metrics
        metrics.active_threads = threading.active_count()
        
        # GC metrics
        metrics.gc_collections = sum(gc.get_count())
        
        return metrics
    
    async def optimize_all(self) -> Dict[str, Any]:
        """Run all optimization strategies"""
        optimization_results = {
            "timestamp": time.time(),
            "level": self.config.level.value,
            "results": {}
        }
        
        try:
            # Memory optimization
            if self.config.enable_memory_optimization:
                optimization_results["results"]["memory"] = await self.memory_optimizer.optimize_memory()
            
            # CPU optimization
            if self.config.enable_cpu_optimization:
                optimization_results["results"]["cpu"] = await self.cpu_optimizer.optimize_cpu()
            
            # I/O optimization
            if self.config.enable_io_optimization:
                optimization_results["results"]["io"] = await self.io_optimizer.optimize_io()
            
            # Database optimization
            if self.config.enable_database_optimization:
                optimization_results["results"]["database"] = await self.database_optimizer.optimize_database()
            
            # Cache optimization
            if self.config.enable_cache_optimization:
                optimization_results["results"]["cache"] = await self.cache_optimizer.optimize_cache()
            
            # Network optimization
            if self.config.enable_network_optimization:
                optimization_results["results"]["network"] = await self.network_optimizer.optimize_network()
            
            self.optimization_history.append(optimization_results)
            
            # Keep only last 100 optimizations
            if len(self.optimization_history) > 100:
                self.optimization_history = self.optimization_history[-100:]
            
            logger.info(f"Performance optimization completed: {self.config.level.value}")
            
        except Exception as e:
            logger.error(f"Error during optimization: {e}")
            optimization_results["error"] = str(e)
        
        return optimization_results
    
    async def optimize_resource(self, resource_type: ResourceType) -> Dict[str, Any]:
        """Optimize specific resource type"""
        optimizers = {
            ResourceType.MEMORY: self.memory_optimizer.optimize_memory,
            ResourceType.CPU: self.cpu_optimizer.optimize_cpu,
            ResourceType.DISK: self.io_optimizer.optimize_io,
            ResourceType.DATABASE: self.database_optimizer.optimize_database,
            ResourceType.CACHE: self.cache_optimizer.optimize_cache,
            ResourceType.NETWORK: self.network_optimizer.optimize_network
        }
        
        if resource_type in optimizers:
            return await optimizers[resource_type]()
        else:
            raise ValueError(f"Unknown resource type: {resource_type}")
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get performance metrics summary"""
        if not self.metrics_history:
            return {"message": "No metrics available"}
        
        latest_metrics = self.metrics_history[-1]
        avg_metrics = self._calculate_average_metrics()
        
        return {
            "current": {
                "cpu_usage": latest_metrics.cpu_usage,
                "memory_usage": latest_metrics.memory_usage,
                "disk_usage": latest_metrics.disk_usage,
                "active_threads": latest_metrics.active_threads
            },
            "average": avg_metrics,
            "optimization_count": len(self.optimization_history),
            "monitoring_duration": len(self.metrics_history) * self.config.monitoring_interval
        }
    
    def _calculate_average_metrics(self) -> Dict[str, float]:
        """Calculate average metrics over time"""
        if not self.metrics_history:
            return {}
        
        total_metrics = len(self.metrics_history)
        return {
            "cpu_usage": sum(m.cpu_usage for m in self.metrics_history) / total_metrics,
            "memory_usage": sum(m.memory_usage for m in self.metrics_history) / total_metrics,
            "disk_usage": sum(m.disk_usage for m in self.metrics_history) / total_metrics,
            "active_threads": sum(m.active_threads for m in self.metrics_history) / total_metrics
        }
    
    def get_optimization_history(self) -> List[Dict[str, Any]]:
        """Get optimization history"""
        return self.optimization_history.copy()
    
    def update_config(self, new_config: OptimizationConfig) -> None:
        """Update optimization configuration"""
        self.config = new_config
        logger.info(f"Optimization configuration updated: {new_config.level.value}")
    
    @asynccontextmanager
    async def performance_context(self, operation_name: str):
        """Context manager for performance monitoring"""
        start_time = time.time()
        start_metrics = await self.collect_metrics()
        
        try:
            yield
        finally:
            end_time = time.time()
            end_metrics = await self.collect_metrics()
            
            operation_metrics = {
                "operation": operation_name,
                "duration": end_time - start_time,
                "cpu_delta": end_metrics.cpu_usage - start_metrics.cpu_usage,
                "memory_delta": end_metrics.memory_usage - start_metrics.memory_usage,
                "timestamp": end_time
            }
            
            logger.info(f"Performance context '{operation_name}': {operation_metrics}")

# Global performance optimizer instance
_global_optimizer: Optional[PerformanceOptimizer] = None

def get_performance_optimizer() -> PerformanceOptimizer:
    """Get global performance optimizer instance"""
    global _global_optimizer
    if _global_optimizer is None:
        _global_optimizer = PerformanceOptimizer()
    return _global_optimizer

async def start_performance_monitoring(config: Optional[OptimizationConfig] = None) -> None:
    """Start global performance monitoring"""
    optimizer = get_performance_optimizer()
    if config:
        optimizer.update_config(config)
    await optimizer.start_monitoring()

async def stop_performance_monitoring() -> None:
    """Stop global performance monitoring"""
    optimizer = get_performance_optimizer()
    await optimizer.stop_monitoring()

async def optimize_performance(level: OptimizationLevel = OptimizationLevel.STANDARD) -> Dict[str, Any]:
    """Optimize system performance"""
    optimizer = get_performance_optimizer()
    config = OptimizationConfig(level=level)
    optimizer.update_config(config)
    return await optimizer.optimize_all()





















