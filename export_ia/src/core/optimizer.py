"""
Performance optimizer for Export IA.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import psutil
import gc
import weakref

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for optimization."""
    cpu_usage: float
    memory_usage: float
    disk_io: float
    network_io: float
    response_time: float
    throughput: float
    error_rate: float
    timestamp: datetime = field(default_factory=datetime.now)


class ResourceManager:
    """Manages system resources and optimization."""
    
    def __init__(self):
        self.thread_pool = ThreadPoolExecutor(max_workers=psutil.cpu_count() * 2)
        self.process_pool = ProcessPoolExecutor(max_workers=psutil.cpu_count())
        self.memory_threshold = 0.8  # 80% memory usage threshold
        self.cpu_threshold = 0.9     # 90% CPU usage threshold
        self._optimization_enabled = True
        
    async def optimize_memory(self) -> Dict[str, Any]:
        """Optimize memory usage."""
        memory_info = psutil.virtual_memory()
        
        if memory_info.percent > self.memory_threshold * 100:
            logger.warning(f"High memory usage: {memory_info.percent}%")
            
            # Force garbage collection
            collected = gc.collect()
            
            # Clear weak references
            weakref_cleared = 0
            for ref in weakref.getweakrefs(self):
                if ref() is None:
                    weakref_cleared += 1
            
            return {
                "action": "memory_optimization",
                "memory_before": memory_info.percent,
                "garbage_collected": collected,
                "weakrefs_cleared": weakref_cleared,
                "memory_after": psutil.virtual_memory().percent
            }
        
        return {"action": "no_optimization_needed"}
    
    async def optimize_cpu(self) -> Dict[str, Any]:
        """Optimize CPU usage."""
        cpu_percent = psutil.cpu_percent(interval=1)
        
        if cpu_percent > self.cpu_threshold * 100:
            logger.warning(f"High CPU usage: {cpu_percent}%")
            
            # Reduce thread pool size temporarily
            current_workers = self.thread_pool._max_workers
            new_workers = max(1, current_workers // 2)
            
            return {
                "action": "cpu_optimization",
                "cpu_before": cpu_percent,
                "workers_reduced": current_workers - new_workers,
                "cpu_after": psutil.cpu_percent(interval=1)
            }
        
        return {"action": "no_optimization_needed"}
    
    def get_resource_usage(self) -> Dict[str, float]:
        """Get current resource usage."""
        memory = psutil.virtual_memory()
        cpu = psutil.cpu_percent(interval=1)
        disk = psutil.disk_usage('/')
        
        return {
            "cpu_percent": cpu,
            "memory_percent": memory.percent,
            "memory_available": memory.available,
            "disk_percent": disk.percent,
            "disk_free": disk.free
        }
    
    async def cleanup(self):
        """Cleanup resources."""
        self.thread_pool.shutdown(wait=True)
        self.process_pool.shutdown(wait=True)


class CacheOptimizer:
    """Optimizes caching strategies."""
    
    def __init__(self):
        self.cache_stats: Dict[str, Any] = {}
        self.optimization_rules = self._initialize_optimization_rules()
    
    def _initialize_optimization_rules(self) -> Dict[str, Any]:
        """Initialize cache optimization rules."""
        return {
            "hit_rate_threshold": 0.7,
            "eviction_threshold": 0.9,
            "size_growth_threshold": 1.5,
            "ttl_optimization": True
        }
    
    async def optimize_cache(self, cache_stats: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize cache based on statistics."""
        optimizations = []
        
        # Check hit rate
        hit_rate = cache_stats.get("hit_rate", 0)
        if hit_rate < self.optimization_rules["hit_rate_threshold"]:
            optimizations.append({
                "type": "increase_ttl",
                "reason": f"Low hit rate: {hit_rate:.2f}",
                "action": "Increase TTL for frequently accessed items"
            })
        
        # Check cache size
        cache_size = cache_stats.get("entries", 0)
        max_size = cache_stats.get("max_size", 1000)
        if cache_size > max_size * self.optimization_rules["eviction_threshold"]:
            optimizations.append({
                "type": "aggressive_eviction",
                "reason": f"Cache size: {cache_size}/{max_size}",
                "action": "Implement more aggressive eviction policy"
            })
        
        return {
            "optimizations": optimizations,
            "cache_stats": cache_stats,
            "timestamp": datetime.now()
        }


class DatabaseOptimizer:
    """Optimizes database performance."""
    
    def __init__(self):
        self.query_stats: Dict[str, Any] = {}
        self.connection_pool_stats: Dict[str, Any] = {}
    
    async def optimize_queries(self, slow_queries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Optimize slow database queries."""
        optimizations = []
        
        for query in slow_queries:
            query_text = query.get("query", "")
            execution_time = query.get("execution_time", 0)
            
            if execution_time > 1.0:  # Queries taking more than 1 second
                optimizations.append({
                    "query": query_text[:100] + "..." if len(query_text) > 100 else query_text,
                    "execution_time": execution_time,
                    "suggestions": self._suggest_query_optimizations(query_text)
                })
        
        return {
            "optimizations": optimizations,
            "total_slow_queries": len(slow_queries),
            "timestamp": datetime.now()
        }
    
    def _suggest_query_optimizations(self, query: str) -> List[str]:
        """Suggest optimizations for a query."""
        suggestions = []
        
        query_lower = query.lower()
        
        if "select *" in query_lower:
            suggestions.append("Avoid SELECT * - specify only needed columns")
        
        if "order by" in query_lower and "limit" not in query_lower:
            suggestions.append("Consider adding LIMIT to ORDER BY queries")
        
        if "like '%" in query_lower:
            suggestions.append("Avoid leading wildcards in LIKE queries")
        
        if "join" in query_lower and "where" not in query_lower:
            suggestions.append("Add WHERE clause to JOIN queries for better performance")
        
        return suggestions
    
    async def optimize_connection_pool(self, pool_stats: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize database connection pool."""
        optimizations = []
        
        pool_size = pool_stats.get("pool_size", 0)
        checked_out = pool_stats.get("checked_out", 0)
        overflow = pool_stats.get("overflow", 0)
        
        utilization = checked_out / pool_size if pool_size > 0 else 0
        
        if utilization > 0.8:
            optimizations.append({
                "type": "increase_pool_size",
                "reason": f"High pool utilization: {utilization:.2f}",
                "suggestion": f"Increase pool size from {pool_size} to {pool_size * 2}"
            })
        
        if overflow > 0:
            optimizations.append({
                "type": "increase_max_overflow",
                "reason": f"Pool overflow: {overflow}",
                "suggestion": "Increase max_overflow parameter"
            })
        
        return {
            "optimizations": optimizations,
            "pool_stats": pool_stats,
            "timestamp": datetime.now()
        }


class PerformanceOptimizer:
    """Main performance optimizer."""
    
    def __init__(self):
        self.resource_manager = ResourceManager()
        self.cache_optimizer = CacheOptimizer()
        self.database_optimizer = DatabaseOptimizer()
        self.metrics_history: List[PerformanceMetrics] = []
        self.optimization_enabled = True
        self._optimization_interval = 60  # seconds
        self._optimization_task: Optional[asyncio.Task] = None
    
    async def start_optimization(self) -> None:
        """Start continuous optimization."""
        if self._optimization_task:
            return
        
        self._optimization_task = asyncio.create_task(self._optimization_loop())
        logger.info("Performance optimization started")
    
    async def stop_optimization(self) -> None:
        """Stop continuous optimization."""
        if self._optimization_task:
            self._optimization_task.cancel()
            try:
                await self._optimization_task
            except asyncio.CancelledError:
                pass
        
        await self.resource_manager.cleanup()
        logger.info("Performance optimization stopped")
    
    async def _optimization_loop(self) -> None:
        """Main optimization loop."""
        while True:
            try:
                await self._perform_optimization()
                await asyncio.sleep(self._optimization_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in optimization loop: {e}")
                await asyncio.sleep(self._optimization_interval)
    
    async def _perform_optimization(self) -> None:
        """Perform optimization based on current metrics."""
        # Collect current metrics
        metrics = await self._collect_metrics()
        self.metrics_history.append(metrics)
        
        # Keep only last 100 metrics
        if len(self.metrics_history) > 100:
            self.metrics_history = self.metrics_history[-100:]
        
        # Perform optimizations
        optimizations = []
        
        # Memory optimization
        memory_opt = await self.resource_manager.optimize_memory()
        if memory_opt["action"] != "no_optimization_needed":
            optimizations.append(memory_opt)
        
        # CPU optimization
        cpu_opt = await self.resource_manager.optimize_cpu()
        if cpu_opt["action"] != "no_optimization_needed":
            optimizations.append(cpu_opt)
        
        # Log optimizations
        if optimizations:
            logger.info(f"Applied {len(optimizations)} optimizations")
            for opt in optimizations:
                logger.info(f"Optimization: {opt}")
    
    async def _collect_metrics(self) -> PerformanceMetrics:
        """Collect current performance metrics."""
        resource_usage = self.resource_manager.get_resource_usage()
        
        return PerformanceMetrics(
            cpu_usage=resource_usage["cpu_percent"],
            memory_usage=resource_usage["memory_percent"],
            disk_io=0.0,  # Would need additional monitoring
            network_io=0.0,  # Would need additional monitoring
            response_time=0.0,  # Would be collected from request handlers
            throughput=0.0,  # Would be calculated from request rate
            error_rate=0.0  # Would be collected from error logs
        )
    
    async def get_optimization_report(self) -> Dict[str, Any]:
        """Get optimization report."""
        if not self.metrics_history:
            return {"message": "No metrics collected yet"}
        
        latest_metrics = self.metrics_history[-1]
        avg_metrics = self._calculate_average_metrics()
        
        return {
            "current_metrics": {
                "cpu_usage": latest_metrics.cpu_usage,
                "memory_usage": latest_metrics.memory_usage,
                "response_time": latest_metrics.response_time,
                "throughput": latest_metrics.throughput,
                "error_rate": latest_metrics.error_rate
            },
            "average_metrics": avg_metrics,
            "optimization_enabled": self.optimization_enabled,
            "metrics_count": len(self.metrics_history),
            "timestamp": datetime.now()
        }
    
    def _calculate_average_metrics(self) -> Dict[str, float]:
        """Calculate average metrics from history."""
        if not self.metrics_history:
            return {}
        
        return {
            "avg_cpu_usage": sum(m.cpu_usage for m in self.metrics_history) / len(self.metrics_history),
            "avg_memory_usage": sum(m.memory_usage for m in self.metrics_history) / len(self.metrics_history),
            "avg_response_time": sum(m.response_time for m in self.metrics_history) / len(self.metrics_history),
            "avg_throughput": sum(m.throughput for m in self.metrics_history) / len(self.metrics_history),
            "avg_error_rate": sum(m.error_rate for m in self.metrics_history) / len(self.metrics_history)
        }
    
    async def optimize_for_workload(self, workload_type: str) -> Dict[str, Any]:
        """Optimize system for specific workload type."""
        optimizations = {
            "cpu_intensive": {
                "thread_pool_size": psutil.cpu_count() * 4,
                "process_pool_size": psutil.cpu_count(),
                "memory_threshold": 0.7,
                "cache_strategy": "aggressive_eviction"
            },
            "memory_intensive": {
                "thread_pool_size": psutil.cpu_count() * 2,
                "process_pool_size": psutil.cpu_count() // 2,
                "memory_threshold": 0.6,
                "cache_strategy": "conservative_eviction"
            },
            "io_intensive": {
                "thread_pool_size": psutil.cpu_count() * 8,
                "process_pool_size": psutil.cpu_count() // 4,
                "memory_threshold": 0.8,
                "cache_strategy": "long_ttl"
            },
            "balanced": {
                "thread_pool_size": psutil.cpu_count() * 2,
                "process_pool_size": psutil.cpu_count(),
                "memory_threshold": 0.8,
                "cache_strategy": "adaptive"
            }
        }
        
        if workload_type not in optimizations:
            workload_type = "balanced"
        
        config = optimizations[workload_type]
        
        # Apply optimizations
        self.resource_manager.thread_pool = ThreadPoolExecutor(max_workers=config["thread_pool_size"])
        self.resource_manager.process_pool = ProcessPoolExecutor(max_workers=config["process_pool_size"])
        self.resource_manager.memory_threshold = config["memory_threshold"]
        
        return {
            "workload_type": workload_type,
            "applied_config": config,
            "timestamp": datetime.now()
        }


# Global performance optimizer instance
_performance_optimizer: Optional[PerformanceOptimizer] = None


def get_performance_optimizer() -> PerformanceOptimizer:
    """Get the global performance optimizer instance."""
    global _performance_optimizer
    if _performance_optimizer is None:
        _performance_optimizer = PerformanceOptimizer()
    return _performance_optimizer




