"""
PDF Variantes - Performance Optimizations
==========================================

Performance optimization utilities and configurations.
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime
import asyncio
import concurrent.futures
from pathlib import Path
import psutil
import time

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics."""
    operation: str
    duration_ms: float
    memory_mb: float
    cpu_percent: float
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "operation": self.operation,
            "duration_ms": self.duration_ms,
            "memory_mb": self.memory_mb,
            "cpu_percent": self.cpu_percent,
            "timestamp": self.timestamp.isoformat()
        }


class PerformanceOptimizer:
    """Performance optimization engine."""
    
    def __init__(self):
        self.metrics: List[PerformanceMetrics] = []
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        self.process_pool = concurrent.futures.ProcessPoolExecutor(max_workers=2)
        logger.info("Initialized Performance Optimizer")
    
    def measure_performance(self, operation_name: str):
        """Decorator to measure performance."""
        def decorator(func):
            async def async_wrapper(*args, **kwargs):
                start_time = time.time()
                start_memory = psutil.Process().memory_info().rss / 1024 / 1024
                start_cpu = psutil.Process().cpu_percent()
                
                try:
                    result = await func(*args, **kwargs)
                    return result
                finally:
                    end_time = time.time()
                    end_memory = psutil.Process().memory_info().rss / 1024 / 1024
                    end_cpu = psutil.Process().cpu_percent()
                    
                    metric = PerformanceMetrics(
                        operation=operation_name,
                        duration_ms=(end_time - start_time) * 1000,
                        memory_mb=end_memory - start_memory,
                        cpu_percent=end_cpu - start_cpu,
                        timestamp=datetime.utcnow()
                    )
                    
                    self.metrics.append(metric)
                    logger.info(f"Performance: {operation_name} - {metric.duration_ms:.2f}ms")
            
            def sync_wrapper(*args, **kwargs):
                start_time = time.time()
                start_memory = psutil.Process().memory_info().rss / 1024 / 1024
                start_cpu = psutil.Process().cpu_percent()
                
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    end_time = time.time()
                    end_memory = psutil.Process().memory_info().rss / 1024 / 1024
                    end_cpu = psutil.Process().cpu_percent()
                    
                    metric = PerformanceMetrics(
                        operation=operation_name,
                        duration_ms=(end_time - start_time) * 1000,
                        memory_mb=end_memory - start_memory,
                        cpu_percent=end_cpu - start_cpu,
                        timestamp=datetime.utcnow()
                    )
                    
                    self.metrics.append(metric)
                    logger.info(f"Performance: {operation_name} - {metric.duration_ms:.2f}ms")
            
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper
        
        return decorator
    
    async def optimize_concurrent_operations(
        self,
        operations: List[callable],
        max_concurrent: int = 5
    ) -> List[Any]:
        """Optimize concurrent operations."""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def run_operation(op):
            async with semaphore:
                return await op()
        
        tasks = [run_operation(op) for op in operations]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return results
    
    def optimize_memory_usage(self):
        """Optimize memory usage."""
        import gc
        gc.collect()
        
        # Log memory usage
        memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
        logger.info(f"Memory usage: {memory_mb:.2f} MB")
        
        return memory_mb
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        if not self.metrics:
            return {"message": "No metrics available"}
        
        total_operations = len(self.metrics)
        avg_duration = sum(m.duration_ms for m in self.metrics) / total_operations
        avg_memory = sum(m.memory_mb for m in self.metrics) / total_operations
        avg_cpu = sum(m.cpu_percent for m in self.metrics) / total_operations
        
        return {
            "total_operations": total_operations,
            "average_duration_ms": avg_duration,
            "average_memory_mb": avg_memory,
            "average_cpu_percent": avg_cpu,
            "slowest_operation": max(self.metrics, key=lambda m: m.duration_ms).operation,
            "fastest_operation": min(self.metrics, key=lambda m: m.duration_ms).operation
        }
    
    def clear_metrics(self):
        """Clear performance metrics."""
        self.metrics.clear()
        logger.info("Performance metrics cleared")


class ResourceManager:
    """Resource management utilities."""
    
    def __init__(self):
        self.active_connections: Dict[str, Any] = {}
        logger.info("Initialized Resource Manager")
    
    def register_connection(self, name: str, connection: Any):
        """Register a connection."""
        self.active_connections[name] = connection
        logger.info(f"Registered connection: {name}")
    
    def get_connection(self, name: str) -> Optional[Any]:
        """Get a connection."""
        return self.active_connections.get(name)
    
    def close_connection(self, name: str):
        """Close a connection."""
        if name in self.active_connections:
            connection = self.active_connections.pop(name)
            if hasattr(connection, 'close'):
                connection.close()
            logger.info(f"Closed connection: {name}")
    
    def close_all_connections(self):
        """Close all connections."""
        for name in list(self.active_connections.keys()):
            self.close_connection(name)
    
    def get_resource_usage(self) -> Dict[str, Any]:
        """Get current resource usage."""
        process = psutil.Process()
        
        return {
            "cpu_percent": process.cpu_percent(),
            "memory_mb": process.memory_info().rss / 1024 / 1024,
            "open_files": len(process.open_files()),
            "threads": process.num_threads(),
            "connections": len(self.active_connections)
        }


class CacheOptimizer:
    """Cache optimization utilities."""
    
    def __init__(self, cache_manager):
        self.cache_manager = cache_manager
        logger.info("Initialized Cache Optimizer")
    
    def optimize_cache_size(self, target_size: int = 1000):
        """Optimize cache size."""
        current_size = self.cache_manager.size()
        
        if current_size > target_size:
            # Remove least recently used items
            entries = self.cache_manager.get_entries_info()
            entries.sort(key=lambda x: x["last_accessed"])
            
            items_to_remove = current_size - target_size
            for entry in entries[:items_to_remove]:
                self.cache_manager.delete(entry["key"])
            
            logger.info(f"Optimized cache size from {current_size} to {target_size}")
    
    def preload_frequent_items(self, items: List[str]):
        """Preload frequently accessed items."""
        for item in items:
            if not self.cache_manager.exists(item):
                # This would typically load from a data source
                self.cache_manager.set(item, f"preloaded_{item}")
        
        logger.info(f"Preloaded {len(items)} frequent items")
    
    def get_cache_efficiency(self) -> Dict[str, Any]:
        """Get cache efficiency metrics."""
        stats = self.cache_manager.get_stats()
        
        if stats["total_entries"] == 0:
            return {"efficiency": 0.0, "hit_rate": 0.0}
        
        # Calculate efficiency based on access patterns
        entries = self.cache_manager.get_entries_info()
        total_accesses = sum(entry["access_count"] for entry in entries)
        
        if total_accesses == 0:
            return {"efficiency": 0.0, "hit_rate": 0.0}
        
        avg_accesses = total_accesses / len(entries)
        efficiency = min(1.0, avg_accesses / 10.0)  # Normalize to 0-1
        
        return {
            "efficiency": efficiency,
            "hit_rate": efficiency,
            "total_entries": stats["total_entries"],
            "avg_accesses": avg_accesses
        }


class DatabaseOptimizer:
    """Database optimization utilities."""
    
    def __init__(self):
        self.query_cache: Dict[str, Any] = {}
        logger.info("Initialized Database Optimizer")
    
    def optimize_query(self, query: str, params: Dict[str, Any] = None) -> str:
        """Optimize database query."""
        # Simple query optimization
        optimized_query = query.strip()
        
        # Cache query results
        cache_key = f"{query}_{hash(str(params))}"
        if cache_key in self.query_cache:
            logger.info("Query cache hit")
            return self.query_cache[cache_key]
        
        # Store in cache
        self.query_cache[cache_key] = optimized_query
        
        return optimized_query
    
    def batch_operations(self, operations: List[Dict[str, Any]]) -> List[Any]:
        """Batch database operations."""
        # Group operations by type
        grouped_ops = {}
        for op in operations:
            op_type = op.get("type", "unknown")
            if op_type not in grouped_ops:
                grouped_ops[op_type] = []
            grouped_ops[op_type].append(op)
        
        # Execute batches
        results = []
        for op_type, ops in grouped_ops.items():
            batch_result = self._execute_batch(ops)
            results.extend(batch_result)
        
        logger.info(f"Batched {len(operations)} operations")
        return results
    
    def _execute_batch(self, operations: List[Dict[str, Any]]) -> List[Any]:
        """Execute a batch of operations."""
        # Mock implementation
        return [{"result": f"batch_{i}"} for i in range(len(operations))]


# Global performance optimizer instance
performance_optimizer = PerformanceOptimizer()
resource_manager = ResourceManager()

# Performance decorators
def measure_performance(operation_name: str):
    """Decorator to measure performance."""
    return performance_optimizer.measure_performance(operation_name)

def optimize_concurrent(max_concurrent: int = 5):
    """Decorator to optimize concurrent operations."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            return await performance_optimizer.optimize_concurrent_operations(
                [lambda: func(*args, **kwargs)], max_concurrent
            )[0]
        return wrapper
    return decorator
