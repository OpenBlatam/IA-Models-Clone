"""
Performance Optimizer
====================

Advanced performance optimizations for upscaling operations.
"""

import logging
import time
import asyncio
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from collections import deque
import statistics

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for optimization."""
    operation_time: float
    memory_usage_mb: float
    gpu_utilization: float = 0.0
    cpu_utilization: float = 0.0
    cache_hit_rate: float = 0.0
    throughput: float = 0.0  # Images per second


class PerformanceOptimizer:
    """
    Performance optimizer that learns and adapts.
    
    Features:
    - Automatic parameter tuning
    - Performance monitoring
    - Adaptive batch sizing
    - Resource optimization
    - Throughput maximization
    """
    
    def __init__(
        self,
        target_throughput: float = 1.0,  # Images per second
        max_memory_mb: Optional[float] = None,
        history_size: int = 100
    ):
        """
        Initialize performance optimizer.
        
        Args:
            target_throughput: Target images per second
            max_memory_mb: Maximum memory usage in MB
            history_size: Size of performance history
        """
        self.target_throughput = target_throughput
        self.max_memory_mb = max_memory_mb
        self.history_size = history_size
        
        # Performance history
        self.metrics_history: deque = deque(maxlen=history_size)
        self.operation_times: deque = deque(maxlen=history_size)
        
        # Current settings
        self.optimal_batch_size = 1
        self.optimal_tile_size = 512
        self.optimal_concurrency = 2
        
        logger.info("PerformanceOptimizer initialized")
    
    def record_operation(
        self,
        operation_time: float,
        memory_usage_mb: float = 0.0,
        gpu_utilization: float = 0.0,
        cpu_utilization: float = 0.0,
        cache_hit: bool = False
    ) -> None:
        """
        Record operation metrics.
        
        Args:
            operation_time: Operation time in seconds
            memory_usage_mb: Memory usage in MB
            gpu_utilization: GPU utilization (0.0-1.0)
            cpu_utilization: CPU utilization (0.0-1.0)
            cache_hit: Whether cache was hit
        """
        # Calculate cache hit rate
        cache_hits = sum(1 for m in self.metrics_history if getattr(m, 'cache_hit', False))
        cache_hit_rate = cache_hits / max(1, len(self.metrics_history))
        
        # Calculate throughput
        throughput = 1.0 / operation_time if operation_time > 0 else 0.0
        
        metrics = PerformanceMetrics(
            operation_time=operation_time,
            memory_usage_mb=memory_usage_mb,
            gpu_utilization=gpu_utilization,
            cpu_utilization=cpu_utilization,
            cache_hit_rate=cache_hit_rate,
            throughput=throughput
        )
        
        self.metrics_history.append(metrics)
        self.operation_times.append(operation_time)
        
        # Update optimal settings
        self._update_optimal_settings()
    
    def _update_optimal_settings(self) -> None:
        """Update optimal settings based on history."""
        if len(self.operation_times) < 5:
            return
        
        # Calculate statistics
        avg_time = statistics.mean(self.operation_times)
        median_time = statistics.median(self.operation_times)
        std_time = statistics.stdev(self.operation_times) if len(self.operation_times) > 1 else 0
        
        # Adjust batch size based on throughput
        current_throughput = 1.0 / avg_time if avg_time > 0 else 0.0
        
        if current_throughput < self.target_throughput * 0.8:
            # Too slow, reduce batch size or increase concurrency
            if self.optimal_batch_size > 1:
                self.optimal_batch_size = max(1, self.optimal_batch_size - 1)
            else:
                self.optimal_concurrency = min(4, self.optimal_concurrency + 1)
        elif current_throughput > self.target_throughput * 1.2:
            # Too fast, can increase batch size
            self.optimal_batch_size = min(8, self.optimal_batch_size + 1)
        
        # Adjust tile size based on memory
        if self.max_memory_mb:
            avg_memory = statistics.mean([m.memory_usage_mb for m in self.metrics_history])
            if avg_memory > self.max_memory_mb * 0.9:
                # High memory usage, reduce tile size
                self.optimal_tile_size = max(256, self.optimal_tile_size - 64)
            elif avg_memory < self.max_memory_mb * 0.5:
                # Low memory usage, can increase tile size
                self.optimal_tile_size = min(1024, self.optimal_tile_size + 64)
        
        logger.debug(
            f"Optimal settings: batch={self.optimal_batch_size}, "
            f"tile={self.optimal_tile_size}, concurrency={self.optimal_concurrency}"
        )
    
    def get_optimal_batch_size(self) -> int:
        """Get optimal batch size."""
        return self.optimal_batch_size
    
    def get_optimal_tile_size(self) -> int:
        """Get optimal tile size."""
        return self.optimal_tile_size
    
    def get_optimal_concurrency(self) -> int:
        """Get optimal concurrency level."""
        return self.optimal_concurrency
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get performance statistics."""
        if not self.metrics_history:
            return {
                "total_operations": 0,
                "avg_throughput": 0.0,
                "avg_operation_time": 0.0,
                "cache_hit_rate": 0.0,
            }
        
        return {
            "total_operations": len(self.metrics_history),
            "avg_throughput": statistics.mean([m.throughput for m in self.metrics_history]),
            "avg_operation_time": statistics.mean([m.operation_time for m in self.metrics_history]),
            "median_operation_time": statistics.median([m.operation_time for m in self.metrics_history]),
            "cache_hit_rate": statistics.mean([m.cache_hit_rate for m in self.metrics_history]),
            "avg_memory_mb": statistics.mean([m.memory_usage_mb for m in self.metrics_history]),
            "optimal_batch_size": self.optimal_batch_size,
            "optimal_tile_size": self.optimal_tile_size,
            "optimal_concurrency": self.optimal_concurrency,
        }
    
    def reset(self) -> None:
        """Reset optimizer state."""
        self.metrics_history.clear()
        self.operation_times.clear()
        self.optimal_batch_size = 1
        self.optimal_tile_size = 512
        self.optimal_concurrency = 2
        logger.info("PerformanceOptimizer reset")


