"""
Memory Optimizer
Memory usage optimization and garbage collection tuning
"""

import logging
import gc
from typing import Any, Dict
import sys

logger = logging.getLogger(__name__)


class MemoryOptimizer:
    """
    Memory optimization utilities
    
    Features:
    - Garbage collection tuning
    - Memory profiling
    - Object pooling
    - Memory-efficient data structures
    """
    
    def __init__(self):
        self._gc_thresholds = (700, 10, 10)  # Default thresholds
        self._object_pools: Dict[str, list] = {}
    
    def optimize_gc(self) -> None:
        """Optimize garbage collection settings"""
        # Set GC thresholds for better performance
        gc.set_threshold(700, 10, 10)
        
        # Disable GC in tight loops (re-enable manually)
        # gc.disable()  # Only in specific scenarios
        
        logger.info("Garbage collection optimized")
    
    def force_gc(self) -> None:
        """Force garbage collection"""
        collected = gc.collect()
        logger.debug(f"Garbage collected: {collected} objects")
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage"""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            
            return {
                "rss_mb": memory_info.rss / 1024 / 1024,  # Resident Set Size
                "vms_mb": memory_info.vms / 1024 / 1024,  # Virtual Memory Size
                "percent": process.memory_percent()
            }
        except ImportError:
            return {"error": "psutil not available"}
    
    def create_object_pool(self, pool_name: str, factory: callable, size: int = 10) -> None:
        """Create object pool for reuse"""
        self._object_pools[pool_name] = [factory() for _ in range(size)]
        logger.info(f"Created object pool: {pool_name} (size: {size})")
    
    def get_from_pool(self, pool_name: str):
        """Get object from pool"""
        if pool_name in self._object_pools and self._object_pools[pool_name]:
            return self._object_pools[pool_name].pop()
        return None
    
    def return_to_pool(self, pool_name: str, obj: Any) -> None:
        """Return object to pool"""
        if pool_name in self._object_pools:
            self._object_pools[pool_name].append(obj)


class SlotsMixin:
    """Mixin to use __slots__ for memory optimization"""
    __slots__ = ()


# Global optimizer
_memory_optimizer: MemoryOptimizer = None


def get_memory_optimizer() -> MemoryOptimizer:
    """Get global memory optimizer"""
    global _memory_optimizer
    if _memory_optimizer is None:
        _memory_optimizer = MemoryOptimizer()
        _memory_optimizer.optimize_gc()
    return _memory_optimizer















