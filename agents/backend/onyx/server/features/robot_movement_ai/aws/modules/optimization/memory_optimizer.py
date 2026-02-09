"""
Memory Optimizer
================

Advanced memory optimization techniques.
"""

import logging
import gc
import sys
import tracemalloc
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import weakref

logger = logging.getLogger(__name__)


@dataclass
class MemoryStats:
    """Memory statistics."""
    current: int
    peak: int
    allocated: int
    freed: int


class MemoryOptimizer:
    """Memory optimizer with advanced techniques."""
    
    def __init__(self):
        self._tracking = False
        self._snapshots: List[Any] = []
        self._weak_refs: List[weakref.ref] = []
    
    def start_tracking(self):
        """Start memory tracking."""
        tracemalloc.start()
        self._tracking = True
        logger.info("Memory tracking started")
    
    def stop_tracking(self):
        """Stop memory tracking."""
        if self._tracking:
            tracemalloc.stop()
            self._tracking = False
            logger.info("Memory tracking stopped")
    
    def take_snapshot(self) -> Optional[Any]:
        """Take memory snapshot."""
        if not self._tracking:
            return None
        
        snapshot = tracemalloc.take_snapshot()
        self._snapshots.append(snapshot)
        return snapshot
    
    def get_memory_stats(self) -> MemoryStats:
        """Get current memory statistics."""
        if not self._tracking:
            return MemoryStats(0, 0, 0, 0)
        
        current, peak = tracemalloc.get_traced_memory()
        
        return MemoryStats(
            current=current,
            peak=peak,
            allocated=current,
            freed=0
        )
    
    def optimize_memory(self):
        """Optimize memory usage."""
        # Force garbage collection
        collected = gc.collect()
        
        # Clear weak references
        self._weak_refs = [ref for ref in self._weak_refs if ref() is not None]
        
        logger.info(f"Memory optimized: {collected} objects collected")
        return collected
    
    def get_top_memory_consumers(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get top memory consumers."""
        if not self._snapshots:
            return []
        
        snapshot = self._snapshots[-1]
        top_stats = snapshot.statistics('lineno')
        
        consumers = []
        for index, stat in enumerate(top_stats[:limit], 1):
            consumers.append({
                "rank": index,
                "filename": stat.traceback[0].filename,
                "line": stat.traceback[0].lineno,
                "size": stat.size,
                "size_mb": stat.size / 1024 / 1024,
                "count": stat.count
            })
        
        return consumers
    
    def clear_snapshots(self):
        """Clear memory snapshots."""
        self._snapshots.clear()
        logger.debug("Memory snapshots cleared")
    
    def register_weak_ref(self, obj: Any):
        """Register weak reference for automatic cleanup."""
        ref = weakref.ref(obj)
        self._weak_refs.append(ref)
        return ref
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        
        return {
            "rss": mem_info.rss,  # Resident Set Size
            "rss_mb": mem_info.rss / 1024 / 1024,
            "vms": mem_info.vms,  # Virtual Memory Size
            "vms_mb": mem_info.vms / 1024 / 1024,
            "percent": process.memory_percent(),
            "available_mb": psutil.virtual_memory().available / 1024 / 1024
        }















