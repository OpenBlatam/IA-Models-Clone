"""
Memory Optimization Utilities
==============================

Advanced memory management and optimization.
"""

import gc
import sys
import logging
import psutil
import tracemalloc
from typing import Dict, Any, Optional, List
from datetime import datetime
import weakref

logger = logging.getLogger(__name__)

class MemoryOptimizer:
    """Advanced memory optimizer with tracking and cleanup."""
    
    def __init__(self):
        self.tracemalloc_enabled = False
        self.memory_snapshots = []
        self.peak_memory = 0
        self.cleanup_threshold_mb = 500.0
    
    def start_tracking(self):
        """Start memory tracking."""
        if not self.tracemalloc_enabled:
            tracemalloc.start()
            self.tracemalloc_enabled = True
            logger.info("Memory tracking started")
    
    def stop_tracking(self):
        """Stop memory tracking."""
        if self.tracemalloc_enabled:
            tracemalloc.stop()
            self.tracemalloc_enabled = False
            logger.info("Memory tracking stopped")
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage."""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        current_mb = memory_info.rss / (1024 * 1024)
        if current_mb > self.peak_memory:
            self.peak_memory = current_mb
        
        return {
            "rss_mb": round(current_mb, 2),
            "vms_mb": round(memory_info.vms / (1024 * 1024), 2),
            "peak_mb": round(self.peak_memory, 2),
            "percent": process.memory_percent(),
            "available_mb": round(psutil.virtual_memory().available / (1024 * 1024), 2),
            "total_mb": round(psutil.virtual_memory().total / (1024 * 1024), 2)
        }
    
    def get_tracemalloc_stats(self) -> Optional[Dict[str, Any]]:
        """Get tracemalloc statistics."""
        if not self.tracemalloc_enabled:
            return None
        
        try:
            snapshot = tracemalloc.take_snapshot()
            top_stats = snapshot.statistics('lineno')
            
            total_size = sum(stat.size for stat in top_stats)
            
            return {
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "top_10": [
                    {
                        "file": stat.traceback[0].filename,
                        "line": stat.traceback[0].lineno,
                        "size_mb": round(stat.size / (1024 * 1024), 2),
                        "count": stat.count
                    }
                    for stat in top_stats[:10]
                ]
            }
        except Exception as e:
            logger.error(f"Failed to get tracemalloc stats: {e}")
            return None
    
    def optimize_memory(self, aggressive: bool = False) -> Dict[str, Any]:
        """Optimize memory usage."""
        before = self.get_memory_usage()
        
        # Force garbage collection
        collected = gc.collect()
        
        if aggressive:
            # Multiple GC passes
            for _ in range(3):
                gc.collect()
            
            # Clear caches if possible
            try:
                import sys
                for module in list(sys.modules.values()):
                    if hasattr(module, '__dict__'):
                        for key in list(module.__dict__.keys()):
                            if key.startswith('_cache'):
                                delattr(module, key)
            except Exception as e:
                logger.warning(f"Failed to clear module caches: {e}")
        
        after = self.get_memory_usage()
        
        freed_mb = before["rss_mb"] - after["rss_mb"]
        
        logger.info(f"Memory optimization: freed {freed_mb:.2f} MB, collected {collected} objects")
        
        return {
            "before_mb": before["rss_mb"],
            "after_mb": after["rss_mb"],
            "freed_mb": round(freed_mb, 2),
            "collected_objects": collected,
            "aggressive": aggressive
        }
    
    def should_cleanup(self) -> bool:
        """Check if memory cleanup is needed."""
        usage = self.get_memory_usage()
        return usage["rss_mb"] > self.cleanup_threshold_mb
    
    def auto_cleanup(self):
        """Automatic memory cleanup if needed."""
        if self.should_cleanup():
            logger.info("Auto cleanup triggered")
            return self.optimize_memory(aggressive=False)
        return None
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics."""
        stats = {
            "current": self.get_memory_usage(),
            "tracemalloc": self.get_tracemalloc_stats(),
            "gc_stats": {
                "collections": {
                    str(gen): count
                    for gen, count in enumerate(gc.get_stats())
                },
                "thresholds": gc.get_threshold(),
                "counts": gc.get_count()
            },
            "timestamp": datetime.now().isoformat()
        }
        
        return stats

# Global instance
memory_optimizer = MemoryOptimizer()
































