"""
Memory Manager for Piel Mejorador AI SAM3
=========================================

Advanced memory management and optimization.
"""

import gc
import logging
import sys
import tracemalloc
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class MemorySnapshot:
    """Memory snapshot."""
    timestamp: datetime
    current_mb: float
    peak_mb: float
    limit_mb: Optional[float] = None


class MemoryManager:
    """
    Advanced memory management.
    
    Features:
    - Memory tracking
    - Automatic cleanup
    - Memory limits
    - Leak detection
    - Optimization recommendations
    """
    
    def __init__(self, max_memory_mb: Optional[float] = None):
        """
        Initialize memory manager.
        
        Args:
            max_memory_mb: Maximum memory limit in MB
        """
        self.max_memory_mb = max_memory_mb
        self._snapshots: List[MemorySnapshot] = []
        self._tracemalloc_enabled = False
        
        # Enable tracemalloc if available
        try:
            tracemalloc.start()
            self._tracemalloc_enabled = True
            logger.info("Tracemalloc enabled for memory tracking")
        except Exception as e:
            logger.warning(f"Could not enable tracemalloc: {e}")
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """
        Get current memory usage.
        
        Returns:
            Memory usage dictionary
        """
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        current_mb = memory_info.rss / (1024 * 1024)
        
        # Get tracemalloc stats if available
        tracemalloc_stats = {}
        if self._tracemalloc_enabled:
            try:
                current, peak = tracemalloc.get_traced_memory()
                tracemalloc_stats = {
                    "traced_current_mb": current / (1024 * 1024),
                    "traced_peak_mb": peak / (1024 * 1024),
                }
            except Exception:
                pass
        
        return {
            "current_mb": current_mb,
            "current_percent": process.memory_percent(),
            "max_memory_mb": self.max_memory_mb,
            "limit_exceeded": self.max_memory_mb and current_mb > self.max_memory_mb,
            **tracemalloc_stats,
        }
    
    def take_snapshot(self) -> MemorySnapshot:
        """
        Take a memory snapshot.
        
        Returns:
            MemorySnapshot
        """
        usage = self.get_memory_usage()
        
        snapshot = MemorySnapshot(
            timestamp=datetime.now(),
            current_mb=usage["current_mb"],
            peak_mb=usage.get("traced_peak_mb", usage["current_mb"]),
            limit_mb=self.max_memory_mb,
        )
        
        self._snapshots.append(snapshot)
        
        # Keep only last 100 snapshots
        if len(self._snapshots) > 100:
            self._snapshots.pop(0)
        
        return snapshot
    
    def optimize_memory(self, force: bool = False) -> Dict[str, Any]:
        """
        Optimize memory usage.
        
        Args:
            force: Force optimization even if not needed
            
        Returns:
            Optimization results
        """
        before = self.get_memory_usage()
        
        # Force garbage collection
        collected = gc.collect()
        
        # Collect generation 2 (oldest objects)
        gen2_before = len(gc.get_objects())
        gc.collect(2)
        gen2_after = len(gc.get_objects())
        
        after = self.get_memory_usage()
        
        freed_mb = before["current_mb"] - after["current_mb"]
        
        result = {
            "before_mb": before["current_mb"],
            "after_mb": after["current_mb"],
            "freed_mb": freed_mb,
            "collected_objects": collected,
            "gen2_objects_freed": gen2_before - gen2_after,
        }
        
        logger.info(
            f"Memory optimization: freed {freed_mb:.2f}MB, "
            f"collected {collected} objects"
        )
        
        return result
    
    def check_memory_limit(self) -> bool:
        """
        Check if memory limit is exceeded.
        
        Returns:
            True if limit exceeded
        """
        if not self.max_memory_mb:
            return False
        
        usage = self.get_memory_usage()
        return usage["limit_exceeded"]
    
    def get_top_memory_consumers(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get top memory consumers (requires tracemalloc).
        
        Args:
            limit: Number of top consumers to return
            
        Returns:
            List of memory consumers
        """
        if not self._tracemalloc_enabled:
            return []
        
        try:
            snapshot = tracemalloc.take_snapshot()
            top_stats = snapshot.statistics('lineno')
            
            consumers = []
            for index, stat in enumerate(top_stats[:limit], 1):
                consumers.append({
                    "rank": index,
                    "filename": stat.traceback[0].filename,
                    "line": stat.traceback[0].lineno,
                    "size_mb": stat.size / (1024 * 1024),
                    "count": stat.count,
                })
            
            return consumers
        except Exception as e:
            logger.warning(f"Could not get memory consumers: {e}")
            return []
    
    def get_memory_trend(self) -> Dict[str, Any]:
        """
        Get memory usage trend.
        
        Returns:
            Trend analysis
        """
        if len(self._snapshots) < 2:
            return {"trend": "insufficient_data"}
        
        recent = self._snapshots[-10:]  # Last 10 snapshots
        first = recent[0]
        last = recent[-1]
        
        change_mb = last.current_mb - first.current_mb
        change_percent = (change_mb / first.current_mb * 100) if first.current_mb > 0 else 0
        
        trend = "stable"
        if change_percent > 10:
            trend = "increasing"
        elif change_percent < -10:
            trend = "decreasing"
        
        return {
            "trend": trend,
            "change_mb": change_mb,
            "change_percent": change_percent,
            "snapshots_count": len(self._snapshots),
        }
    
    def cleanup(self):
        """Cleanup memory manager resources."""
        if self._tracemalloc_enabled:
            try:
                tracemalloc.stop()
            except Exception:
                pass




