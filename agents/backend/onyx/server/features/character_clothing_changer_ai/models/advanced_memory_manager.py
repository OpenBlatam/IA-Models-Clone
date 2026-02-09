"""
Advanced Memory Manager for Flux2 Clothing Changer
===================================================

Advanced memory management with intelligent allocation and cleanup.
"""

import gc
import time
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import logging
import torch

logger = logging.getLogger(__name__)


class MemoryStrategy(Enum):
    """Memory management strategy."""
    AGGRESSIVE = "aggressive"
    BALANCED = "balanced"
    CONSERVATIVE = "conservative"


@dataclass
class MemorySnapshot:
    """Memory snapshot."""
    timestamp: float
    cpu_memory_mb: float
    gpu_memory_mb: float
    gpu_memory_allocated_mb: float
    gpu_memory_reserved_mb: float
    objects_count: int
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class AdvancedMemoryManager:
    """Advanced memory management system."""
    
    def __init__(
        self,
        strategy: MemoryStrategy = MemoryStrategy.BALANCED,
        cleanup_threshold: float = 0.85,
        auto_cleanup: bool = True,
    ):
        """
        Initialize advanced memory manager.
        
        Args:
            strategy: Memory management strategy
            cleanup_threshold: Memory threshold for cleanup (0.0-1.0)
            auto_cleanup: Enable automatic cleanup
        """
        self.strategy = strategy
        self.cleanup_threshold = cleanup_threshold
        self.auto_cleanup = auto_cleanup
        self.snapshots: List[MemorySnapshot] = []
        self.cleanup_handlers: List[Callable] = []
        self.memory_history: List[Dict[str, float]] = []
    
    def register_cleanup_handler(self, handler: Callable) -> None:
        """
        Register cleanup handler.
        
        Args:
            handler: Cleanup function
        """
        self.cleanup_handlers.append(handler)
        logger.info(f"Registered cleanup handler: {handler.__name__}")
    
    def get_memory_usage(self) -> Dict[str, float]:
        """
        Get current memory usage.
        
        Returns:
            Memory usage dictionary
        """
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        cpu_memory_mb = process.memory_info().rss / 1024 / 1024
        
        gpu_memory_mb = 0.0
        gpu_allocated_mb = 0.0
        gpu_reserved_mb = 0.0
        
        if torch.cuda.is_available():
            gpu_memory_mb = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
            gpu_allocated_mb = torch.cuda.memory_allocated() / 1024 / 1024
            gpu_reserved_mb = torch.cuda.memory_reserved() / 1024 / 1024
        
        return {
            "cpu_memory_mb": cpu_memory_mb,
            "gpu_memory_mb": gpu_memory_mb,
            "gpu_allocated_mb": gpu_allocated_mb,
            "gpu_reserved_mb": gpu_reserved_mb,
            "gpu_usage_percent": (gpu_allocated_mb / gpu_memory_mb * 100) if gpu_memory_mb > 0 else 0.0,
        }
    
    def take_snapshot(self) -> MemorySnapshot:
        """
        Take memory snapshot.
        
        Returns:
            Memory snapshot
        """
        usage = self.get_memory_usage()
        
        snapshot = MemorySnapshot(
            timestamp=time.time(),
            cpu_memory_mb=usage["cpu_memory_mb"],
            gpu_memory_mb=usage["gpu_memory_mb"],
            gpu_memory_allocated_mb=usage["gpu_allocated_mb"],
            gpu_memory_reserved_mb=usage["gpu_reserved_mb"],
            objects_count=len(gc.get_objects()),
        )
        
        self.snapshots.append(snapshot)
        self.memory_history.append(usage)
        
        # Keep only last 1000 snapshots
        if len(self.snapshots) > 1000:
            self.snapshots = self.snapshots[-1000:]
        
        return snapshot
    
    def cleanup(self, force: bool = False) -> Dict[str, Any]:
        """
        Perform memory cleanup.
        
        Args:
            force: Force cleanup regardless of threshold
            
        Returns:
            Cleanup results
        """
        usage = self.get_memory_usage()
        gpu_usage = usage.get("gpu_usage_percent", 0.0) / 100.0
        
        if not force and gpu_usage < self.cleanup_threshold:
            return {"cleaned": False, "reason": "below_threshold"}
        
        results = {
            "cleaned": True,
            "before": usage.copy(),
            "handlers_executed": 0,
        }
        
        # Execute cleanup handlers
        for handler in self.cleanup_handlers:
            try:
                handler()
                results["handlers_executed"] += 1
            except Exception as e:
                logger.error(f"Cleanup handler failed: {e}")
        
        # Python garbage collection
        collected = gc.collect()
        results["gc_collected"] = collected
        
        # CUDA cache cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Get usage after cleanup
        results["after"] = self.get_memory_usage()
        results["freed_mb"] = (
            results["before"]["gpu_allocated_mb"] - 
            results["after"]["gpu_allocated_mb"]
        )
        
        logger.info(f"Memory cleanup completed: freed {results['freed_mb']:.2f} MB")
        return results
    
    def auto_cleanup_if_needed(self) -> Optional[Dict[str, Any]]:
        """
        Auto cleanup if threshold exceeded.
        
        Returns:
            Cleanup results if performed, None otherwise
        """
        if not self.auto_cleanup:
            return None
        
        usage = self.get_memory_usage()
        gpu_usage = usage.get("gpu_usage_percent", 0.0) / 100.0
        
        if gpu_usage >= self.cleanup_threshold:
            return self.cleanup()
        
        return None
    
    def get_memory_trend(self, window: int = 100) -> Dict[str, Any]:
        """
        Get memory usage trend.
        
        Args:
            window: Number of recent snapshots to analyze
            
        Returns:
            Trend analysis
        """
        if not self.memory_history:
            return {}
        
        recent = self.memory_history[-window:]
        
        cpu_values = [h["cpu_memory_mb"] for h in recent]
        gpu_values = [h["gpu_allocated_mb"] for h in recent]
        
        return {
            "cpu_avg": sum(cpu_values) / len(cpu_values),
            "cpu_max": max(cpu_values),
            "cpu_min": min(cpu_values),
            "gpu_avg": sum(gpu_values) / len(gpu_values),
            "gpu_max": max(gpu_values),
            "gpu_min": min(gpu_values),
            "samples": len(recent),
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get memory manager statistics."""
        usage = self.get_memory_usage()
        trend = self.get_memory_trend()
        
        return {
            "current_usage": usage,
            "trend": trend,
            "snapshots_count": len(self.snapshots),
            "cleanup_handlers": len(self.cleanup_handlers),
            "strategy": self.strategy.value,
            "auto_cleanup": self.auto_cleanup,
        }


