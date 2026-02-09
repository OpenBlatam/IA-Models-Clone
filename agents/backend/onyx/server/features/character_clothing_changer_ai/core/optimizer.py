"""
Performance Optimizer
====================

System for optimizing performance and resource usage.
"""

import asyncio
import logging
import gc
import os
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logger.warning("psutil not available, some optimizer features will be limited")


@dataclass
class OptimizationResult:
    """Optimization result."""
    name: str
    before: Dict[str, Any]
    after: Dict[str, Any]
    improvement: Dict[str, float]  # Percentage improvement
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "before": self.before,
            "after": self.after,
            "improvement": self.improvement,
            "timestamp": self.timestamp.isoformat()
        }


class PerformanceOptimizer:
    """Performance optimizer."""
    
    def __init__(self):
        """Initialize optimizer."""
        self.optimizations: List[OptimizationResult] = []
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get current memory statistics."""
        if not PSUTIL_AVAILABLE:
            return {
                "rss_mb": 0.0,
                "vms_mb": 0.0,
                "percent": 0.0,
                "available_mb": 0.0,
                "total_mb": 0.0
            }
        
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        return {
            "rss_mb": memory_info.rss / 1024 / 1024,
            "vms_mb": memory_info.vms / 1024 / 1024,
            "percent": process.memory_percent(),
            "available_mb": psutil.virtual_memory().available / 1024 / 1024,
            "total_mb": psutil.virtual_memory().total / 1024 / 1024
        }
    
    def get_cpu_stats(self) -> Dict[str, Any]:
        """Get current CPU statistics."""
        if not PSUTIL_AVAILABLE:
            return {
                "percent": 0.0,
                "num_threads": 0,
                "num_fds": None
            }
        
        process = psutil.Process(os.getpid())
        
        return {
            "percent": process.cpu_percent(interval=0.1),
            "num_threads": process.num_threads(),
            "num_fds": process.num_fds() if hasattr(process, 'num_fds') else None
        }
    
    def optimize_memory(self) -> OptimizationResult:
        """
        Optimize memory usage.
        
        Returns:
            Optimization result
        """
        before = self.get_memory_stats()
        
        # Force garbage collection
        collected = gc.collect()
        
        # Get after stats
        after = self.get_memory_stats()
        
        # Calculate improvement
        improvement = {
            "rss_mb": ((before["rss_mb"] - after["rss_mb"]) / before["rss_mb"] * 100) if before["rss_mb"] > 0 else 0,
            "vms_mb": ((before["vms_mb"] - after["vms_mb"]) / before["vms_mb"] * 100) if before["vms_mb"] > 0 else 0,
            "collected_objects": collected
        }
        
        result = OptimizationResult(
            name="memory_optimization",
            before=before,
            after=after,
            improvement=improvement
        )
        
        self.optimizations.append(result)
        logger.info(f"Memory optimization: freed {collected} objects, RSS: {before['rss_mb']:.1f}MB -> {after['rss_mb']:.1f}MB")
        
        return result
    
    def optimize_cache(self, cache_manager) -> OptimizationResult:
        """
        Optimize cache usage.
        
        Args:
            cache_manager: Cache manager instance
            
        Returns:
            Optimization result
        """
        before = {
            "size": cache_manager.get_size() if hasattr(cache_manager, 'get_size') else 0,
            "hits": cache_manager.stats.get("hits", 0) if hasattr(cache_manager, 'stats') else 0,
            "misses": cache_manager.stats.get("misses", 0) if hasattr(cache_manager, 'stats') else 0
        }
        
        # Cleanup expired entries
        cleaned = 0
        if hasattr(cache_manager, 'cleanup'):
            cleaned = cache_manager.cleanup()
        
        after = {
            "size": cache_manager.get_size() if hasattr(cache_manager, 'get_size') else 0,
            "hits": cache_manager.stats.get("hits", 0) if hasattr(cache_manager, 'stats') else 0,
            "misses": cache_manager.stats.get("misses", 0) if hasattr(cache_manager, 'stats') else 0
        }
        
        improvement = {
            "size_reduction": ((before["size"] - after["size"]) / before["size"] * 100) if before["size"] > 0 else 0,
            "cleaned_entries": cleaned
        }
        
        result = OptimizationResult(
            name="cache_optimization",
            before=before,
            after=after,
            improvement=improvement
        )
        
        self.optimizations.append(result)
        logger.info(f"Cache optimization: cleaned {cleaned} entries, size: {before['size']} -> {after['size']}")
        
        return result
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        return {
            "memory": self.get_memory_stats(),
            "cpu": self.get_cpu_stats(),
            "timestamp": datetime.now().isoformat()
        }
    
    def get_optimization_history(self) -> List[OptimizationResult]:
        """Get optimization history."""
        return self.optimizations.copy()
    
    def clear_history(self):
        """Clear optimization history."""
        self.optimizations.clear()


class ResourceMonitor:
    """Monitor system resources."""
    
    def __init__(self, interval: float = 60.0):
        """
        Initialize resource monitor.
        
        Args:
            interval: Monitoring interval in seconds
        """
        self.interval = interval
        self.monitoring = False
        self.stats: List[Dict[str, Any]] = []
        self._task: Optional[asyncio.Task] = None
        self.optimizer = PerformanceOptimizer()
    
    async def start(self):
        """Start monitoring."""
        if self.monitoring:
            return
        
        self.monitoring = True
        self._task = asyncio.create_task(self._monitor_loop())
        logger.info("Resource monitoring started")
    
    async def stop(self):
        """Stop monitoring."""
        self.monitoring = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Resource monitoring stopped")
    
    async def _monitor_loop(self):
        """Monitoring loop."""
        while self.monitoring:
            try:
                stats = self.optimizer.get_system_stats()
                self.stats.append(stats)
                
                # Keep only last 1000 entries
                if len(self.stats) > 1000:
                    self.stats = self.stats[-1000:]
                
                # Auto-optimize if memory usage is high
                if PSUTIL_AVAILABLE and stats["memory"]["percent"] > 80:
                    logger.warning(f"High memory usage: {stats['memory']['percent']:.1f}%")
                    self.optimizer.optimize_memory()
                
                await asyncio.sleep(self.interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(self.interval)
    
    def get_recent_stats(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent statistics."""
        return self.stats[-limit:]
    
    def get_average_stats(self) -> Dict[str, Any]:
        """Get average statistics."""
        if not self.stats:
            return {}
        
        memory_values = [s["memory"]["rss_mb"] for s in self.stats]
        cpu_values = [s["cpu"]["percent"] for s in self.stats]
        
        return {
            "avg_memory_mb": sum(memory_values) / len(memory_values),
            "avg_cpu_percent": sum(cpu_values) / len(cpu_values),
            "max_memory_mb": max(memory_values),
            "max_cpu_percent": max(cpu_values),
            "sample_count": len(self.stats)
        }

