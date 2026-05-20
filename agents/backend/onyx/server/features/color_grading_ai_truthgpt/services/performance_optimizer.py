"""
Performance Optimizer for Color Grading AI
==========================================

Optimizes processing performance with caching, batching, and resource management.
"""

import logging
import asyncio
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime, timedelta
import psutil
import os

logger = logging.getLogger(__name__)


@dataclass
class SystemResources:
    """System resource information."""
    cpu_percent: float
    memory_percent: float
    memory_available: int  # bytes
    disk_usage_percent: float
    active_processes: int
    timestamp: datetime


class PerformanceOptimizer:
    """
    Optimizes performance.
    
    Features:
    - Resource monitoring
    - Adaptive processing
    - Memory management
    - CPU throttling
    """
    
    def __init__(self):
        """Initialize performance optimizer."""
        self._resource_history: List[SystemResources] = []
        self._max_history = 100
        self._cpu_threshold = 80.0
        self._memory_threshold = 85.0
    
    def get_system_resources(self) -> SystemResources:
        """Get current system resources."""
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage("/")
        
        return SystemResources(
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_available=memory.available,
            disk_usage_percent=disk.percent,
            active_processes=len(psutil.pids()),
            timestamp=datetime.now()
        )
    
    def should_throttle(self) -> bool:
        """
        Check if processing should be throttled.
        
        Returns:
            True if should throttle
        """
        resources = self.get_system_resources()
        
        # Add to history
        self._resource_history.append(resources)
        if len(self._resource_history) > self._max_history:
            self._resource_history.pop(0)
        
        # Check thresholds
        if resources.cpu_percent > self._cpu_threshold:
            logger.warning(f"High CPU usage: {resources.cpu_percent}%")
            return True
        
        if resources.memory_percent > self._memory_threshold:
            logger.warning(f"High memory usage: {resources.memory_percent}%")
            return True
        
        return False
    
    def get_optimal_workers(self, base_workers: int = 3) -> int:
        """
        Get optimal number of workers based on system resources.
        
        Args:
            base_workers: Base number of workers
            
        Returns:
            Optimal number of workers
        """
        resources = self.get_system_resources()
        
        # Adjust based on CPU
        cpu_factor = max(0.5, min(1.5, 100 / max(resources.cpu_percent, 1)))
        
        # Adjust based on memory
        memory_factor = max(0.5, min(1.5, 100 / max(resources.memory_percent, 1)))
        
        # Calculate optimal
        optimal = int(base_workers * cpu_factor * memory_factor)
        
        return max(1, min(optimal, base_workers * 2))  # Cap at 2x base
    
    def get_resource_stats(self) -> Dict[str, Any]:
        """Get resource statistics."""
        if not self._resource_history:
            return {}
        
        recent = self._resource_history[-10:]  # Last 10 samples
        
        return {
            "current": {
                "cpu_percent": recent[-1].cpu_percent,
                "memory_percent": recent[-1].memory_percent,
                "memory_available_gb": recent[-1].memory_available / (1024**3),
            },
            "average": {
                "cpu_percent": sum(r.cpu_percent for r in recent) / len(recent),
                "memory_percent": sum(r.memory_percent for r in recent) / len(recent),
            },
            "max": {
                "cpu_percent": max(r.cpu_percent for r in recent),
                "memory_percent": max(r.memory_percent for r in recent),
            },
        }
    
    async def wait_if_throttled(self, delay: float = 1.0):
        """Wait if system is throttled."""
        if self.should_throttle():
            logger.info(f"Throttling: waiting {delay}s")
            await asyncio.sleep(delay)
    
    def cleanup_old_cache(self, cache_dir: str, max_age_days: int = 7):
        """
        Cleanup old cache files.
        
        Args:
            cache_dir: Cache directory
            max_age_days: Maximum age in days
        """
        from pathlib import Path
        cache_path = Path(cache_dir)
        if not cache_path.exists():
            return
        
        cutoff = datetime.now() - timedelta(days=max_age_days)
        deleted = 0
        
        for cache_file in cache_path.glob("*"):
            if cache_file.is_file():
                mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
                if mtime < cutoff:
                    cache_file.unlink()
                    deleted += 1
        
        if deleted > 0:
            logger.info(f"Cleaned up {deleted} old cache files")




