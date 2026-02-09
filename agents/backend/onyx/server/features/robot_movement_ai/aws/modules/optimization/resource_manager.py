"""
Resource Manager
================

Advanced resource management and optimization.
"""

import logging
import asyncio
import psutil
import os
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class ResourceLimits:
    """Resource limits."""
    cpu_percent: float = 80.0
    memory_percent: float = 80.0
    memory_mb: Optional[int] = None
    connections: int = 1000
    file_descriptors: int = 10000


class ResourceManager:
    """Resource manager with monitoring and optimization."""
    
    def __init__(self, limits: Optional[ResourceLimits] = None):
        self.limits = limits or ResourceLimits()
        self._monitoring = False
        self._monitor_task: Optional[asyncio.Task] = None
        self._stats_history: List[Dict[str, Any]] = []
    
    def start_monitoring(self, interval: float = 5.0):
        """Start resource monitoring."""
        if self._monitoring:
            return
        
        self._monitoring = True
        
        async def monitor():
            while self._monitoring:
                stats = self.get_current_stats()
                self._stats_history.append({
                    "timestamp": datetime.now().isoformat(),
                    **stats
                })
                
                # Keep only last 100 entries
                if len(self._stats_history) > 100:
                    self._stats_history.pop(0)
                
                # Check limits
                self._check_limits(stats)
                
                await asyncio.sleep(interval)
        
        self._monitor_task = asyncio.create_task(monitor())
        logger.info("Resource monitoring started")
    
    def stop_monitoring(self):
        """Stop resource monitoring."""
        self._monitoring = False
        if self._monitor_task:
            self._monitor_task.cancel()
        logger.info("Resource monitoring stopped")
    
    def get_current_stats(self) -> Dict[str, Any]:
        """Get current resource statistics."""
        process = psutil.Process(os.getpid())
        
        # CPU
        cpu_percent = process.cpu_percent(interval=0.1)
        cpu_count = os.cpu_count() or 1
        
        # Memory
        mem_info = process.memory_info()
        mem_percent = process.memory_percent()
        system_mem = psutil.virtual_memory()
        
        # Network
        net_io = psutil.net_io_counters()
        
        # Connections
        connections = len(process.connections())
        
        return {
            "cpu_percent": cpu_percent,
            "cpu_cores": cpu_count,
            "memory_rss_mb": mem_info.rss / 1024 / 1024,
            "memory_vms_mb": mem_info.vms / 1024 / 1024,
            "memory_percent": mem_percent,
            "memory_available_mb": system_mem.available / 1024 / 1024,
            "network_bytes_sent_mb": net_io.bytes_sent / 1024 / 1024,
            "network_bytes_recv_mb": net_io.bytes_recv / 1024 / 1024,
            "connections": connections,
            "threads": process.num_threads(),
            "open_files": len(process.open_files())
        }
    
    def _check_limits(self, stats: Dict[str, Any]):
        """Check resource limits and log warnings."""
        warnings = []
        
        if stats["cpu_percent"] > self.limits.cpu_percent:
            warnings.append(f"CPU usage high: {stats['cpu_percent']:.1f}%")
        
        if stats["memory_percent"] > self.limits.memory_percent:
            warnings.append(f"Memory usage high: {stats['memory_percent']:.1f}%")
        
        if self.limits.memory_mb and stats["memory_rss_mb"] > self.limits.memory_mb:
            warnings.append(f"Memory usage high: {stats['memory_rss_mb']:.1f}MB")
        
        if stats["connections"] > self.limits.connections:
            warnings.append(f"Connections high: {stats['connections']}")
        
        if warnings:
            logger.warning(f"Resource limits exceeded: {', '.join(warnings)}")
    
    def optimize_resources(self):
        """Optimize resource usage."""
        # Force garbage collection
        import gc
        collected = gc.collect()
        
        # Clear stats history if too large
        if len(self._stats_history) > 1000:
            self._stats_history = self._stats_history[-500:]
        
        logger.info(f"Resources optimized: {collected} objects collected")
        return collected
    
    def get_stats_summary(self, last_minutes: int = 5) -> Dict[str, Any]:
        """Get resource statistics summary."""
        cutoff = datetime.now() - timedelta(minutes=last_minutes)
        
        recent_stats = [
            s for s in self._stats_history
            if datetime.fromisoformat(s["timestamp"]) > cutoff
        ]
        
        if not recent_stats:
            return {}
        
        # Calculate averages
        avg_cpu = sum(s["cpu_percent"] for s in recent_stats) / len(recent_stats)
        avg_memory = sum(s["memory_percent"] for s in recent_stats) / len(recent_stats)
        max_memory = max(s["memory_rss_mb"] for s in recent_stats)
        avg_connections = sum(s["connections"] for s in recent_stats) / len(recent_stats)
        
        return {
            "period_minutes": last_minutes,
            "samples": len(recent_stats),
            "avg_cpu_percent": avg_cpu,
            "avg_memory_percent": avg_memory,
            "max_memory_mb": max_memory,
            "avg_connections": avg_connections
        }















