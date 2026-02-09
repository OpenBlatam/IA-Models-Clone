"""
Performance Monitor
===================

Real-time performance monitoring.
"""

import logging
import time
import psutil
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics."""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    disk_io_read_mb: float
    disk_io_write_mb: float
    network_sent_mb: float
    network_recv_mb: float


class PerformanceMonitor:
    """Real-time performance monitor."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self._metrics: deque = deque(maxlen=window_size)
        self._monitoring = False
        self._process = psutil.Process()
        self._last_disk_io = None
        self._last_network_io = None
    
    def start_monitoring(self, interval: float = 1.0):
        """Start performance monitoring."""
        if self._monitoring:
            return
        
        self._monitoring = True
        
        async def monitor():
            while self._monitoring:
                metrics = self._collect_metrics()
                self._metrics.append(metrics)
                await asyncio.sleep(interval)
        
        asyncio.create_task(monitor())
        logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop performance monitoring."""
        self._monitoring = False
        logger.info("Performance monitoring stopped")
    
    def _collect_metrics(self) -> PerformanceMetrics:
        """Collect current metrics."""
        cpu_percent = self._process.cpu_percent()
        memory_info = self._process.memory_info()
        memory_percent = self._process.memory_percent()
        
        # Disk I/O
        disk_io = psutil.disk_io_counters()
        disk_read = (disk_io.read_bytes / 1024 / 1024) if disk_io else 0
        disk_write = (disk_io.write_bytes / 1024 / 1024) if disk_io else 0
        
        if self._last_disk_io:
            disk_read_diff = disk_read - self._last_disk_io[0]
            disk_write_diff = disk_write - self._last_disk_io[1]
        else:
            disk_read_diff = 0
            disk_write_diff = 0
        
        self._last_disk_io = (disk_read, disk_write)
        
        # Network I/O
        network_io = psutil.net_io_counters()
        net_sent = (network_io.bytes_sent / 1024 / 1024) if network_io else 0
        net_recv = (network_io.bytes_recv / 1024 / 1024) if network_io else 0
        
        if self._last_network_io:
            net_sent_diff = net_sent - self._last_network_io[0]
            net_recv_diff = net_recv - self._last_network_io[1]
        else:
            net_sent_diff = 0
            net_recv_diff = 0
        
        self._last_network_io = (net_sent, net_recv)
        
        return PerformanceMetrics(
            timestamp=datetime.now(),
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            memory_used_mb=memory_info.rss / 1024 / 1024,
            memory_available_mb=psutil.virtual_memory().available / 1024 / 1024,
            disk_io_read_mb=disk_read_diff,
            disk_io_write_mb=disk_write_diff,
            network_sent_mb=net_sent_diff,
            network_recv_mb=net_recv_diff
        )
    
    def get_current_metrics(self) -> Optional[PerformanceMetrics]:
        """Get current metrics."""
        if not self._metrics:
            return None
        return self._metrics[-1]
    
    def get_metrics_history(self, limit: int = 100) -> List[PerformanceMetrics]:
        """Get metrics history."""
        return list(self._metrics)[-limit:]
    
    def get_average_metrics(self) -> Optional[Dict[str, float]]:
        """Get average metrics."""
        if not self._metrics:
            return None
        
        metrics_list = list(self._metrics)
        
        return {
            "avg_cpu_percent": sum(m.cpu_percent for m in metrics_list) / len(metrics_list),
            "avg_memory_percent": sum(m.memory_percent for m in metrics_list) / len(metrics_list),
            "avg_memory_used_mb": sum(m.memory_used_mb for m in metrics_list) / len(metrics_list),
            "max_cpu_percent": max(m.cpu_percent for m in metrics_list),
            "max_memory_percent": max(m.memory_percent for m in metrics_list)
        }


# Import asyncio
import asyncio















