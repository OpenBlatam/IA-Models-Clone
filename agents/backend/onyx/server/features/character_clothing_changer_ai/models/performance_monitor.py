"""
Performance Monitor for Flux2 Clothing Changer
==============================================

Real-time performance monitoring and metrics collection.
"""

import time
import threading
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from collections import deque
import logging
import psutil
import torch

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics snapshot."""
    timestamp: float
    cpu_percent: float
    memory_mb: float
    gpu_memory_mb: Optional[float] = None
    gpu_utilization: Optional[float] = None
    processing_time: float = 0.0
    queue_size: int = 0
    active_requests: int = 0


class PerformanceMonitor:
    """Real-time performance monitoring."""
    
    def __init__(
        self,
        history_size: int = 1000,
        update_interval: float = 1.0,
        enable_gpu_monitoring: bool = True,
    ):
        """
        Initialize performance monitor.
        
        Args:
            history_size: Size of metrics history
            update_interval: Update interval in seconds
            enable_gpu_monitoring: Enable GPU monitoring
        """
        self.history_size = history_size
        self.update_interval = update_interval
        self.enable_gpu_monitoring = enable_gpu_monitoring
        
        self.metrics_history: deque = deque(maxlen=history_size)
        self.is_monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        
        self.current_metrics = PerformanceMetrics(
            timestamp=time.time(),
            cpu_percent=0.0,
            memory_mb=0.0,
        )
    
    def start_monitoring(self) -> None:
        """Start performance monitoring."""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.stop_event.clear()
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Performance monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop performance monitoring."""
        if not self.is_monitoring:
            return
        
        self.is_monitoring = False
        self.stop_event.set()
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        logger.info("Performance monitoring stopped")
    
    def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while not self.stop_event.is_set():
            try:
                metrics = self._collect_metrics()
                self.metrics_history.append(metrics)
                self.current_metrics = metrics
                time.sleep(self.update_interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.update_interval)
    
    def _collect_metrics(self) -> PerformanceMetrics:
        """Collect current performance metrics."""
        process = psutil.Process()
        
        # CPU and memory
        cpu_percent = process.cpu_percent()
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / (1024 * 1024)
        
        # GPU metrics
        gpu_memory_mb = None
        gpu_utilization = None
        
        if self.enable_gpu_monitoring and torch.cuda.is_available():
            try:
                gpu_memory_mb = torch.cuda.memory_allocated() / (1024 * 1024)
                # GPU utilization requires nvidia-ml-py or similar
                # For now, we'll use memory as a proxy
                gpu_utilization = min(100.0, (gpu_memory_mb / 1024) * 10)  # Rough estimate
            except Exception:
                pass
        
        return PerformanceMetrics(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            memory_mb=memory_mb,
            gpu_memory_mb=gpu_memory_mb,
            gpu_utilization=gpu_utilization,
        )
    
    def get_current_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics."""
        return self.current_metrics
    
    def get_average_metrics(
        self,
        duration: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Get average metrics over a duration.
        
        Args:
            duration: Duration in seconds (None for all history)
            
        Returns:
            Dictionary of average metrics
        """
        if not self.metrics_history:
            return {}
        
        cutoff_time = time.time() - duration if duration else 0
        
        relevant_metrics = [
            m for m in self.metrics_history
            if m.timestamp >= cutoff_time
        ]
        
        if not relevant_metrics:
            return {}
        
        return {
            "avg_cpu_percent": sum(m.cpu_percent for m in relevant_metrics) / len(relevant_metrics),
            "avg_memory_mb": sum(m.memory_mb for m in relevant_metrics) / len(relevant_metrics),
            "max_memory_mb": max(m.memory_mb for m in relevant_metrics),
            "avg_gpu_memory_mb": (
                sum(m.gpu_memory_mb for m in relevant_metrics if m.gpu_memory_mb)
                / len([m for m in relevant_metrics if m.gpu_memory_mb])
                if any(m.gpu_memory_mb for m in relevant_metrics) else None
            ),
            "max_gpu_memory_mb": (
                max(m.gpu_memory_mb for m in relevant_metrics if m.gpu_memory_mb)
                if any(m.gpu_memory_mb for m in relevant_metrics) else None
            ),
        }
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary."""
        return {
            "current": {
                "cpu_percent": self.current_metrics.cpu_percent,
                "memory_mb": self.current_metrics.memory_mb,
                "gpu_memory_mb": self.current_metrics.gpu_memory_mb,
                "gpu_utilization": self.current_metrics.gpu_utilization,
            },
            "averages": self.get_average_metrics(),
            "history_size": len(self.metrics_history),
        }


