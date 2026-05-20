"""
Resource Management for Recovery AI
"""

import torch
import psutil
import gc
from typing import Dict, List, Optional, Any
import logging
import threading
import time

logger = logging.getLogger(__name__)


class ResourceMonitor:
    """Monitor system resources"""
    
    def __init__(self):
        """Initialize resource monitor"""
        self.monitoring = False
        self.monitor_thread = None
        self.stats = {
            "cpu_percent": [],
            "memory_percent": [],
            "gpu_memory": []
        }
        logger.info("ResourceMonitor initialized")
    
    def start_monitoring(self, interval: float = 1.0):
        """
        Start resource monitoring
        
        Args:
            interval: Monitoring interval in seconds
        """
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
        logger.info("Resource monitoring started")
    
    def _monitor_loop(self, interval: float):
        """Monitoring loop"""
        while self.monitoring:
            # CPU
            cpu_percent = psutil.cpu_percent(interval=None)
            self.stats["cpu_percent"].append(cpu_percent)
            
            # Memory
            memory = psutil.virtual_memory()
            self.stats["memory_percent"].append(memory.percent)
            
            # GPU
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated() / 1024**3  # GB
                self.stats["gpu_memory"].append(gpu_memory)
            
            # Keep only last 1000 samples
            for key in self.stats:
                if len(self.stats[key]) > 1000:
                    self.stats[key] = self.stats[key][-1000:]
            
            time.sleep(interval)
    
    def stop_monitoring(self):
        """Stop resource monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        logger.info("Resource monitoring stopped")
    
    def get_current_stats(self) -> Dict[str, Any]:
        """Get current resource statistics"""
        stats = {
            "cpu_percent": psutil.cpu_percent(interval=None),
            "memory_percent": psutil.virtual_memory().percent,
            "memory_available_gb": psutil.virtual_memory().available / 1024**3
        }
        
        if torch.cuda.is_available():
            stats["gpu_memory_allocated_gb"] = torch.cuda.memory_allocated() / 1024**3
            stats["gpu_memory_reserved_gb"] = torch.cuda.memory_reserved() / 1024**3
            stats["gpu_memory_free_gb"] = (
                torch.cuda.get_device_properties(0).total_memory / 1024**3
                - stats["gpu_memory_reserved_gb"]
            )
        
        return stats
    
    def get_average_stats(self) -> Dict[str, Any]:
        """Get average statistics"""
        if not self.stats["cpu_percent"]:
            return {}
        
        avg_stats = {
            "avg_cpu_percent": sum(self.stats["cpu_percent"]) / len(self.stats["cpu_percent"]),
            "avg_memory_percent": sum(self.stats["memory_percent"]) / len(self.stats["memory_percent"])
        }
        
        if self.stats["gpu_memory"]:
            avg_stats["avg_gpu_memory_gb"] = sum(self.stats["gpu_memory"]) / len(self.stats["gpu_memory"])
        
        return avg_stats


class MemoryManager:
    """Memory management utilities"""
    
    @staticmethod
    def clear_cache():
        """Clear PyTorch cache"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        logger.info("Cache cleared")
    
    @staticmethod
    def get_memory_usage() -> Dict[str, float]:
        """Get current memory usage in GB"""
        usage = {
            "cpu_memory_gb": psutil.virtual_memory().used / 1024**3
        }
        
        if torch.cuda.is_available():
            usage["gpu_memory_allocated_gb"] = torch.cuda.memory_allocated() / 1024**3
            usage["gpu_memory_reserved_gb"] = torch.cuda.memory_reserved() / 1024**3
        
        return usage
    
    @staticmethod
    def optimize_memory():
        """Optimize memory usage"""
        MemoryManager.clear_cache()
        logger.info("Memory optimized")


class ResourceLimiter:
    """Limit resource usage"""
    
    def __init__(
        self,
        max_cpu_percent: Optional[float] = None,
        max_memory_percent: Optional[float] = None,
        max_gpu_memory_gb: Optional[float] = None
    ):
        """
        Initialize resource limiter
        
        Args:
            max_cpu_percent: Maximum CPU usage percent
            max_memory_percent: Maximum memory usage percent
            max_gpu_memory_gb: Maximum GPU memory in GB
        """
        self.max_cpu_percent = max_cpu_percent
        self.max_memory_percent = max_memory_percent
        self.max_gpu_memory_gb = max_gpu_memory_gb
        logger.info("ResourceLimiter initialized")
    
    def check_limits(self) -> Dict[str, bool]:
        """
        Check if resource limits are exceeded
        
        Returns:
            Dictionary with limit status
        """
        status = {}
        
        if self.max_cpu_percent:
            cpu_percent = psutil.cpu_percent(interval=None)
            status["cpu_ok"] = cpu_percent < self.max_cpu_percent
        
        if self.max_memory_percent:
            memory_percent = psutil.virtual_memory().percent
            status["memory_ok"] = memory_percent < self.max_memory_percent
        
        if self.max_gpu_memory_gb and torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024**3
            status["gpu_ok"] = gpu_memory < self.max_gpu_memory_gb
        
        return status
    
    def wait_if_needed(self):
        """Wait if resource limits are exceeded"""
        while True:
            status = self.check_limits()
            if all(status.values()):
                break
            time.sleep(0.1)

