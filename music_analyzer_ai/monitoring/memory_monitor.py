"""
Memory Monitor
Monitor GPU and CPU memory usage
"""

from typing import Dict, Any, Optional
import logging
import psutil

logger = logging.getLogger(__name__)

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available")


class MemoryMonitor:
    """Monitor memory usage"""
    
    def __init__(self):
        self.memory_history: list = []
    
    def get_cpu_memory(self) -> Dict[str, float]:
        """Get CPU memory usage"""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            "cpu_memory_mb": memory_info.rss / 1024 / 1024,
            "cpu_memory_percent": process.memory_percent()
        }
    
    def get_gpu_memory(self, device: int = 0) -> Dict[str, float]:
        """Get GPU memory usage"""
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return {}
        
        try:
            memory_allocated = torch.cuda.memory_allocated(device) / 1024 / 1024
            memory_reserved = torch.cuda.memory_reserved(device) / 1024 / 1024
            memory_total = torch.cuda.get_device_properties(device).total_memory / 1024 / 1024
            
            return {
                "gpu_memory_allocated_mb": memory_allocated,
                "gpu_memory_reserved_mb": memory_reserved,
                "gpu_memory_total_mb": memory_total,
                "gpu_memory_free_mb": memory_total - memory_reserved,
                "gpu_memory_percent": (memory_reserved / memory_total) * 100
            }
        except Exception as e:
            logger.warning(f"Error getting GPU memory: {str(e)}")
            return {}
    
    def get_memory_stats(self, device: int = 0) -> Dict[str, float]:
        """Get complete memory statistics"""
        stats = self.get_cpu_memory()
        stats.update(self.get_gpu_memory(device))
        return stats
    
    def record_snapshot(self, device: int = 0):
        """Record memory snapshot"""
        snapshot = self.get_memory_stats(device)
        self.memory_history.append(snapshot)
        return snapshot
    
    def get_history(self) -> list:
        """Get memory history"""
        return self.memory_history.copy()
    
    def clear_history(self):
        """Clear memory history"""
        self.memory_history.clear()
    
    def get_peak_memory(self) -> Dict[str, float]:
        """Get peak memory usage"""
        if not self.memory_history:
            return {}
        
        peak = {}
        for key in self.memory_history[0].keys():
            peak[key] = max(snapshot.get(key, 0) for snapshot in self.memory_history)
        
        return peak



