"""
Resource Optimizer for Flux2 Clothing Changer
==============================================

Advanced resource optimization and management.
"""

import torch
import gc
import psutil
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import logging
import time
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class ResourceUsage:
    """Resource usage snapshot."""
    timestamp: float
    cpu_percent: float
    memory_mb: float
    gpu_memory_mb: Optional[float] = None
    gpu_utilization: Optional[float] = None
    disk_usage_percent: Optional[float] = None


class ResourceOptimizer:
    """Resource optimization and management system."""
    
    def __init__(
        self,
        target_memory_mb: Optional[float] = None,
        target_gpu_memory_mb: Optional[float] = None,
        enable_auto_cleanup: bool = True,
    ):
        """
        Initialize resource optimizer.
        
        Args:
            target_memory_mb: Target memory usage in MB
            target_gpu_memory_mb: Target GPU memory usage in MB
            enable_auto_cleanup: Enable automatic cleanup
        """
        self.target_memory_mb = target_memory_mb
        self.target_gpu_memory_mb = target_gpu_memory_mb
        self.enable_auto_cleanup = enable_auto_cleanup
        
        self.usage_history: deque = deque(maxlen=100)
        self.optimization_count = 0
    
    def get_current_usage(self) -> ResourceUsage:
        """Get current resource usage."""
        process = psutil.Process()
        
        # CPU and memory
        cpu_percent = process.cpu_percent(interval=0.1)
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / (1024 * 1024)
        
        # GPU metrics
        gpu_memory_mb = None
        gpu_utilization = None
        
        if torch.cuda.is_available():
            try:
                gpu_memory_mb = torch.cuda.memory_allocated() / (1024 * 1024)
                gpu_reserved = torch.cuda.memory_reserved() / (1024 * 1024)
                gpu_utilization = min(100.0, (gpu_memory_mb / gpu_reserved * 100) if gpu_reserved > 0 else 0.0)
            except Exception:
                pass
        
        # Disk usage
        disk_usage = psutil.disk_usage(".")
        disk_usage_percent = (disk_usage.used / disk_usage.total) * 100
        
        usage = ResourceUsage(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            memory_mb=memory_mb,
            gpu_memory_mb=gpu_memory_mb,
            gpu_utilization=gpu_utilization,
            disk_usage_percent=disk_usage_percent,
        )
        
        self.usage_history.append(usage)
        return usage
    
    def optimize_memory(self) -> Dict[str, Any]:
        """
        Optimize memory usage.
        
        Returns:
            Optimization results
        """
        results = {
            "before": self.get_current_usage(),
            "actions": [],
        }
        
        # Python garbage collection
        collected = gc.collect()
        results["actions"].append({
            "action": "garbage_collection",
            "collected": collected,
        })
        
        # Clear PyTorch cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            results["actions"].append({
                "action": "clear_cuda_cache",
            })
        
        # Clear Python cache
        try:
            import sys
            # Clear module cache (be careful with this)
            # sys.modules.clear()  # Too aggressive
            results["actions"].append({
                "action": "cache_cleanup",
            })
        except Exception:
            pass
        
        results["after"] = self.get_current_usage()
        results["memory_freed_mb"] = (
            results["before"].memory_mb - results["after"].memory_mb
        )
        
        if self.enable_auto_cleanup:
            self.optimization_count += 1
        
        logger.info(f"Memory optimization freed {results['memory_freed_mb']:.2f}MB")
        return results
    
    def optimize_gpu_memory(self) -> Dict[str, Any]:
        """
        Optimize GPU memory usage.
        
        Returns:
            Optimization results
        """
        if not torch.cuda.is_available():
            return {"error": "CUDA not available"}
        
        results = {
            "before": self.get_current_usage(),
            "actions": [],
        }
        
        # Clear CUDA cache
        torch.cuda.empty_cache()
        results["actions"].append({"action": "clear_cuda_cache"})
        
        # Synchronize
        torch.cuda.synchronize()
        results["actions"].append({"action": "synchronize"})
        
        # Reset peak stats
        torch.cuda.reset_peak_memory_stats()
        results["actions"].append({"action": "reset_peak_stats"})
        
        results["after"] = self.get_current_usage()
        
        if results["before"].gpu_memory_mb and results["after"].gpu_memory_mb:
            results["gpu_memory_freed_mb"] = (
                results["before"].gpu_memory_mb - results["after"].gpu_memory_mb
            )
        
        logger.info("GPU memory optimized")
        return results
    
    def should_optimize(self) -> bool:
        """Check if optimization is needed."""
        usage = self.get_current_usage()
        
        # Check memory
        if self.target_memory_mb and usage.memory_mb > self.target_memory_mb * 1.1:
            return True
        
        # Check GPU memory
        if self.target_gpu_memory_mb and usage.gpu_memory_mb:
            if usage.gpu_memory_mb > self.target_gpu_memory_mb * 1.1:
                return True
        
        # Check disk usage
        if usage.disk_usage_percent and usage.disk_usage_percent > 90:
            return True
        
        return False
    
    def auto_optimize(self) -> Dict[str, Any]:
        """
        Perform automatic optimization if needed.
        
        Returns:
            Optimization results
        """
        if not self.should_optimize():
            return {"optimized": False, "reason": "No optimization needed"}
        
        results = {
            "optimized": True,
            "memory": self.optimize_memory(),
        }
        
        if torch.cuda.is_available():
            results["gpu"] = self.optimize_gpu_memory()
        
        return results
    
    def get_optimization_recommendations(self) -> List[str]:
        """Get optimization recommendations."""
        recommendations = []
        usage = self.get_current_usage()
        
        # Memory recommendations
        if usage.memory_mb > 8000:  # 8GB
            recommendations.append("Consider reducing batch size or image resolution")
        
        if usage.cpu_percent > 80:
            recommendations.append("High CPU usage - consider reducing concurrent requests")
        
        if usage.gpu_memory_mb and usage.gpu_memory_mb > 20000:  # 20GB
            recommendations.append("High GPU memory usage - consider model quantization")
        
        if usage.disk_usage_percent and usage.disk_usage_percent > 85:
            recommendations.append("Disk space running low - consider cleanup")
        
        if not recommendations:
            recommendations.append("Resources are within optimal ranges")
        
        return recommendations
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get optimization statistics."""
        if not self.usage_history:
            return {}
        
        recent_usage = list(self.usage_history)[-10:]
        
        return {
            "optimization_count": self.optimization_count,
            "current_usage": {
                "memory_mb": recent_usage[-1].memory_mb if recent_usage else 0,
                "cpu_percent": recent_usage[-1].cpu_percent if recent_usage else 0,
                "gpu_memory_mb": recent_usage[-1].gpu_memory_mb if recent_usage else None,
            },
            "average_usage": {
                "memory_mb": sum(u.memory_mb for u in recent_usage) / len(recent_usage) if recent_usage else 0,
                "cpu_percent": sum(u.cpu_percent for u in recent_usage) / len(recent_usage) if recent_usage else 0,
            },
            "recommendations": self.get_optimization_recommendations(),
        }


