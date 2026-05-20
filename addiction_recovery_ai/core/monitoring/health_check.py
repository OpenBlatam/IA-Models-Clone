"""
Health Check System
Monitor system and model health
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
import logging
import time
import psutil
import os

logger = logging.getLogger(__name__)


class SystemHealthMonitor:
    """
    Monitor system health
    """
    
    def __init__(self):
        """Initialize health monitor"""
        self.process = psutil.Process(os.getpid())
    
    def get_cpu_usage(self) -> float:
        """Get CPU usage percentage"""
        return self.process.cpu_percent(interval=0.1)
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get memory usage"""
        memory_info = self.process.memory_info()
        return {
            "rss_mb": memory_info.rss / 1024**2,
            "vms_mb": memory_info.vms / 1024**2,
            "percent": self.process.memory_percent()
        }
    
    def get_gpu_usage(self) -> Optional[Dict[str, Any]]:
        """Get GPU usage"""
        if not torch.cuda.is_available():
            return None
        
        return {
            "allocated_gb": torch.cuda.memory_allocated() / 1024**3,
            "reserved_gb": torch.cuda.memory_reserved() / 1024**3,
            "max_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3,
            "device_count": torch.cuda.device_count(),
            "current_device": torch.cuda.current_device()
        }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get complete health status"""
        health = {
            "status": "healthy",
            "cpu_usage_percent": self.get_cpu_usage(),
            "memory": self.get_memory_usage(),
            "gpu": self.get_gpu_usage(),
            "timestamp": time.time()
        }
        
        # Check for issues
        issues = []
        
        if health["cpu_usage_percent"] > 90:
            issues.append("High CPU usage")
            health["status"] = "warning"
        
        if health["memory"]["percent"] > 90:
            issues.append("High memory usage")
            health["status"] = "warning"
        
        if health["gpu"] and health["gpu"]["allocated_gb"] > 8:
            issues.append("High GPU memory usage")
            health["status"] = "warning"
        
        health["issues"] = issues
        return health


class ModelHealthMonitor:
    """
    Monitor model health
    """
    
    def __init__(self, model: nn.Module):
        """
        Initialize model health monitor
        
        Args:
            model: Model to monitor
        """
        self.model = model
        self.inference_times = []
        self.error_count = 0
        self.total_inferences = 0
    
    def check_model_health(self) -> Dict[str, Any]:
        """
        Check model health
        
        Returns:
            Health status
        """
        health = {
            "status": "healthy",
            "parameter_count": sum(p.numel() for p in self.model.parameters()),
            "has_nan": False,
            "has_inf": False,
            "inference_stats": {}
        }
        
        # Check parameters
        for param in self.model.parameters():
            if torch.isnan(param).any():
                health["has_nan"] = True
                health["status"] = "unhealthy"
            
            if torch.isinf(param).any():
                health["has_inf"] = True
                health["status"] = "unhealthy"
        
        # Inference statistics
        if self.inference_times:
            import numpy as np
            health["inference_stats"] = {
                "mean_time_ms": np.mean(self.inference_times),
                "std_time_ms": np.std(self.inference_times),
                "min_time_ms": np.min(self.inference_times),
                "max_time_ms": np.max(self.inference_times),
                "total_inferences": self.total_inferences,
                "error_rate": self.error_count / max(self.total_inferences, 1)
            }
        
        return health
    
    def record_inference(self, inference_time_ms: float, success: bool = True):
        """
        Record inference
        
        Args:
            inference_time_ms: Inference time in milliseconds
            success: Whether inference was successful
        """
        self.inference_times.append(inference_time_ms)
        self.total_inferences += 1
        if not success:
            self.error_count += 1
        
        # Keep only recent times (last 1000)
        if len(self.inference_times) > 1000:
            self.inference_times = self.inference_times[-1000:]


def create_system_monitor() -> SystemHealthMonitor:
    """Factory for system monitor"""
    return SystemHealthMonitor()


def create_model_monitor(model: nn.Module) -> ModelHealthMonitor:
    """Factory for model monitor"""
    return ModelHealthMonitor(model)













