"""
Advanced Monitoring
===================
System monitoring and health checks
"""

from typing import Dict, Any, List, Optional, Callable
import torch
import torch.nn as nn
import structlog
import psutil
import time
from datetime import datetime

logger = structlog.get_logger()


class SystemMonitor:
    """
    System resource monitoring
    """
    
    def __init__(self):
        """Initialize monitor"""
        logger.info("SystemMonitor initialized")
    
    def get_system_stats(self) -> Dict[str, Any]:
        """
        Get system statistics
        
        Returns:
            System statistics
        """
        stats = {
            "cpu": {
                "percent": psutil.cpu_percent(interval=1),
                "count": psutil.cpu_count(),
                "frequency": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None
            },
            "memory": {
                "total": psutil.virtual_memory().total / (1024**3),  # GB
                "available": psutil.virtual_memory().available / (1024**3),  # GB
                "percent": psutil.virtual_memory().percent,
                "used": psutil.virtual_memory().used / (1024**3)  # GB
            },
            "disk": {
                "total": psutil.disk_usage('/').total / (1024**3),  # GB
                "used": psutil.disk_usage('/').used / (1024**3),  # GB
                "free": psutil.disk_usage('/').free / (1024**3),  # GB
                "percent": psutil.disk_usage('/').percent
            }
        }
        
        # GPU stats
        if torch.cuda.is_available():
            stats["gpu"] = {
                "available": True,
                "device_count": torch.cuda.device_count(),
                "current_device": torch.cuda.current_device(),
                "device_name": torch.cuda.get_device_name(0),
                "memory_allocated": torch.cuda.memory_allocated(0) / (1024**3),  # GB
                "memory_reserved": torch.cuda.memory_reserved(0) / (1024**3),  # GB
                "memory_total": torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
            }
        else:
            stats["gpu"] = {"available": False}
        
        return stats
    
    def get_model_stats(self, model: nn.Module) -> Dict[str, Any]:
        """
        Get model statistics
        
        Args:
            model: Model
            
        Returns:
            Model statistics
        """
        from .model_utils import count_parameters, get_model_summary
        
        total_params = count_parameters(model, trainable_only=False)
        trainable_params = count_parameters(model, trainable_only=True)
        
        # Estimate memory usage
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        model_size_mb = (param_size + buffer_size) / (1024 ** 2)
        
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "non_trainable_parameters": total_params - trainable_params,
            "model_size_mb": model_size_mb
        }


class HealthChecker:
    """Health check system"""
    
    def __init__(self):
        """Initialize health checker"""
        self.checks: List[Callable] = []
        logger.info("HealthChecker initialized")
    
    def register_check(self, check_fn: Callable) -> None:
        """
        Register health check
        
        Args:
            check_fn: Check function that returns (status, message)
        """
        self.checks.append(check_fn)
    
    def check_health(self) -> Dict[str, Any]:
        """
        Run all health checks
        
        Returns:
            Health status
        """
        results = {
            "status": "healthy",
            "checks": {},
            "timestamp": datetime.utcnow().isoformat()
        }
        
        for check_fn in self.checks:
            try:
                status, message = check_fn()
                check_name = check_fn.__name__
                results["checks"][check_name] = {
                    "status": status,
                    "message": message
                }
                
                if status != "healthy":
                    results["status"] = "unhealthy"
            except Exception as e:
                results["checks"][check_fn.__name__] = {
                    "status": "error",
                    "message": str(e)
                }
                results["status"] = "unhealthy"
        
        return results


# Global instances
system_monitor = SystemMonitor()
health_checker = HealthChecker()

