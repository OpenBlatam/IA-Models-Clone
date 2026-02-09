"""
Health Checker
Health monitoring and diagnostics.
"""

import torch
import time
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)


class HealthStatus:
    """Health status enumeration."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class ComponentHealth:
    """Health status of a component."""
    
    def __init__(
        self,
        name: str,
        status: str,
        message: Optional[str] = None,
        metrics: Optional[Dict[str, Any]] = None,
    ):
        self.name = name
        self.status = status
        self.message = message
        self.metrics = metrics or {}
        self.timestamp = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "status": self.status,
            "message": self.message,
            "metrics": self.metrics,
            "timestamp": self.timestamp,
        }


class HealthChecker:
    """Check health of system components."""
    
    def __init__(self):
        self.checks: List[callable] = []
    
    def register_check(self, check_func: callable):
        """Register a health check function."""
        self.checks.append(check_func)
    
    def check_gpu(self) -> ComponentHealth:
        """Check GPU health."""
        try:
            if not torch.cuda.is_available():
                return ComponentHealth(
                    "gpu",
                    HealthStatus.UNHEALTHY,
                    "CUDA not available",
                )
            
            device_count = torch.cuda.device_count()
            memory_allocated = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
            memory_reserved = torch.cuda.memory_reserved() / (1024 ** 3)  # GB
            
            # Check if memory usage is high
            if memory_reserved > 0.9 * torch.cuda.get_device_properties(0).total_memory / (1024 ** 3):
                status = HealthStatus.DEGRADED
                message = "GPU memory usage is high"
            else:
                status = HealthStatus.HEALTHY
                message = "GPU is healthy"
            
            return ComponentHealth(
                "gpu",
                status,
                message,
                {
                    "device_count": device_count,
                    "memory_allocated_gb": round(memory_allocated, 2),
                    "memory_reserved_gb": round(memory_reserved, 2),
                },
            )
        except Exception as e:
            return ComponentHealth(
                "gpu",
                HealthStatus.UNHEALTHY,
                f"GPU check failed: {str(e)}",
            )
    
    def check_model(self, model: Optional[torch.nn.Module] = None) -> ComponentHealth:
        """Check model health."""
        try:
            if model is None:
                return ComponentHealth(
                    "model",
                    HealthStatus.DEGRADED,
                    "No model loaded",
                )
            
            # Check if model is on correct device
            device = next(model.parameters()).device
            param_count = sum(p.numel() for p in model.parameters())
            
            return ComponentHealth(
                "model",
                HealthStatus.HEALTHY,
                "Model is loaded and ready",
                {
                    "device": str(device),
                    "parameter_count": param_count,
                },
            )
        except Exception as e:
            return ComponentHealth(
                "model",
                HealthStatus.UNHEALTHY,
                f"Model check failed: {str(e)}",
            )
    
    def check_memory(self) -> ComponentHealth:
        """Check system memory."""
        try:
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / (1024 * 1024)
            memory_percent = process.memory_percent()
            
            if memory_percent > 90:
                status = HealthStatus.DEGRADED
                message = "Memory usage is very high"
            elif memory_percent > 75:
                status = HealthStatus.DEGRADED
                message = "Memory usage is high"
            else:
                status = HealthStatus.HEALTHY
                message = "Memory usage is normal"
            
            return ComponentHealth(
                "memory",
                status,
                message,
                {
                    "memory_mb": round(memory_mb, 2),
                    "memory_percent": round(memory_percent, 2),
                },
            )
        except Exception as e:
            return ComponentHealth(
                "memory",
                HealthStatus.UNHEALTHY,
                f"Memory check failed: {str(e)}",
            )
    
    def run_all_checks(self, model: Optional[torch.nn.Module] = None) -> Dict[str, ComponentHealth]:
        """Run all registered health checks."""
        results = {}
        
        # Default checks
        results["gpu"] = self.check_gpu()
        results["memory"] = self.check_memory()
        if model is not None:
            results["model"] = self.check_model(model)
        
        # Custom checks
        for check_func in self.checks:
            try:
                result = check_func()
                if isinstance(result, ComponentHealth):
                    results[result.name] = result
            except Exception as e:
                logger.error(f"Health check failed: {e}")
        
        return results
    
    def get_overall_status(self, results: Dict[str, ComponentHealth]) -> str:
        """Get overall health status."""
        statuses = [comp.status for comp in results.values()]
        
        if HealthStatus.UNHEALTHY in statuses:
            return HealthStatus.UNHEALTHY
        elif HealthStatus.DEGRADED in statuses:
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.HEALTHY



