"""
Health Checker for Flux2 Clothing Changer
==========================================

System health monitoring and diagnostics.
"""

import time
import torch
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status enumeration."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


@dataclass
class HealthCheck:
    """Individual health check result."""
    name: str
    status: HealthStatus
    message: str
    details: Dict[str, Any] = None
    response_time: float = 0.0
    
    def __post_init__(self):
        if self.details is None:
            self.details = {}


class HealthChecker:
    """System health checker."""
    
    def __init__(self, model=None):
        """
        Initialize health checker.
        
        Args:
            model: Optional model instance for checks
        """
        self.model = model
        self.checks: List[HealthCheck] = []
    
    def check_all(self) -> Dict[str, Any]:
        """
        Run all health checks.
        
        Returns:
            Health status summary
        """
        self.checks.clear()
        
        # Run checks
        self._check_system_resources()
        self._check_gpu_availability()
        self._check_model_availability()
        self._check_dependencies()
        self._check_disk_space()
        
        # Determine overall status
        overall_status = self._determine_overall_status()
        
        return {
            "status": overall_status.value,
            "timestamp": time.time(),
            "checks": [
                {
                    "name": check.name,
                    "status": check.status.value,
                    "message": check.message,
                    "response_time": check.response_time,
                    "details": check.details,
                }
                for check in self.checks
            ],
            "summary": {
                "total_checks": len(self.checks),
                "healthy": sum(1 for c in self.checks if c.status == HealthStatus.HEALTHY),
                "degraded": sum(1 for c in self.checks if c.status == HealthStatus.DEGRADED),
                "unhealthy": sum(1 for c in self.checks if c.status == HealthStatus.UNHEALTHY),
                "critical": sum(1 for c in self.checks if c.status == HealthStatus.CRITICAL),
            },
        }
    
    def _check_system_resources(self) -> None:
        """Check system resources."""
        start = time.time()
        try:
            import psutil
            
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            
            status = HealthStatus.HEALTHY
            message = "System resources OK"
            
            if cpu_percent > 90:
                status = HealthStatus.CRITICAL
                message = "CPU usage critically high"
            elif cpu_percent > 75:
                status = HealthStatus.UNHEALTHY
                message = "CPU usage high"
            elif cpu_percent > 50:
                status = HealthStatus.DEGRADED
                message = "CPU usage elevated"
            
            if memory.percent > 95:
                status = HealthStatus.CRITICAL
                message = "Memory usage critically high"
            elif memory.percent > 85:
                status = max(status, HealthStatus.UNHEALTHY)
                message = "Memory usage high"
            
            self.checks.append(HealthCheck(
                name="system_resources",
                status=status,
                message=message,
                response_time=time.time() - start,
                details={
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory.percent,
                    "memory_available_mb": memory.available / (1024 * 1024),
                }
            ))
        except Exception as e:
            self.checks.append(HealthCheck(
                name="system_resources",
                status=HealthStatus.UNHEALTHY,
                message=f"Failed to check system resources: {e}",
                response_time=time.time() - start,
            ))
    
    def _check_gpu_availability(self) -> None:
        """Check GPU availability."""
        start = time.time()
        try:
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpu_name = torch.cuda.get_device_name(0)
                memory_allocated = torch.cuda.memory_allocated(0) / (1024 * 1024)
                memory_reserved = torch.cuda.memory_reserved(0) / (1024 * 1024)
                
                status = HealthStatus.HEALTHY
                message = f"GPU available: {gpu_name}"
                
                # Check memory usage
                if memory_reserved > 20000:  # 20GB
                    status = HealthStatus.DEGRADED
                    message = "GPU memory usage high"
                
                self.checks.append(HealthCheck(
                    name="gpu_availability",
                    status=status,
                    message=message,
                    response_time=time.time() - start,
                    details={
                        "gpu_count": gpu_count,
                        "gpu_name": gpu_name,
                        "memory_allocated_mb": memory_allocated,
                        "memory_reserved_mb": memory_reserved,
                    }
                ))
            else:
                self.checks.append(HealthCheck(
                    name="gpu_availability",
                    status=HealthStatus.DEGRADED,
                    message="GPU not available, using CPU",
                    response_time=time.time() - start,
                ))
        except Exception as e:
            self.checks.append(HealthCheck(
                name="gpu_availability",
                status=HealthStatus.UNHEALTHY,
                message=f"Failed to check GPU: {e}",
                response_time=time.time() - start,
            ))
    
    def _check_model_availability(self) -> None:
        """Check model availability."""
        start = time.time()
        try:
            if self.model is None:
                self.checks.append(HealthCheck(
                    name="model_availability",
                    status=HealthStatus.UNHEALTHY,
                    message="Model not initialized",
                    response_time=time.time() - start,
                ))
                return
            
            # Check if model has required components
            has_pipeline = hasattr(self.model, 'pipeline') and self.model.pipeline is not None
            has_clip = hasattr(self.model, 'clip_vision') and self.model.clip_vision is not None
            
            if has_pipeline and has_clip:
                self.checks.append(HealthCheck(
                    name="model_availability",
                    status=HealthStatus.HEALTHY,
                    message="Model components available",
                    response_time=time.time() - start,
                    details={
                        "has_pipeline": has_pipeline,
                        "has_clip": has_clip,
                    }
                ))
            else:
                self.checks.append(HealthCheck(
                    name="model_availability",
                    status=HealthStatus.UNHEALTHY,
                    message="Model components missing",
                    response_time=time.time() - start,
                    details={
                        "has_pipeline": has_pipeline,
                        "has_clip": has_clip,
                    }
                ))
        except Exception as e:
            self.checks.append(HealthCheck(
                name="model_availability",
                status=HealthStatus.UNHEALTHY,
                message=f"Failed to check model: {e}",
                response_time=time.time() - start,
            ))
    
    def _check_dependencies(self) -> None:
        """Check required dependencies."""
        start = time.time()
        missing = []
        
        try:
            import diffusers
        except ImportError:
            missing.append("diffusers")
        
        try:
            import transformers
        except ImportError:
            missing.append("transformers")
        
        try:
            import torch
        except ImportError:
            missing.append("torch")
        
        if missing:
            self.checks.append(HealthCheck(
                name="dependencies",
                status=HealthStatus.CRITICAL,
                message=f"Missing dependencies: {', '.join(missing)}",
                response_time=time.time() - start,
                details={"missing": missing}
            ))
        else:
            self.checks.append(HealthCheck(
                name="dependencies",
                status=HealthStatus.HEALTHY,
                message="All dependencies available",
                response_time=time.time() - start,
            ))
    
    def _check_disk_space(self) -> None:
        """Check disk space."""
        start = time.time()
        try:
            import shutil
            
            total, used, free = shutil.disk_usage(".")
            free_gb = free / (1024 ** 3)
            free_percent = (free / total) * 100
            
            status = HealthStatus.HEALTHY
            message = f"Disk space OK: {free_gb:.2f} GB free"
            
            if free_percent < 5:
                status = HealthStatus.CRITICAL
                message = "Disk space critically low"
            elif free_percent < 10:
                status = HealthStatus.UNHEALTHY
                message = "Disk space low"
            elif free_percent < 20:
                status = HealthStatus.DEGRADED
                message = "Disk space getting low"
            
            self.checks.append(HealthCheck(
                name="disk_space",
                status=status,
                message=message,
                response_time=time.time() - start,
                details={
                    "free_gb": free_gb,
                    "free_percent": free_percent,
                }
            ))
        except Exception as e:
            self.checks.append(HealthCheck(
                name="disk_space",
                status=HealthStatus.DEGRADED,
                message=f"Failed to check disk space: {e}",
                response_time=time.time() - start,
            ))
    
    def _determine_overall_status(self) -> HealthStatus:
        """Determine overall health status."""
        if not self.checks:
            return HealthStatus.UNHEALTHY
        
        # Get worst status
        statuses = [check.status for check in self.checks]
        
        if HealthStatus.CRITICAL in statuses:
            return HealthStatus.CRITICAL
        elif HealthStatus.UNHEALTHY in statuses:
            return HealthStatus.UNHEALTHY
        elif HealthStatus.DEGRADED in statuses:
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.HEALTHY


