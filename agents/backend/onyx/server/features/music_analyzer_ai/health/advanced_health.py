"""
Advanced Health Checks
Comprehensive health checking system
"""

from typing import Dict, Any, Optional, List, Callable
import logging
import time
from datetime import datetime

logger = logging.getLogger(__name__)

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class HealthChecker:
    """
    Advanced health checking system
    """
    
    def __init__(self):
        self.checks: List[Callable] = []
        self.last_check: Optional[datetime] = None
        self.health_status: Dict[str, Any] = {}
    
    def register_check(self, check_func: Callable, name: str):
        """Register a health check"""
        self.checks.append({
            "name": name,
            "func": check_func
        })
    
    def check_all(self) -> Dict[str, Any]:
        """Run all health checks"""
        results = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "checks": {}
        }
        
        for check in self.checks:
            try:
                check_result = check["func"]()
                results["checks"][check["name"]] = check_result
                
                if not check_result.get("healthy", True):
                    results["status"] = "unhealthy"
            except Exception as e:
                results["checks"][check["name"]] = {
                    "healthy": False,
                    "error": str(e)
                }
                results["status"] = "unhealthy"
        
        self.last_check = datetime.now()
        self.health_status = results
        
        return results
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get last health check status"""
        return self.health_status.copy()


class SystemHealthChecks:
    """
    Predefined system health checks
    """
    
    @staticmethod
    def check_gpu() -> Dict[str, Any]:
        """Check GPU availability"""
        if not TORCH_AVAILABLE:
            return {"healthy": False, "message": "PyTorch not available"}
        
        if torch.cuda.is_available():
            return {
                "healthy": True,
                "gpu_count": torch.cuda.device_count(),
                "gpu_name": torch.cuda.get_device_name(0),
                "cuda_version": torch.version.cuda
            }
        else:
            return {
                "healthy": True,
                "message": "GPU not available, using CPU"
            }
    
    @staticmethod
    def check_memory() -> Dict[str, Any]:
        """Check memory usage"""
        try:
            import psutil
            memory = psutil.virtual_memory()
            
            return {
                "healthy": memory.percent < 90,
                "total_gb": memory.total / (1024**3),
                "available_gb": memory.available / (1024**3),
                "used_percent": memory.percent
            }
        except ImportError:
            return {"healthy": True, "message": "psutil not available"}
    
    @staticmethod
    def check_disk_space(path: str = ".") -> Dict[str, Any]:
        """Check disk space"""
        try:
            import shutil
            total, used, free = shutil.disk_usage(path)
            
            return {
                "healthy": (free / total) > 0.1,  # At least 10% free
                "total_gb": total / (1024**3),
                "used_gb": used / (1024**3),
                "free_gb": free / (1024**3),
                "free_percent": (free / total) * 100
            }
        except Exception as e:
            return {"healthy": False, "error": str(e)}
    
    @staticmethod
    def check_model_loading(model_path: str) -> Dict[str, Any]:
        """Check if model can be loaded"""
        if not TORCH_AVAILABLE:
            return {"healthy": False, "message": "PyTorch not available"}
        
        try:
            from pathlib import Path
            if not Path(model_path).exists():
                return {"healthy": False, "message": f"Model not found: {model_path}"}
            
            model = torch.load(model_path, map_location="cpu")
            return {"healthy": True, "message": "Model can be loaded"}
        except Exception as e:
            return {"healthy": False, "error": str(e)}


class ModelHealthMonitor:
    """
    Monitor health of deployed models
    """
    
    def __init__(self):
        self.model_status: Dict[str, Dict[str, Any]] = {}
    
    def check_model_health(
        self,
        model_id: str,
        model: Any,
        test_input: Any
    ) -> Dict[str, Any]:
        """Check health of a specific model"""
        try:
            import time
            start = time.time()
            
            # Test inference
            with torch.no_grad():
                if hasattr(model, '__call__'):
                    output = model(test_input)
                else:
                    output = model.forward(test_input)
            
            latency = time.time() - start
            
            # Check output
            is_valid = True
            if isinstance(output, torch.Tensor):
                if torch.isnan(output).any() or torch.isinf(output).any():
                    is_valid = False
            
            status = {
                "model_id": model_id,
                "healthy": is_valid and latency < 1.0,
                "latency_ms": latency * 1000,
                "output_valid": is_valid,
                "timestamp": datetime.now().isoformat()
            }
            
            self.model_status[model_id] = status
            return status
        
        except Exception as e:
            status = {
                "model_id": model_id,
                "healthy": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            self.model_status[model_id] = status
            return status
    
    def get_all_model_health(self) -> Dict[str, Dict[str, Any]]:
        """Get health status of all models"""
        return self.model_status.copy()

