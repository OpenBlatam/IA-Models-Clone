"""
Health Checker
==============

Health checks del sistema.
"""

import logging
import torch
import psutil
from typing import Dict, Any, List
from datetime import datetime

logger = logging.getLogger(__name__)


class HealthChecker:
    """Health checker del sistema."""
    
    def __init__(self):
        """Inicializar health checker."""
        self._logger = logger
    
    def check_system_health(self) -> Dict[str, Any]:
        """
        Verificar salud del sistema.
        
        Returns:
            Estado de salud
        """
        health = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "checks": {}
        }
        
        # Check GPU
        gpu_health = self._check_gpu()
        health["checks"]["gpu"] = gpu_health
        if not gpu_health["available"] and gpu_health.get("required", False):
            health["status"] = "degraded"
        
        # Check CPU
        cpu_health = self._check_cpu()
        health["checks"]["cpu"] = cpu_health
        if cpu_health["usage"] > 90:
            health["status"] = "degraded"
        
        # Check Memory
        memory_health = self._check_memory()
        health["checks"]["memory"] = memory_health
        if memory_health["usage_percent"] > 90:
            health["status"] = "degraded"
        
        # Check Disk
        disk_health = self._check_disk()
        health["checks"]["disk"] = disk_health
        if disk_health["usage_percent"] > 90:
            health["status"] = "degraded"
        
        return health
    
    def _check_gpu(self) -> Dict[str, Any]:
        """Verificar GPU."""
        if not torch.cuda.is_available():
            return {
                "available": False,
                "device_count": 0
            }
        
        try:
            device_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            memory_allocated = torch.cuda.memory_allocated() / 1e9  # GB
            memory_reserved = torch.cuda.memory_reserved() / 1e9  # GB
            
            return {
                "available": True,
                "device_count": device_count,
                "current_device": current_device,
                "memory_allocated_gb": memory_allocated,
                "memory_reserved_gb": memory_reserved,
                "memory_free_gb": memory_reserved - memory_allocated
            }
        except Exception as e:
            self._logger.error(f"Error verificando GPU: {str(e)}")
            return {
                "available": False,
                "error": str(e)
            }
    
    def _check_cpu(self) -> Dict[str, Any]:
        """Verificar CPU."""
        try:
            return {
                "usage": psutil.cpu_percent(interval=0.1),
                "count": psutil.cpu_count(),
                "load_avg": psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None
            }
        except Exception as e:
            self._logger.error(f"Error verificando CPU: {str(e)}")
            return {"error": str(e)}
    
    def _check_memory(self) -> Dict[str, Any]:
        """Verificar memoria."""
        try:
            memory = psutil.virtual_memory()
            return {
                "total_gb": memory.total / 1e9,
                "available_gb": memory.available / 1e9,
                "used_gb": memory.used / 1e9,
                "usage_percent": memory.percent
            }
        except Exception as e:
            self._logger.error(f"Error verificando memoria: {str(e)}")
            return {"error": str(e)}
    
    def _check_disk(self) -> Dict[str, Any]:
        """Verificar disco."""
        try:
            disk = psutil.disk_usage('/')
            return {
                "total_gb": disk.total / 1e9,
                "used_gb": disk.used / 1e9,
                "free_gb": disk.free / 1e9,
                "usage_percent": (disk.used / disk.total) * 100
            }
        except Exception as e:
            self._logger.error(f"Error verificando disco: {str(e)}")
            return {"error": str(e)}
    
    def check_model_health(self, model: Any) -> Dict[str, Any]:
        """
        Verificar salud del modelo.
        
        Args:
            model: Modelo a verificar
        
        Returns:
            Estado de salud del modelo
        """
        try:
            # Verificar que el modelo existe
            if model is None:
                return {"status": "unhealthy", "error": "Model is None"}
            
            # Verificar parámetros
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            return {
                "status": "healthy",
                "total_params": total_params,
                "trainable_params": trainable_params,
                "frozen_params": total_params - trainable_params
            }
        except Exception as e:
            self._logger.error(f"Error verificando modelo: {str(e)}")
            return {"status": "unhealthy", "error": str(e)}




