"""
GPU Optimization
================

Optimizaciones específicas para GPU.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class GPUOptimizer:
    """
    Optimizador para GPU.
    """
    
    @staticmethod
    def enable_all_optimizations():
        """Habilitar todas las optimizaciones de GPU."""
        if not torch.cuda.is_available():
            logger.warning("CUDA no disponible")
            return
        
        # cuDNN optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.allow_tf32 = True
        
        # TensorFloat-32
        torch.backends.cuda.matmul.allow_tf32 = True
        
        # Memory optimizations
        torch.cuda.empty_cache()
        
        logger.info("Todas las optimizaciones de GPU habilitadas")
    
    @staticmethod
    def optimize_model_for_gpu(model: nn.Module) -> nn.Module:
        """
        Optimizar modelo para GPU.
        
        Args:
            model: Modelo
            
        Returns:
            Modelo optimizado
        """
        if not torch.cuda.is_available():
            return model
        
        model = model.cuda()
        
        # Compilar si disponible
        if hasattr(torch, 'compile'):
            try:
                model = torch.compile(model, mode="reduce-overhead")
                logger.info("Modelo compilado para GPU")
            except Exception as e:
                logger.warning(f"Error compilando: {e}")
        
        return model
    
    @staticmethod
    def get_gpu_info() -> Dict[str, Any]:
        """
        Obtener información de GPU.
        
        Returns:
            Información de GPU
        """
        if not torch.cuda.is_available():
            return {"cuda_available": False}
        
        device_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        
        info = {
            "cuda_available": True,
            "device_count": device_count,
            "current_device": current_device,
            "device_name": torch.cuda.get_device_name(current_device),
            "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
            "cudnn_enabled": torch.backends.cudnn.enabled,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "tf32_enabled": torch.backends.cuda.matmul.allow_tf32
        }
        
        # Memoria
        for i in range(device_count):
            info[f"device_{i}_memory_allocated_mb"] = (
                torch.cuda.memory_allocated(i) / 1024**2
            )
            info[f"device_{i}_memory_reserved_mb"] = (
                torch.cuda.memory_reserved(i) / 1024**2
            )
        
        return info
    
    @staticmethod
    def set_memory_fraction(fraction: float, device: int = 0):
        """
        Establecer fracción de memoria a usar.
        
        Args:
            fraction: Fracción (0.0-1.0)
            device: ID de dispositivo
        """
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(fraction, device)
            logger.info(f"Memoria GPU limitada a {fraction*100:.1f}%")

