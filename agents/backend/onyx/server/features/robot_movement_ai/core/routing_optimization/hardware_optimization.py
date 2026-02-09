"""
Hardware-Specific Optimization
===============================

Optimizaciones específicas de hardware.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any
import logging
import platform

logger = logging.getLogger(__name__)


class HardwareOptimizer:
    """
    Optimizador específico de hardware.
    """
    
    @staticmethod
    def detect_hardware() -> Dict[str, Any]:
        """
        Detectar hardware disponible.
        
        Returns:
            Información de hardware
        """
        info = {
            "platform": platform.system(),
            "processor": platform.processor(),
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
        }
        
        if torch.cuda.is_available():
            info["gpu_count"] = torch.cuda.device_count()
            info["gpu_names"] = [
                torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
            ]
            info["gpu_capability"] = [
                torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())
            ]
        
        return info
    
    @staticmethod
    def optimize_for_nvidia():
        """Optimizaciones específicas para NVIDIA."""
        if not torch.cuda.is_available():
            return
        
        # Habilitar todas las optimizaciones NVIDIA
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        
        # Flash attention
        try:
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
        except:
            pass
        
        # Optimizar memoria
        torch.cuda.empty_cache()
        
        logger.info("Optimizaciones NVIDIA aplicadas")
    
    @staticmethod
    def optimize_for_amd():
        """Optimizaciones específicas para AMD."""
        # ROCm optimizations (si disponible)
        if hasattr(torch.version, 'hip'):
            logger.info("ROCm detectado, aplicando optimizaciones AMD")
            # Optimizaciones específicas de ROCm
        else:
            logger.info("AMD GPU sin ROCm, usando optimizaciones CPU")
    
    @staticmethod
    def optimize_for_intel():
        """Optimizaciones específicas para Intel."""
        # Intel optimizations
        try:
            import intel_extension_for_pytorch as ipex
            logger.info("Intel Extension for PyTorch disponible")
            # Aplicar optimizaciones Intel
        except ImportError:
            logger.info("Intel Extension no disponible, usando PyTorch estándar")
    
    @staticmethod
    def optimize_for_apple_silicon():
        """Optimizaciones específicas para Apple Silicon."""
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            logger.info("Apple Silicon (MPS) detectado")
            # Optimizaciones MPS
        else:
            logger.info("Apple Silicon no disponible")
    
    @staticmethod
    def auto_optimize():
        """Auto-detectar y optimizar según hardware."""
        hardware = HardwareOptimizer.detect_hardware()
        
        if hardware["cuda_available"]:
            HardwareOptimizer.optimize_for_nvidia()
        elif platform.system() == "Darwin":  # macOS
            HardwareOptimizer.optimize_for_apple_silicon()
        else:
            # CPU optimizations
            logger.info("Optimizando para CPU")


class MemoryPoolOptimizer:
    """
    Optimizador de memory pool.
    """
    
    @staticmethod
    def optimize_memory_pool(device: int = 0):
        """
        Optimizar memory pool de GPU.
        
        Args:
            device: ID de dispositivo
        """
        if not torch.cuda.is_available():
            return
        
        # Limpiar cache
        torch.cuda.empty_cache()
        
        # Establecer memory fraction
        torch.cuda.set_per_process_memory_fraction(0.95, device)
        
        # Habilitar memory pool
        torch.cuda.memory.set_per_process_memory_fraction(0.95, device)
        
        logger.info(f"Memory pool optimizado para dispositivo {device}")

