"""
Memory Optimization
===================

Optimizaciones de memoria para entrenamiento e inferencia.
"""

import torch
import torch.nn as nn
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class MemoryOptimizer:
    """
    Optimizador de memoria.
    """
    
    @staticmethod
    def enable_memory_efficient_attention():
        """Habilitar atención eficiente en memoria."""
        try:
            torch.backends.cuda.enable_flash_sdp(True)
            logger.info("Flash attention habilitado")
        except:
            logger.warning("Flash attention no disponible")
    
    @staticmethod
    def clear_cache():
        """Limpiar cache de GPU."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    @staticmethod
    def get_memory_usage() -> dict:
        """
        Obtener uso de memoria.
        
        Returns:
            Diccionario con uso de memoria
        """
        if not torch.cuda.is_available():
            return {"cuda_available": False}
        
        return {
            "allocated_mb": torch.cuda.memory_allocated() / 1024**2,
            "reserved_mb": torch.cuda.memory_reserved() / 1024**2,
            "max_allocated_mb": torch.cuda.max_memory_allocated() / 1024**2
        }
    
    @staticmethod
    def optimize_model_memory(model: nn.Module):
        """
        Optimizar memoria del modelo.
        
        Args:
            model: Modelo
        """
        # Habilitar optimizaciones de memoria
        if hasattr(torch, 'set_grad_enabled'):
            torch.set_grad_enabled(False)  # Para inferencia
        
        # Compilar con optimizaciones de memoria
        if hasattr(torch, 'compile'):
            try:
                model = torch.compile(model, mode="reduce-overhead")
            except:
                pass
        
        return model


class GradientCheckpointing:
    """
    Gradient checkpointing para ahorrar memoria.
    """
    
    @staticmethod
    def enable_checkpointing(model: nn.Module):
        """
        Habilitar gradient checkpointing.
        
        Args:
            model: Modelo
        """
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing habilitado")
        else:
            logger.warning("Modelo no soporta gradient checkpointing")
    
    @staticmethod
    def disable_checkpointing(model: nn.Module):
        """Deshabilitar gradient checkpointing."""
        if hasattr(model, 'gradient_checkpointing_disable'):
            model.gradient_checkpointing_disable()

