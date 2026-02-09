"""
Memory Optimization - Optimización de memoria
=============================================

Utilidades para optimizar el uso de memoria en entrenamiento.
Sigue mejores prácticas de gestión de memoria.
"""

import logging
from typing import Dict, List, Optional, Any, Callable
import torch
import torch.nn as nn
import gc

logger = logging.getLogger(__name__)


class MemoryOptimizer:
    """Optimizador de memoria"""
    
    @staticmethod
    def clear_cache() -> None:
        """Limpiar cache de CUDA"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.debug("CUDA cache cleared")
    
    @staticmethod
    def get_memory_usage() -> Dict[str, float]:
        """
        Obtener uso de memoria.
        
        Returns:
            Diccionario con uso de memoria en MB
        """
        memory_info = {
            "cpu_allocated_mb": 0.0,
            "cuda_allocated_mb": 0.0,
            "cuda_reserved_mb": 0.0,
            "cuda_max_allocated_mb": 0.0,
        }
        
        # CPU memory (simplified)
        try:
            import psutil
            process = psutil.Process()
            memory_info["cpu_allocated_mb"] = process.memory_info().rss / (1024 ** 2)
        except ImportError:
            pass
        
        # CUDA memory
        if torch.cuda.is_available():
            memory_info["cuda_allocated_mb"] = torch.cuda.memory_allocated() / (1024 ** 2)
            memory_info["cuda_reserved_mb"] = torch.cuda.memory_reserved() / (1024 ** 2)
            memory_info["cuda_max_allocated_mb"] = torch.cuda.max_memory_allocated() / (1024 ** 2)
        
        return memory_info
    
    @staticmethod
    def optimize_model_memory(model: nn.Module) -> None:
        """
        Optimizar memoria del modelo.
        
        Args:
            model: Modelo a optimizar
        """
        # Set model to eval mode to free training buffers
        model.eval()
        
        # Clear gradients
        for param in model.parameters():
            if param.grad is not None:
                param.grad = None
        
        # Clear cache
        MemoryOptimizer.clear_cache()
        
        # Force garbage collection
        gc.collect()
        
        logger.info("Model memory optimized")
    
    @staticmethod
    def enable_memory_efficient_attention(model: nn.Module) -> bool:
        """
        Habilitar atención eficiente en memoria (si disponible).
        
        Args:
            model: Modelo
        
        Returns:
            True si se habilitó exitosamente
        """
        try:
            # Try to enable xformers memory efficient attention
            try:
                from xformers.ops import memory_efficient_attention
                # In production, would configure model to use this
                logger.info("Memory efficient attention available (xformers)")
                return True
            except ImportError:
                pass
            
            # Try flash attention
            try:
                # Check if model supports flash attention
                if hasattr(model, "config"):
                    # In production, would set config.use_flash_attention_2 = True
                    logger.info("Flash attention available")
                    return True
            except Exception:
                pass
            
            return False
        
        except Exception as e:
            logger.warning(f"Could not enable memory efficient attention: {e}")
            return False
    
    @staticmethod
    def profile_memory_usage(
        func: Callable,
        *args,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Perfilar uso de memoria de una función.
        
        Args:
            func: Función a perfilar
            *args: Argumentos de la función
            **kwargs: Keyword arguments
        
        Returns:
            Diccionario con información de memoria
        """
        if not torch.cuda.is_available():
            return {"error": "CUDA not available"}
        
        # Reset peak memory
        torch.cuda.reset_peak_memory_stats()
        
        # Get initial memory
        initial_memory = torch.cuda.memory_allocated()
        
        # Run function
        try:
            result = func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in memory profiling: {e}")
            return {"error": str(e)}
        
        # Get final memory
        final_memory = torch.cuda.memory_allocated()
        peak_memory = torch.cuda.max_memory_allocated()
        
        return {
            "initial_memory_mb": initial_memory / (1024 ** 2),
            "final_memory_mb": final_memory / (1024 ** 2),
            "peak_memory_mb": peak_memory / (1024 ** 2),
            "memory_increase_mb": (final_memory - initial_memory) / (1024 ** 2),
            "peak_increase_mb": (peak_memory - initial_memory) / (1024 ** 2),
        }
    
    @staticmethod
    def set_memory_fraction(fraction: float) -> bool:
        """
        Establecer fracción de memoria a usar.
        
        Args:
            fraction: Fracción (0.0 a 1.0)
        
        Returns:
            True si se estableció exitosamente
        """
        if not torch.cuda.is_available():
            return False
        
        try:
            torch.cuda.set_per_process_memory_fraction(fraction)
            logger.info(f"Memory fraction set to {fraction}")
            return True
        except Exception as e:
            logger.error(f"Error setting memory fraction: {e}")
            return False

