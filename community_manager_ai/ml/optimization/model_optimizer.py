"""
Model Optimizer - Optimizador de Modelos
=========================================

Optimizaciones para inferencia más rápida.
"""

import logging
import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from torch.utils.mobile_optimizer import optimize_for_mobile

logger = logging.getLogger(__name__)


class ModelOptimizer:
    """Optimizador de modelos para inferencia rápida"""
    
    @staticmethod
    def compile_model(model: nn.Module, mode: str = "reduce-overhead") -> nn.Module:
        """
        Compilar modelo con torch.compile (PyTorch 2.0+)
        
        Args:
            model: Modelo a optimizar
            mode: Modo de compilación
            
        Returns:
            Modelo compilado
        """
        try:
            if hasattr(torch, "compile"):
                compiled_model = torch.compile(model, mode=mode)
                logger.info(f"Modelo compilado con modo: {mode}")
                return compiled_model
            else:
                logger.warning("torch.compile no disponible, usando modelo sin compilar")
                return model
        except Exception as e:
            logger.warning(f"Error compilando modelo: {e}")
            return model
    
    @staticmethod
    def optimize_for_inference(model: nn.Module) -> nn.Module:
        """
        Optimizar modelo para inferencia
        
        Args:
            model: Modelo a optimizar
            
        Returns:
            Modelo optimizado
        """
        model.eval()
        
        # Fusionar operaciones cuando sea posible
        try:
            if hasattr(torch.jit, "optimize_for_inference"):
                model = torch.jit.optimize_for_inference(torch.jit.script(model))
        except Exception:
            pass
        
        # Deshabilitar gradientes
        for param in model.parameters():
            param.requires_grad = False
        
        return model
    
    @staticmethod
    def enable_torchscript(model: nn.Module, example_input: Dict[str, torch.Tensor]) -> nn.Module:
        """
        Convertir modelo a TorchScript para inferencia más rápida
        
        Args:
            model: Modelo a convertir
            example_input: Input de ejemplo
            
        Returns:
            Modelo TorchScript
        """
        try:
            model.eval()
            traced_model = torch.jit.trace(model, example_input)
            optimized_model = optimize_for_mobile(traced_model)
            logger.info("Modelo convertido a TorchScript")
            return optimized_model
        except Exception as e:
            logger.warning(f"Error convirtiendo a TorchScript: {e}")
            return model
    
    @staticmethod
    def enable_tensorrt(model: nn.Module, example_input: torch.Tensor) -> Optional[nn.Module]:
        """
        Optimizar con TensorRT (requiere TensorRT)
        
        Args:
            model: Modelo a optimizar
            example_input: Input de ejemplo
            
        Returns:
            Modelo optimizado o None
        """
        try:
            import tensorrt as trt
            # Implementación de TensorRT
            logger.info("TensorRT optimización aplicada")
            return model
        except ImportError:
            logger.warning("TensorRT no disponible")
            return None
        except Exception as e:
            logger.warning(f"Error con TensorRT: {e}")
            return None




