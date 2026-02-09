"""
Model Debugging Tools - Herramientas de debugging para modelos
================================================================
"""

import logging
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class DebugInfo:
    """Información de debugging"""
    layer_name: str
    input_shape: tuple
    output_shape: tuple
    input_stats: Dict[str, float]
    output_stats: Dict[str, float]
    gradient_stats: Optional[Dict[str, float]] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario"""
        return {
            "layer_name": self.layer_name,
            "input_shape": self.input_shape,
            "output_shape": self.output_shape,
            "input_stats": self.input_stats,
            "output_stats": self.output_stats,
            "gradient_stats": self.gradient_stats,
            "timestamp": self.timestamp.isoformat()
        }


class ModelDebugger:
    """Debugger de modelos"""
    
    def __init__(self):
        self.debug_info: List[DebugInfo] = []
        self.hooks = []
        self.enabled = False
    
    def enable_debugging(self, model: nn.Module):
        """Habilita debugging en un modelo"""
        self.enabled = True
        self._register_hooks(model)
        logger.info("Debugging habilitado")
    
    def disable_debugging(self):
        """Deshabilita debugging"""
        self.enabled = False
        self._remove_hooks()
        logger.info("Debugging deshabilitado")
    
    def _register_hooks(self, model: nn.Module):
        """Registra hooks de debugging"""
        def create_hook(name: str):
            def forward_hook(module, input, output):
                if not self.enabled:
                    return
                
                # Estadísticas de input
                input_tensor = input[0] if isinstance(input, tuple) else input
                if isinstance(input_tensor, torch.Tensor):
                    input_stats = {
                        "mean": input_tensor.mean().item(),
                        "std": input_tensor.std().item(),
                        "min": input_tensor.min().item(),
                        "max": input_tensor.max().item(),
                        "has_nan": torch.isnan(input_tensor).any().item(),
                        "has_inf": torch.isinf(input_tensor).any().item()
                    }
                    input_shape = tuple(input_tensor.shape)
                else:
                    input_stats = {}
                    input_shape = ()
                
                # Estadísticas de output
                if isinstance(output, torch.Tensor):
                    output_stats = {
                        "mean": output.mean().item(),
                        "std": output.std().item(),
                        "min": output.min().item(),
                        "max": output.max().item(),
                        "has_nan": torch.isnan(output).any().item(),
                        "has_inf": torch.isinf(output).any().item()
                    }
                    output_shape = tuple(output.shape)
                else:
                    output_stats = {}
                    output_shape = ()
                
                debug_info = DebugInfo(
                    layer_name=name,
                    input_shape=input_shape,
                    output_shape=output_shape,
                    input_stats=input_stats,
                    output_stats=output_stats
                )
                
                self.debug_info.append(debug_info)
            
            def backward_hook(module, grad_input, grad_output):
                if not self.enabled:
                    return
                
                # Estadísticas de gradientes
                if grad_output and grad_output[0] is not None:
                    grad = grad_output[0]
                    grad_stats = {
                        "mean": grad.mean().item(),
                        "std": grad.std().item(),
                        "min": grad.min().item(),
                        "max": grad.max().item(),
                        "has_nan": torch.isnan(grad).any().item(),
                        "has_inf": torch.isinf(grad).any().item()
                    }
                    
                    # Actualizar último debug info
                    if self.debug_info:
                        self.debug_info[-1].gradient_stats = grad_stats
            
            return forward_hook, backward_hook
        
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Solo hojas
                forward_hook, backward_hook = create_hook(name)
                self.hooks.append(module.register_forward_hook(forward_hook))
                self.hooks.append(module.register_backward_hook(backward_hook))
    
    def _remove_hooks(self):
        """Remueve hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
    
    def check_gradient_flow(self, model: nn.Module) -> Dict[str, float]:
        """Verifica el flujo de gradientes"""
        grad_flow = {}
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_mean = param.grad.abs().mean().item()
                grad_flow[name] = grad_mean
            else:
                grad_flow[name] = 0.0
        
        return grad_flow
    
    def detect_vanishing_gradients(
        self,
        model: nn.Module,
        threshold: float = 1e-6
    ) -> List[str]:
        """Detecta gradientes que desaparecen"""
        grad_flow = self.check_gradient_flow(model)
        vanishing = [name for name, grad in grad_flow.items() if grad < threshold]
        
        if vanishing:
            logger.warning(f"Gradientes desapareciendo en {len(vanishing)} capas")
        
        return vanishing
    
    def detect_exploding_gradients(
        self,
        model: nn.Module,
        threshold: float = 1e6
    ) -> List[str]:
        """Detecta gradientes que explotan"""
        grad_flow = self.check_gradient_flow(model)
        exploding = [name for name, grad in grad_flow.items() if grad > threshold]
        
        if exploding:
            logger.warning(f"Gradientes explotando en {len(exploding)} capas")
        
        return exploding
    
    def get_debug_summary(self) -> Dict[str, Any]:
        """Obtiene resumen de debugging"""
        if not self.debug_info:
            return {}
        
        layers_with_nan = [
            info.layer_name for info in self.debug_info
            if info.input_stats.get("has_nan") or info.output_stats.get("has_nan")
        ]
        
        layers_with_inf = [
            info.layer_name for info in self.debug_info
            if info.input_stats.get("has_inf") or info.output_stats.get("has_inf")
        ]
        
        return {
            "total_layers_debugged": len(self.debug_info),
            "layers_with_nan": layers_with_nan,
            "layers_with_inf": layers_with_inf,
            "debug_info": [info.to_dict() for info in self.debug_info[-10:]]  # Últimos 10
        }
    
    def clear_debug_info(self):
        """Limpia información de debugging"""
        self.debug_info.clear()




