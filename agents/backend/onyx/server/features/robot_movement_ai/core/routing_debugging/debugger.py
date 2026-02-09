"""
Model Debugger
==============

Herramientas de debugging para modelos.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class DebugConfig:
    """Configuración de debugging."""
    detect_anomaly: bool = True
    check_gradients: bool = True
    check_activations: bool = True
    check_weights: bool = True
    log_level: str = "INFO"


class ModelDebugger:
    """
    Debugger de modelos.
    """
    
    def __init__(self, model: nn.Module, config: Optional[DebugConfig] = None):
        """
        Inicializar debugger.
        
        Args:
            model: Modelo a debuggear
            config: Configuración (opcional)
        """
        self.model = model
        self.config = config or DebugConfig()
        self.hooks = []
    
    def enable_anomaly_detection(self):
        """Habilitar detección de anomalías."""
        if self.config.detect_anomaly:
            torch.autograd.set_detect_anomaly(True)
            logger.info("Detección de anomalías habilitada")
    
    def disable_anomaly_detection(self):
        """Deshabilitar detección de anomalías."""
        torch.autograd.set_detect_anomaly(False)
        logger.info("Detección de anomalías deshabilitada")
    
    def register_hooks(self):
        """Registrar hooks para debugging."""
        def forward_hook(name):
            def hook(module, input, output):
                if torch.isnan(output).any():
                    logger.warning(f"NaN detectado en {name} output")
                if torch.isinf(output).any():
                    logger.warning(f"Inf detectado en {name} output")
            return hook
        
        def backward_hook(name):
            def hook(module, grad_input, grad_output):
                if grad_output[0] is not None:
                    if torch.isnan(grad_output[0]).any():
                        logger.warning(f"NaN en gradiente de {name}")
                    if torch.isinf(grad_output[0]).any():
                        logger.warning(f"Inf en gradiente de {name}")
            return hook
        
        # Registrar hooks
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                hook_f = module.register_forward_hook(forward_hook(name))
                hook_b = module.register_backward_hook(backward_hook(name))
                self.hooks.extend([hook_f, hook_b])
    
    def remove_hooks(self):
        """Remover hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def check_weights(self) -> Dict[str, Any]:
        """
        Verificar pesos del modelo.
        
        Returns:
            Estadísticas de pesos
        """
        stats = {}
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                stats[name] = {
                    "mean": param.data.mean().item(),
                    "std": param.data.std().item(),
                    "min": param.data.min().item(),
                    "max": param.data.max().item(),
                    "has_nan": torch.isnan(param.data).any().item(),
                    "has_inf": torch.isinf(param.data).any().item()
                }
        
        return stats
    
    def diagnose(self, input_tensor: torch.Tensor) -> Dict[str, Any]:
        """
        Diagnóstico completo del modelo.
        
        Args:
            input_tensor: Input de prueba
            
        Returns:
            Diagnóstico
        """
        self.model.eval()
        
        diagnosis = {
            "weights": self.check_weights(),
            "forward_pass": None,
            "backward_pass": None
        }
        
        # Forward pass
        try:
            with torch.no_grad():
                output = self.model(input_tensor)
                diagnosis["forward_pass"] = {
                    "success": True,
                    "output_shape": list(output.shape),
                    "has_nan": torch.isnan(output).any().item(),
                    "has_inf": torch.isinf(output).any().item(),
                    "output_range": (output.min().item(), output.max().item())
                }
        except Exception as e:
            diagnosis["forward_pass"] = {
                "success": False,
                "error": str(e)
            }
        
        # Backward pass (si hay gradientes)
        if input_tensor.requires_grad:
            try:
                self.model.train()
                output = self.model(input_tensor)
                loss = output.mean()
                loss.backward()
                
                diagnosis["backward_pass"] = {
                    "success": True,
                    "gradient_norms": {}
                }
                
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        norm = param.grad.norm().item()
                        diagnosis["backward_pass"]["gradient_norms"][name] = norm
            except Exception as e:
                diagnosis["backward_pass"] = {
                    "success": False,
                    "error": str(e)
                }
        
        return diagnosis

