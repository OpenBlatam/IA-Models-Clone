"""
Activation Analyzer
===================

Análisis de activaciones para debugging.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)


class ActivationAnalyzer:
    """
    Analizador de activaciones.
    """
    
    def __init__(self, model: nn.Module):
        """
        Inicializar analizador.
        
        Args:
            model: Modelo
        """
        self.model = model
        self.activations = {}
        self.hooks = []
    
    def register_hooks(self):
        """Registrar hooks para capturar activaciones."""
        def get_activation_hook(name):
            def hook(module, input, output):
                self.activations[name] = {
                    "output": output.detach().cpu(),
                    "mean": output.mean().item(),
                    "std": output.std().item(),
                    "min": output.min().item(),
                    "max": output.max().item(),
                    "has_nan": torch.isnan(output).any().item(),
                    "has_inf": torch.isinf(output).any().item(),
                    "dead_neurons": (output == 0).sum().item() if output.numel() > 0 else 0
                }
            return hook
        
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.ReLU, nn.ELU, nn.GELU, nn.Linear)):
                hook = module.register_forward_hook(get_activation_hook(name))
                self.hooks.append(hook)
    
    def remove_hooks(self):
        """Remover hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def analyze(self, input_tensor: torch.Tensor) -> Dict[str, Any]:
        """
        Analizar activaciones.
        
        Args:
            input_tensor: Input
            
        Returns:
            Análisis de activaciones
        """
        self.activations.clear()
        self.model.eval()
        
        with torch.no_grad():
            _ = self.model(input_tensor)
        
        return {
            "activations": self.activations,
            "summary": self._generate_summary()
        }
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generar resumen."""
        if not self.activations:
            return {}
        
        dead_neurons_total = sum(
            act.get("dead_neurons", 0) for act in self.activations.values()
        )
        
        has_nan_layers = [
            name for name, act in self.activations.items()
            if act.get("has_nan", False)
        ]
        
        has_inf_layers = [
            name for name, act in self.activations.items()
            if act.get("has_inf", False)
        ]
        
        return {
            "total_layers": len(self.activations),
            "dead_neurons_total": dead_neurons_total,
            "layers_with_nan": has_nan_layers,
            "layers_with_inf": has_inf_layers
        }


def analyze_activations(model: nn.Module, input_tensor: torch.Tensor) -> Dict[str, Any]:
    """
    Función helper para analizar activaciones.
    
    Args:
        model: Modelo
        input_tensor: Input
        
    Returns:
        Análisis
    """
    analyzer = ActivationAnalyzer(model)
    analyzer.register_hooks()
    result = analyzer.analyze(input_tensor)
    analyzer.remove_hooks()
    return result

