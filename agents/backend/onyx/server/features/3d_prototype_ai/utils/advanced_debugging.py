"""
Advanced Debugging Tools - Herramientas avanzadas de debugging
===============================================================
Debugging tools para modelos y entrenamiento
"""

import logging
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional
from contextlib import contextmanager
import numpy as np

logger = logging.getLogger(__name__)


class ModelDebugger:
    """Sistema de debugging para modelos"""
    
    def __init__(self):
        self.gradient_norms: Dict[str, List[float]] = {}
        self.activation_stats: Dict[str, List[Dict[str, float]]] = {}
        self.weight_stats: Dict[str, List[Dict[str, float]]] = {}
    
    @contextmanager
    def detect_anomalies(self):
        """Detecta anomalías en autograd"""
        torch.autograd.set_detect_anomaly(True)
        try:
            yield
        finally:
            torch.autograd.set_detect_anomaly(False)
    
    def register_gradient_hooks(self, model: nn.Module):
        """Registra hooks para monitorear gradientes"""
        def gradient_hook(name):
            def hook(grad):
                if grad is not None:
                    grad_norm = grad.norm().item()
                    if name not in self.gradient_norms:
                        self.gradient_norms[name] = []
                    self.gradient_norms[name].append(grad_norm)
                    
                    # Detectar gradientes problemáticos
                    if torch.isnan(grad).any():
                        logger.warning(f"NaN gradient detected in {name}")
                    if torch.isinf(grad).any():
                        logger.warning(f"Inf gradient detected in {name}")
            return hook
        
        hooks = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                hook = param.register_hook(gradient_hook(name))
                hooks.append((name, hook))
        
        return hooks
    
    def register_activation_hooks(self, model: nn.Module):
        """Registra hooks para monitorear activaciones"""
        def activation_hook(name):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    stats = {
                        "mean": output.mean().item(),
                        "std": output.std().item(),
                        "min": output.min().item(),
                        "max": output.max().item(),
                        "nan_count": torch.isnan(output).sum().item(),
                        "inf_count": torch.isinf(output).sum().item()
                    }
                    
                    if name not in self.activation_stats:
                        self.activation_stats[name] = []
                    self.activation_stats[name].append(stats)
                    
                    # Detectar problemas
                    if stats["nan_count"] > 0:
                        logger.warning(f"NaN activations in {name}: {stats['nan_count']}")
                    if stats["inf_count"] > 0:
                        logger.warning(f"Inf activations in {name}: {stats['inf_count']}")
            return hook
        
        hooks = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.ReLU, nn.Tanh, nn.Sigmoid, nn.Linear, nn.Conv2d)):
                hook = module.register_forward_hook(activation_hook(name))
                hooks.append((name, hook))
        
        return hooks
    
    def check_weight_stats(self, model: nn.Module) -> Dict[str, Dict[str, float]]:
        """Verifica estadísticas de pesos"""
        stats = {}
        
        for name, param in model.named_parameters():
            if param.data is not None:
                stats[name] = {
                    "mean": param.data.mean().item(),
                    "std": param.data.std().item(),
                    "min": param.data.min().item(),
                    "max": param.data.max().item(),
                    "nan_count": torch.isnan(param.data).sum().item(),
                    "inf_count": torch.isinf(param.data).sum().item()
                }
        
        return stats
    
    def detect_dead_neurons(self, model: nn.Module, threshold: float = 1e-6) -> List[str]:
        """Detecta neuronas muertas"""
        dead_neurons = []
        
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                if hasattr(module, "weight"):
                    weight_norm = module.weight.data.norm(dim=0)
                    dead = (weight_norm < threshold).sum().item()
                    if dead > 0:
                        dead_neurons.append(f"{name}: {dead} dead neurons")
        
        return dead_neurons
    
    def get_gradient_summary(self) -> Dict[str, Dict[str, float]]:
        """Resumen de gradientes"""
        summary = {}
        
        for name, norms in self.gradient_norms.items():
            if norms:
                summary[name] = {
                    "mean": np.mean(norms),
                    "std": np.std(norms),
                    "min": np.min(norms),
                    "max": np.max(norms),
                    "count": len(norms)
                }
        
        return summary
    
    def clear_stats(self):
        """Limpia estadísticas"""
        self.gradient_norms.clear()
        self.activation_stats.clear()
        self.weight_stats.clear()




