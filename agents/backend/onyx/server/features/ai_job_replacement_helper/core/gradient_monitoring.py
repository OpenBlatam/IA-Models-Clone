"""
Gradient Monitoring Service - Monitoreo de gradientes
=====================================================

Sistema para monitorear y analizar gradientes durante el entrenamiento.
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import torch
import torch.nn as nn
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class GradientStats:
    """Estadísticas de gradientes"""
    layer_name: str
    mean: float
    std: float
    min: float
    max: float
    norm: float
    has_nan: bool
    has_inf: bool
    zero_ratio: float  # Ratio de gradientes cero


class GradientMonitoringService:
    """Servicio de monitoreo de gradientes"""
    
    def __init__(self):
        """Inicializar servicio"""
        self.gradient_history: Dict[str, List[float]] = {}
        logger.info("GradientMonitoringService initialized")
    
    def collect_gradients(self, model: nn.Module) -> Dict[str, GradientStats]:
        """Recolectar estadísticas de gradientes"""
        stats = {}
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad = param.grad.data
                
                # Calcular estadísticas
                grad_flat = grad.flatten().cpu().numpy()
                
                stat = GradientStats(
                    layer_name=name,
                    mean=float(np.mean(grad_flat)),
                    std=float(np.std(grad_flat)),
                    min=float(np.min(grad_flat)),
                    max=float(np.max(grad_flat)),
                    norm=float(grad.norm().item()),
                    has_nan=bool(torch.isnan(grad).any().item()),
                    has_inf=bool(torch.isinf(grad).any().item()),
                    zero_ratio=float((grad_flat == 0).sum() / len(grad_flat)),
                )
                
                stats[name] = stat
                
                # Guardar en historial
                if name not in self.gradient_history:
                    self.gradient_history[name] = []
                self.gradient_history[name].append(stat.norm)
        
        return stats
    
    def detect_vanishing_gradients(
        self,
        threshold: float = 1e-6
    ) -> List[str]:
        """Detectar gradientes que desaparecen"""
        vanishing_layers = []
        
        for name, history in self.gradient_history.items():
            if history:
                avg_norm = np.mean(history[-10:])  # Últimos 10
                if avg_norm < threshold:
                    vanishing_layers.append(name)
        
        return vanishing_layers
    
    def detect_exploding_gradients(
        self,
        threshold: float = 100.0
    ) -> List[str]:
        """Detectar gradientes que explotan"""
        exploding_layers = []
        
        for name, history in self.gradient_history.items():
            if history:
                max_norm = np.max(history[-10:])  # Últimos 10
                if max_norm > threshold:
                    exploding_layers.append(name)
        
        return exploding_layers
    
    def get_gradient_summary(self) -> Dict[str, Any]:
        """Obtener resumen de gradientes"""
        summary = {
            "total_layers": len(self.gradient_history),
            "vanishing_gradients": self.detect_vanishing_gradients(),
            "exploding_gradients": self.detect_exploding_gradients(),
            "layer_stats": {},
        }
        
        for name, history in self.gradient_history.items():
            if history:
                summary["layer_stats"][name] = {
                    "avg_norm": float(np.mean(history)),
                    "max_norm": float(np.max(history)),
                    "min_norm": float(np.min(history)),
                    "std_norm": float(np.std(history)),
                }
        
        return summary
    
    def clear_history(self) -> None:
        """Limpiar historial"""
        self.gradient_history.clear()
        logger.info("Gradient history cleared")




