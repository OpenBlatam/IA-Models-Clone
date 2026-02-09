"""
Gradient Clipping Manager - Gestor de gradient clipping
========================================================
"""

import logging
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ClippingMethod(Enum):
    """Métodos de gradient clipping"""
    NORM = "norm"  # Clip by norm
    VALUE = "value"  # Clip by value
    GLOBAL_NORM = "global_norm"  # Clip global norm


@dataclass
class GradientClippingConfig:
    """Configuración de gradient clipping"""
    method: ClippingMethod = ClippingMethod.NORM
    max_norm: float = 1.0
    max_value: float = 0.5
    clip_value: Optional[float] = None  # Para value clipping


class GradientClipper:
    """Gestor de gradient clipping"""
    
    def __init__(self, config: GradientClippingConfig):
        self.config = config
        self.clipping_history: List[Dict[str, float]] = []
    
    def clip_gradients(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None
    ) -> Dict[str, float]:
        """Aplica gradient clipping"""
        if self.config.method == ClippingMethod.NORM:
            return self._clip_by_norm(model)
        elif self.config.method == ClippingMethod.VALUE:
            return self._clip_by_value(model)
        elif self.config.method == ClippingMethod.GLOBAL_NORM:
            return self._clip_global_norm(model)
        else:
            raise ValueError(f"Método {self.config.method} no soportado")
    
    def _clip_by_norm(self, model: nn.Module) -> Dict[str, float]:
        """Clip por norma de cada parámetro"""
        total_norm = 0.0
        clipped_count = 0
        
        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                
                clip_coef = self.config.max_norm / (param_norm + 1e-6)
                if clip_coef < 1:
                    param.grad.data.mul_(clip_coef)
                    clipped_count += 1
        
        total_norm = total_norm ** (1. / 2)
        
        result = {
            "total_norm": total_norm,
            "clipped_count": clipped_count,
            "was_clipped": total_norm > self.config.max_norm
        }
        
        self.clipping_history.append(result)
        return result
    
    def _clip_by_value(self, model: nn.Module) -> Dict[str, float]:
        """Clip por valor"""
        clipped_count = 0
        max_value = self.config.clip_value or self.config.max_value
        
        for param in model.parameters():
            if param.grad is not None:
                if torch.any(torch.abs(param.grad) > max_value):
                    param.grad.data.clamp_(-max_value, max_value)
                    clipped_count += 1
        
        result = {
            "clipped_count": clipped_count,
            "max_value": max_value,
            "was_clipped": clipped_count > 0
        }
        
        self.clipping_history.append(result)
        return result
    
    def _clip_global_norm(self, model: nn.Module) -> Dict[str, float]:
        """Clip por norma global"""
        parameters = [p for p in model.parameters() if p.grad is not None]
        
        if len(parameters) == 0:
            return {"total_norm": 0.0, "was_clipped": False}
        
        total_norm = torch.norm(
            torch.stack([torch.norm(p.grad.detach()) for p in parameters])
        )
        
        clip_coef = self.config.max_norm / (total_norm + 1e-6)
        
        if clip_coef < 1:
            for p in parameters:
                p.grad.detach().mul_(clip_coef)
        
        result = {
            "total_norm": total_norm.item(),
            "clip_coef": clip_coef.item() if clip_coef < 1 else 1.0,
            "was_clipped": clip_coef < 1
        }
        
        self.clipping_history.append(result)
        return result
    
    def get_clipping_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas de clipping"""
        if not self.clipping_history:
            return {}
        
        clipped_count = sum(1 for h in self.clipping_history if h.get("was_clipped", False))
        total_norms = [h.get("total_norm", 0) for h in self.clipping_history if "total_norm" in h]
        
        return {
            "total_clips": len(self.clipping_history),
            "clipped_ratio": clipped_count / len(self.clipping_history) if self.clipping_history else 0,
            "avg_norm": sum(total_norms) / len(total_norms) if total_norms else 0,
            "max_norm": max(total_norms) if total_norms else 0,
            "min_norm": min(total_norms) if total_norms else 0
        }




