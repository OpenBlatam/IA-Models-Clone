"""
Optimizadores avanzados
"""

import torch
from torch.optim import Optimizer
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class Lookahead:
    """Lookahead optimizer wrapper"""
    
    def __init__(
        self,
        optimizer: Optimizer,
        k: int = 5,
        alpha: float = 0.5
    ):
        self.optimizer = optimizer
        self.k = k
        self.alpha = alpha
        self.step_count = 0
        self.slow_weights = {}
        
        # Inicializar slow weights
        for group in self.optimizer.param_groups:
            for p in group['params']:
                self.slow_weights[p] = p.data.clone()
    
    def step(self, closure=None):
        """Step con lookahead"""
        loss = self.optimizer.step(closure)
        self.step_count += 1
        
        # Cada k pasos, actualizar slow weights
        if self.step_count % self.k == 0:
            for group in self.optimizer.param_groups:
                for p in group['params']:
                    if p in self.slow_weights:
                        # Interpolación entre slow y fast weights
                        self.slow_weights[p] += self.alpha * (p.data - self.slow_weights[p])
                        p.data.copy_(self.slow_weights[p])
        
        return loss
    
    def zero_grad(self):
        """Zero gradients"""
        self.optimizer.zero_grad()
    
    def state_dict(self):
        """State dict"""
        return {
            'optimizer': self.optimizer.state_dict(),
            'slow_weights': self.slow_weights,
            'step_count': self.step_count
        }
    
    def load_state_dict(self, state_dict):
        """Load state dict"""
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.slow_weights = state_dict['slow_weights']
        self.step_count = state_dict['step_count']


class RAdam:
    """Rectified Adam optimizer (simplificado)"""
    
    @staticmethod
    def create_radam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-8):
        """Crea RAdam optimizer"""
        # En producción usaría implementación completa
        # Por ahora usar Adam como base
        return torch.optim.Adam(params, lr=lr, betas=betas, eps=eps)


class AdaBelief:
    """AdaBelief optimizer (simplificado)"""
    
    @staticmethod
    def create_adabelief(params, lr=0.001, betas=(0.9, 0.999), eps=1e-8):
        """Crea AdaBelief optimizer"""
        # En producción usaría implementación completa
        # Por ahora usar Adam como base
        return torch.optim.Adam(params, lr=lr, betas=betas, eps=eps)


class OptimizerFactory:
    """Factory para crear optimizadores avanzados"""
    
    @staticmethod
    def create_optimizer(
        optimizer_type: str,
        params,
        lr: float = 0.001,
        **kwargs
    ) -> Optimizer:
        """
        Crea optimizador según tipo
        
        Args:
            optimizer_type: Tipo de optimizador
            params: Parámetros del modelo
            lr: Learning rate
            **kwargs: Parámetros adicionales
            
        Returns:
            Optimizador
        """
        if optimizer_type == "adam":
            return torch.optim.Adam(params, lr=lr, **kwargs)
        elif optimizer_type == "adamw":
            return torch.optim.AdamW(params, lr=lr, **kwargs)
        elif optimizer_type == "sgd":
            return torch.optim.SGD(params, lr=lr, **kwargs)
        elif optimizer_type == "radam":
            return RAdam.create_radam(params, lr=lr, **kwargs)
        elif optimizer_type == "adabelief":
            return AdaBelief.create_adabelief(params, lr=lr, **kwargs)
        elif optimizer_type == "lookahead_adam":
            base_optimizer = torch.optim.Adam(params, lr=lr, **kwargs)
            return Lookahead(base_optimizer, k=kwargs.get('k', 5), alpha=kwargs.get('alpha', 0.5))
        else:
            raise ValueError(f"Optimizer type {optimizer_type} no soportado")




