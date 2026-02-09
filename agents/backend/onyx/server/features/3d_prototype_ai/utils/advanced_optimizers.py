"""
Advanced Optimizers and Schedulers - Optimizadores y schedulers avanzados
===========================================================================
Optimizadores personalizados y learning rate schedulers avanzados
"""

import logging
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import (
    LambdaLR, StepLR, ExponentialLR, CosineAnnealingLR,
    ReduceLROnPlateau, CosineAnnealingWarmRestarts
)
from typing import Dict, Any, Optional, Callable
import math

logger = logging.getLogger(__name__)


class AdvancedOptimizerFactory:
    """Factory para crear optimizadores avanzados"""
    
    @staticmethod
    def create_optimizer(
        model: torch.nn.Module,
        optimizer_type: str = "adamw",
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        **kwargs
    ) -> torch.optim.Optimizer:
        """Crea optimizador"""
        params = model.parameters()
        
        if optimizer_type.lower() == "adam":
            return optim.Adam(params, lr=learning_rate, weight_decay=weight_decay, **kwargs)
        elif optimizer_type.lower() == "adamw":
            return optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay, **kwargs)
        elif optimizer_type.lower() == "sgd":
            momentum = kwargs.get("momentum", 0.9)
            return optim.SGD(params, lr=learning_rate, momentum=momentum, weight_decay=weight_decay, **kwargs)
        elif optimizer_type.lower() == "rmsprop":
            return optim.RMSprop(params, lr=learning_rate, weight_decay=weight_decay, **kwargs)
        elif optimizer_type.lower() == "adagrad":
            return optim.Adagrad(params, lr=learning_rate, weight_decay=weight_decay, **kwargs)
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")


class AdvancedSchedulerFactory:
    """Factory para crear schedulers avanzados"""
    
    @staticmethod
    def create_scheduler(
        optimizer: torch.optim.Optimizer,
        scheduler_type: str = "cosine",
        **kwargs
    ) -> Any:
        """Crea scheduler"""
        if scheduler_type.lower() == "step":
            step_size = kwargs.get("step_size", 30)
            gamma = kwargs.get("gamma", 0.1)
            return StepLR(optimizer, step_size=step_size, gamma=gamma)
        
        elif scheduler_type.lower() == "exponential":
            gamma = kwargs.get("gamma", 0.95)
            return ExponentialLR(optimizer, gamma=gamma)
        
        elif scheduler_type.lower() == "cosine":
            T_max = kwargs.get("T_max", 50)
            eta_min = kwargs.get("eta_min", 0)
            return CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
        
        elif scheduler_type.lower() == "cosine_restarts":
            T_0 = kwargs.get("T_0", 10)
            T_mult = kwargs.get("T_mult", 2)
            eta_min = kwargs.get("eta_min", 0)
            return CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min)
        
        elif scheduler_type.lower() == "reduce_on_plateau":
            mode = kwargs.get("mode", "min")
            factor = kwargs.get("factor", 0.5)
            patience = kwargs.get("patience", 10)
            return ReduceLROnPlateau(optimizer, mode=mode, factor=factor, patience=patience)
        
        elif scheduler_type.lower() == "warmup_cosine":
            # Warmup + Cosine annealing
            warmup_steps = kwargs.get("warmup_steps", 1000)
            max_steps = kwargs.get("max_steps", 10000)
            
            def lr_lambda(current_step):
                if current_step < warmup_steps:
                    return float(current_step) / float(max(1, warmup_steps))
                progress = float(current_step - warmup_steps) / float(max(1, max_steps - warmup_steps))
                return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
            
            return LambdaLR(optimizer, lr_lambda)
        
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")


class LookaheadOptimizer:
    """Lookahead optimizer wrapper"""
    
    def __init__(self, optimizer: torch.optim.Optimizer, k: int = 5, alpha: float = 0.5):
        self.optimizer = optimizer
        self.k = k
        self.alpha = alpha
        self.step_count = 0
        self.slow_weights = {param.data.clone() for param in optimizer.param_groups[0]['params']}
    
    def step(self, closure=None):
        loss = self.optimizer.step(closure)
        self.step_count += 1
        
        if self.step_count % self.k == 0:
            # Update slow weights
            for slow_param, fast_param in zip(self.slow_weights, self.optimizer.param_groups[0]['params']):
                slow_param.data.add_(self.alpha * (fast_param.data - slow_param.data))
                fast_param.data.copy_(slow_param.data)
        
        return loss
    
    def zero_grad(self):
        self.optimizer.zero_grad()
    
    def state_dict(self):
        return self.optimizer.state_dict()
    
    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)




