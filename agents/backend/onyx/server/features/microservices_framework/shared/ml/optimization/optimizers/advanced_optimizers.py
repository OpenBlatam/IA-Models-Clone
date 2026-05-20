"""
Advanced Optimizers
Specialized optimizer implementations and utilities.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from ..core.factories import OptimizerFactory


class OptimizerWithWarmup:
    """Optimizer wrapper with warmup support."""
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int = 1000,
        warmup_lr: float = 0.0,
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.warmup_lr = warmup_lr
        self.base_lr = optimizer.param_groups[0]['lr']
        self.current_step = 0
    
    def step(self):
        """Step with warmup."""
        if self.current_step < self.warmup_steps:
            # Warmup phase
            lr = self.warmup_lr + (self.base_lr - self.warmup_lr) * (
                self.current_step / self.warmup_steps
            )
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        
        self.optimizer.step()
        self.current_step += 1
    
    def zero_grad(self):
        """Zero gradients."""
        self.optimizer.zero_grad()
    
    def state_dict(self):
        """Get state dict."""
        return {
            "optimizer": self.optimizer.state_dict(),
            "current_step": self.current_step,
        }
    
    def load_state_dict(self, state_dict):
        """Load state dict."""
        self.optimizer.load_state_dict(state_dict["optimizer"])
        self.current_step = state_dict["current_step"]


class LookaheadOptimizer:
    """Lookahead optimizer wrapper."""
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        k: int = 5,
        alpha: float = 0.5,
    ):
        self.optimizer = optimizer
        self.k = k
        self.alpha = alpha
        self.step_count = 0
        self.slow_weights = {
            name: param.clone()
            for name, param in self._get_named_parameters()
        }
    
    def _get_named_parameters(self):
        """Get named parameters from optimizer."""
        for group in self.optimizer.param_groups:
            for p in group['params']:
                yield id(p), p
    
    def step(self):
        """Step with lookahead."""
        self.optimizer.step()
        self.step_count += 1
        
        if self.step_count % self.k == 0:
            # Update slow weights
            for param_id, param in self._get_named_parameters():
                if param_id in self.slow_weights:
                    slow_param = self.slow_weights[param_id]
                    slow_param.data.add_(
                        self.alpha * (param.data - slow_param.data)
                    )
                    param.data.copy_(slow_param.data)
    
    def zero_grad(self):
        """Zero gradients."""
        self.optimizer.zero_grad()
    
    def state_dict(self):
        """Get state dict."""
        return {
            "optimizer": self.optimizer.state_dict(),
            "slow_weights": self.slow_weights,
            "step_count": self.step_count,
        }
    
    def load_state_dict(self, state_dict):
        """Load state dict."""
        self.optimizer.load_state_dict(state_dict["optimizer"])
        self.slow_weights = state_dict["slow_weights"]
        self.step_count = state_dict["step_count"]


def create_optimizer_with_schedule(
    model: nn.Module,
    optimizer_type: str = "adamw",
    learning_rate: float = 5e-5,
    warmup_steps: int = 1000,
    **optimizer_kwargs
) -> tuple:
    """
    Create optimizer with warmup schedule.
    
    Returns:
        Tuple of (optimizer, scheduler)
    """
    optimizer = OptimizerFactory.create(
        optimizer_type,
        model,
        learning_rate,
        **optimizer_kwargs
    )
    
    if warmup_steps > 0:
        optimizer = OptimizerWithWarmup(optimizer, warmup_steps=warmup_steps)
    
    from ...schedulers.learning_rate_scheduler import LearningRateSchedulerFactory
    scheduler = LearningRateSchedulerFactory.create_scheduler(
        optimizer.optimizer if isinstance(optimizer, OptimizerWithWarmup) else optimizer,
        "cosine",
        num_training_steps=10000,  # Should be passed as parameter
        num_warmup_steps=warmup_steps,
    )
    
    return optimizer, scheduler



