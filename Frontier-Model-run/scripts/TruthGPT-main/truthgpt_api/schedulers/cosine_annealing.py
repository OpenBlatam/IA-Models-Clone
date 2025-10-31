"""
Cosine Annealing Learning Rate Scheduler for TruthGPT API
========================================================

TensorFlow-like cosine annealing learning rate scheduler implementation.
"""

import torch
import torch.optim.lr_scheduler as lr_scheduler
from typing import Optional, Dict, Any


class CosineAnnealingLR:
    """
    Cosine annealing learning rate scheduler.
    
    Similar to tf.keras.optimizers.schedules.CosineDecay,
    this scheduler reduces the learning rate following a cosine schedule.
    """
    
    def __init__(self, 
                 T_max: int,
                 eta_min: float = 0.0,
                 last_epoch: int = -1,
                 name: Optional[str] = None):
        """
        Initialize CosineAnnealingLR scheduler.
        
        Args:
            T_max: Maximum number of iterations
            eta_min: Minimum learning rate
            last_epoch: The index of last epoch
            name: Optional name for the scheduler
        """
        self.T_max = T_max
        self.eta_min = eta_min
        self.last_epoch = last_epoch
        self.name = name or f"cosine_annealing_{T_max}_{eta_min}"
        
        self._scheduler = None
        self._optimizer = None
    
    def _create_scheduler(self, optimizer):
        """Create PyTorch scheduler."""
        self._optimizer = optimizer
        self._scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.T_max,
            eta_min=self.eta_min,
            last_epoch=self.last_epoch
        )
        return self._scheduler
    
    def __call__(self, optimizer):
        """Create scheduler for given optimizer."""
        return self._create_scheduler(optimizer)
    
    def step(self):
        """Step the scheduler."""
        if self._scheduler:
            self._scheduler.step()
    
    def get_last_lr(self):
        """Get the last learning rate."""
        if self._scheduler:
            return self._scheduler.get_last_lr()
        return []
    
    def get_config(self) -> Dict[str, Any]:
        """Get scheduler configuration."""
        return {
            'name': self.name,
            'T_max': self.T_max,
            'eta_min': self.eta_min,
            'last_epoch': self.last_epoch
        }
    
    def __repr__(self):
        return f"CosineAnnealingLR(T_max={self.T_max}, eta_min={self.eta_min})"









