"""
Step Learning Rate Scheduler for TruthGPT API
============================================

TensorFlow-like step learning rate scheduler implementation.
"""

import torch
import torch.optim.lr_scheduler as lr_scheduler
from typing import Optional, Dict, Any


class StepLR:
    """
    Step learning rate scheduler.
    
    Similar to tf.keras.optimizers.schedules.PiecewiseConstantDecay,
    this scheduler reduces the learning rate by a factor at specified steps.
    """
    
    def __init__(self, 
                 step_size: int,
                 gamma: float = 0.1,
                 last_epoch: int = -1,
                 name: Optional[str] = None):
        """
        Initialize StepLR scheduler.
        
        Args:
            step_size: Period of learning rate decay
            gamma: Multiplicative factor of learning rate decay
            last_epoch: The index of last epoch
            name: Optional name for the scheduler
        """
        self.step_size = step_size
        self.gamma = gamma
        self.last_epoch = last_epoch
        self.name = name or f"step_lr_{step_size}_{gamma}"
        
        self._scheduler = None
        self._optimizer = None
    
    def _create_scheduler(self, optimizer):
        """Create PyTorch scheduler."""
        self._optimizer = optimizer
        self._scheduler = lr_scheduler.StepLR(
            optimizer,
            step_size=self.step_size,
            gamma=self.gamma,
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
            'step_size': self.step_size,
            'gamma': self.gamma,
            'last_epoch': self.last_epoch
        }
    
    def __repr__(self):
        return f"StepLR(step_size={self.step_size}, gamma={self.gamma})"


