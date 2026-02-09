"""
Training Context Module

Training context manager with gradient scaling and clipping.
"""

from typing import Optional
import torch
from .device import DeviceContext


class TrainingContext:
    """
    Context manager for training operations with automatic
    gradient scaling and error handling.
    """
    
    def __init__(
        self,
        device_context: DeviceContext,
        optimizer: torch.optim.Optimizer,
        max_grad_norm: Optional[float] = None
    ):
        """
        Initialize training context.
        
        Args:
            device_context: Device context manager
            optimizer: Optimizer
            max_grad_norm: Maximum gradient norm for clipping
        """
        self.device_context = device_context
        self.optimizer = optimizer
        self.max_grad_norm = max_grad_norm
        self.scaler = device_context.get_scaler()
    
    def backward(self, loss: torch.Tensor):
        """
        Backward pass with automatic scaling.
        
        Args:
            loss: Loss tensor
        """
        if self.scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
    
    def step(self):
        """Optimizer step with automatic unscaling."""
        if self.scaler:
            # Unscale gradients
            self.scaler.unscale_(self.optimizer)
            
            # Gradient clipping
            if self.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.device_context.model.parameters(),
                    max_norm=self.max_grad_norm
                )
            
            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # Gradient clipping
            if self.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.device_context.model.parameters(),
                    max_norm=self.max_grad_norm
                )
            
            self.optimizer.step()
        
        self.optimizer.zero_grad()
    
    def zero_grad(self):
        """Zero gradients."""
        self.optimizer.zero_grad()



