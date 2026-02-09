"""
Mixed Precision Training Utilities

Provides utilities for mixed precision training with torch.cuda.amp.
"""

import logging
from typing import Optional, Callable, Any
import torch
from torch.cuda.amp import autocast, GradScaler

logger = logging.getLogger(__name__)


class MixedPrecisionManager:
    """
    Manager for mixed precision training/inference.
    
    Handles:
    - Automatic mixed precision (AMP) scaling
    - Gradient scaling
    - Loss scaling
    """
    
    def __init__(
        self,
        enabled: bool = True,
        init_scale: float = 65536.0,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 2000
    ):
        """
        Initialize mixed precision manager.
        
        Args:
            enabled: Whether mixed precision is enabled
            init_scale: Initial scale for gradient scaler
            growth_factor: Factor to increase scale on successful steps
            backoff_factor: Factor to decrease scale on overflow
            growth_interval: Steps between scale increases
        """
        self.enabled = enabled and torch.cuda.is_available()
        self.scaler = None
        
        if self.enabled:
            self.scaler = GradScaler(
                init_scale=init_scale,
                growth_factor=growth_factor,
                backoff_factor=backoff_factor,
                growth_interval=growth_interval
            )
            logger.info("Mixed precision enabled")
        else:
            logger.info("Mixed precision disabled")
    
    def autocast(self):
        """
        Get autocast context manager.
        
        Returns:
            Autocast context manager (or no-op if disabled)
        """
        if self.enabled:
            return autocast()
        else:
            # Return a no-op context manager
            from contextlib import nullcontext
            return nullcontext()
    
    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """
        Scale loss for mixed precision training.
        
        Args:
            loss: Loss tensor
            
        Returns:
            Scaled loss tensor
        """
        if self.enabled and self.scaler is not None:
            return self.scaler.scale(loss)
        return loss
    
    def step(
        self,
        optimizer: torch.optim.Optimizer,
        loss: Optional[torch.Tensor] = None
    ) -> None:
        """
        Perform optimizer step with gradient scaling.
        
        Args:
            optimizer: Optimizer to step
            loss: Optional loss tensor (if provided, will backward)
        """
        if not self.enabled or self.scaler is None:
            if loss is not None:
                loss.backward()
            optimizer.step()
            return
        
        if loss is not None:
            scaled_loss = self.scaler.scale(loss)
            scaled_loss.backward()
        
        self.scaler.step(optimizer)
        self.scaler.update()
    
    def unscale_gradients(self, optimizer: torch.optim.Optimizer) -> None:
        """
        Unscale gradients before clipping.
        
        Args:
            optimizer: Optimizer with gradients to unscale
        """
        if self.enabled and self.scaler is not None:
            self.scaler.unscale_(optimizer)
    
    def update(self) -> None:
        """Update scaler (call after optimizer step)."""
        if self.enabled and self.scaler is not None:
            self.scaler.update()
    
    def get_scale(self) -> float:
        """
        Get current scale value.
        
        Returns:
            Current scale (1.0 if disabled)
        """
        if self.enabled and self.scaler is not None:
            return self.scaler.get_scale()
        return 1.0


def create_mixed_precision_manager(
    enabled: bool = True,
    **kwargs
) -> MixedPrecisionManager:
    """
    Create mixed precision manager.
    
    Args:
        enabled: Whether to enable mixed precision
        **kwargs: Additional arguments for MixedPrecisionManager
        
    Returns:
        MixedPrecisionManager instance
    """
    return MixedPrecisionManager(enabled=enabled, **kwargs)



