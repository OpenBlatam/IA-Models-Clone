"""
Advanced Schedulers - Specialized Learning Rate Schedulers
==========================================================

Advanced learning rate scheduling strategies:
- Warmup schedulers
- Cosine annealing with restarts
- OneCycleLR
- Custom schedulers
"""

import logging
from typing import Optional, Dict, Any
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import math

logger = logging.getLogger(__name__)


class WarmupScheduler:
    """
    Learning rate scheduler with warmup.
    
    Combines warmup phase with another scheduler.
    """
    
    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_steps: int,
        after_warmup_scheduler: Any,
        warmup_lr: Optional[float] = None
    ):
        """
        Initialize warmup scheduler.
        
        Args:
            optimizer: Optimizer
            warmup_steps: Number of warmup steps
            after_warmup_scheduler: Scheduler to use after warmup
            warmup_lr: Warmup learning rate (uses initial LR if None)
        """
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.after_warmup_scheduler = after_warmup_scheduler
        self.warmup_lr = warmup_lr
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.current_step = 0
    
    def step(self) -> None:
        """Step scheduler."""
        if self.current_step < self.warmup_steps:
            # Warmup phase
            if self.warmup_lr is None:
                warmup_lr = self.base_lrs[0]
            else:
                warmup_lr = self.warmup_lr
            
            lr = warmup_lr * (self.current_step + 1) / self.warmup_steps
            
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        else:
            # After warmup
            self.after_warmup_scheduler.step()
        
        self.current_step += 1
    
    def get_last_lr(self) -> List[float]:
        """Get last learning rates."""
        return [group['lr'] for group in self.optimizer.param_groups]
    
    def state_dict(self) -> Dict[str, Any]:
        """Get state dict."""
        return {
            'current_step': self.current_step,
            'after_warmup_scheduler': self.after_warmup_scheduler.state_dict()
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load state dict."""
        self.current_step = state_dict['current_step']
        self.after_warmup_scheduler.load_state_dict(state_dict['after_warmup_scheduler'])


class CosineAnnealingWarmRestarts(lr_scheduler.CosineAnnealingWarmRestarts):
    """Wrapper for cosine annealing with warm restarts."""
    pass


def create_warmup_scheduler(
    optimizer: optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    scheduler_type: str = 'cosine',
    warmup_lr: Optional[float] = None,
    **kwargs
) -> WarmupScheduler:
    """
    Create scheduler with warmup.
    
    Args:
        optimizer: Optimizer
        warmup_steps: Number of warmup steps
        total_steps: Total number of steps
        scheduler_type: Type of scheduler after warmup
        warmup_lr: Warmup learning rate
        **kwargs: Additional scheduler arguments
        
    Returns:
        Warmup scheduler
    """
    # Create base scheduler
    if scheduler_type == 'cosine':
        base_scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_steps - warmup_steps,
            **kwargs
        )
    elif scheduler_type == 'linear':
        base_scheduler = lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=kwargs.get('end_factor', 0.0),
            total_iters=total_steps - warmup_steps
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    return WarmupScheduler(optimizer, warmup_steps, base_scheduler, warmup_lr)


def create_onecycle_scheduler(
    optimizer: optim.Optimizer,
    max_lr: float,
    total_steps: int,
    pct_start: float = 0.3,
    div_factor: float = 25.0,
    final_div_factor: float = 10000.0
) -> lr_scheduler.OneCycleLR:
    """
    Create OneCycleLR scheduler.
    
    Args:
        optimizer: Optimizer
        max_lr: Maximum learning rate
        total_steps: Total number of steps
        pct_start: Percentage of steps for warmup
        div_factor: Initial LR = max_lr / div_factor
        final_div_factor: Final LR = initial_lr / final_div_factor
        
    Returns:
        OneCycleLR scheduler
    """
    return lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,
        total_steps=total_steps,
        pct_start=pct_start,
        div_factor=div_factor,
        final_div_factor=final_div_factor
    )



