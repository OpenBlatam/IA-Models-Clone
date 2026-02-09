"""
Warmup Scheduler Module

Implements warmup learning rate scheduling.
"""

import logging
import torch.optim as optim

logger = logging.getLogger(__name__)


class WarmupScheduler:
    """
    Wrapper for warmup learning rate scheduling.
    
    Args:
        optimizer: Optimizer instance.
        warmup_steps: Number of warmup steps.
        base_lr: Base learning rate.
    """
    
    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_steps: int,
        base_lr: float
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.base_lr = base_lr
        self.current_step = 0
        logger.debug(f"Initialized WarmupScheduler with warmup_steps={warmup_steps}, base_lr={base_lr}")
    
    def step(self):
        """Update learning rate with warmup."""
        if self.current_step < self.warmup_steps:
            lr = self.base_lr * (self.current_step + 1) / self.warmup_steps
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        self.current_step += 1



