"""
Scheduler Factory Module

Factory for creating learning rate schedulers.
"""

from typing import Optional, Dict, Any
import logging
import torch.optim as optim

from .cosine import create_cosine_scheduler
from .linear import create_linear_scheduler
from .plateau import create_plateau_scheduler
from .step import create_step_scheduler

logger = logging.getLogger(__name__)


class SchedulerFactory:
    """Factory for creating learning rate schedulers."""
    
    @staticmethod
    def create(
        scheduler_type: str,
        optimizer: optim.Optimizer,
        **kwargs
    ) -> Optional[optim.lr_scheduler._LRScheduler]:
        """
        Create scheduler based on type.
        
        Args:
            scheduler_type: Type of scheduler.
            optimizer: Optimizer instance.
            **kwargs: Scheduler-specific arguments.
        
        Returns:
            Scheduler instance or None.
        """
        scheduler_type = scheduler_type.lower()
        
        if scheduler_type == "cosine":
            return create_cosine_scheduler(optimizer, **kwargs)
        elif scheduler_type == "linear":
            return create_linear_scheduler(optimizer, **kwargs)
        elif scheduler_type == "plateau":
            return create_plateau_scheduler(optimizer, **kwargs)
        elif scheduler_type == "step":
            return create_step_scheduler(optimizer, **kwargs)
        elif scheduler_type == "onecycle":
            return optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=kwargs.get("max_lr", 1e-3),
                total_steps=kwargs.get("total_steps", 100),
                pct_start=kwargs.get("pct_start", 0.3),
                anneal_strategy=kwargs.get("anneal_strategy", "cos")
            )
        elif scheduler_type == "cosine_warm_restart":
            return optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=kwargs.get("T_0", 10),
                T_mult=kwargs.get("T_mult", 2),
                eta_min=kwargs.get("eta_min", 1e-6)
            )
        else:
            logger.warning(f"Unknown scheduler type: {scheduler_type}, returning None")
            return None


def create_scheduler(
    scheduler_type: str,
    optimizer: optim.Optimizer,
    **kwargs
) -> Optional[optim.lr_scheduler._LRScheduler]:
    """Convenience function for creating schedulers."""
    return SchedulerFactory.create(scheduler_type, optimizer, **kwargs)



