"""
Modular Learning Rate Scheduler Factory
Creates schedulers based on configuration
"""

from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available")


class WarmupScheduler:
    """Wrapper for warmup learning rate scheduling"""
    
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
    
    def step(self):
        """Update learning rate with warmup"""
        if self.current_step < self.warmup_steps:
            lr = self.base_lr * (self.current_step + 1) / self.warmup_steps
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        self.current_step += 1


class SchedulerFactory:
    """Factory for creating learning rate schedulers"""
    
    @staticmethod
    def create(
        scheduler_type: str,
        optimizer: optim.Optimizer,
        **kwargs
    ) -> Optional[optim.lr_scheduler._LRScheduler]:
        """
        Create scheduler based on type
        
        Args:
            scheduler_type: Type of scheduler
            optimizer: Optimizer instance
            **kwargs: Scheduler-specific arguments
        
        Returns:
            Scheduler instance or None
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for schedulers")
        
        scheduler_type = scheduler_type.lower()
        
        if scheduler_type == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=kwargs.get("T_max", 100),
                eta_min=kwargs.get("eta_min", 1e-6)
            )
        
        elif scheduler_type == "linear":
            return optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=kwargs.get("start_factor", 1.0),
                end_factor=kwargs.get("end_factor", 0.1),
                total_iters=kwargs.get("total_iters", 100)
            )
        
        elif scheduler_type == "plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=kwargs.get("mode", "min"),
                factor=kwargs.get("factor", 0.5),
                patience=kwargs.get("patience", 5),
                verbose=kwargs.get("verbose", True)
            )
        
        elif scheduler_type == "step":
            return optim.lr_scheduler.StepLR(
                optimizer,
                step_size=kwargs.get("step_size", 30),
                gamma=kwargs.get("gamma", 0.1)
            )
        
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
    """Convenience function for creating schedulers"""
    return SchedulerFactory.create(scheduler_type, optimizer, **kwargs)



