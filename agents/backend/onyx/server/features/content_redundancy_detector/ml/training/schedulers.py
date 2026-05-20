"""
Learning Rate Scheduler Factory
Modular scheduler creation and configuration
"""

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import (
    StepLR, ExponentialLR, CosineAnnealingLR,
    ReduceLROnPlateau, OneCycleLR, CosineAnnealingWarmRestarts
)
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class SchedulerFactory:
    """
    Factory for creating learning rate schedulers
    """
    
    @staticmethod
    def create_scheduler(
        optimizer: optim.Optimizer,
        scheduler_type: str = "step",
        **kwargs
    ) -> optim.lr_scheduler._LRScheduler:
        """
        Create learning rate scheduler
        
        Args:
            optimizer: Optimizer to schedule
            scheduler_type: Type of scheduler
            **kwargs: Additional scheduler parameters
            
        Returns:
            Scheduler instance
        """
        if scheduler_type.lower() == "step":
            return StepLR(
                optimizer,
                step_size=kwargs.get('step_size', 30),
                gamma=kwargs.get('gamma', 0.1),
            )
        elif scheduler_type.lower() == "exponential":
            return ExponentialLR(
                optimizer,
                gamma=kwargs.get('gamma', 0.95),
            )
        elif scheduler_type.lower() == "cosine":
            return CosineAnnealingLR(
                optimizer,
                T_max=kwargs.get('T_max', 50),
                eta_min=kwargs.get('eta_min', 0),
            )
        elif scheduler_type.lower() == "cosine_restarts":
            return CosineAnnealingWarmRestarts(
                optimizer,
                T_0=kwargs.get('T_0', 10),
                T_mult=kwargs.get('T_mult', 2),
                eta_min=kwargs.get('eta_min', 0),
            )
        elif scheduler_type.lower() == "reduce_on_plateau":
            return ReduceLROnPlateau(
                optimizer,
                mode=kwargs.get('mode', 'min'),
                factor=kwargs.get('factor', 0.1),
                patience=kwargs.get('patience', 10),
                verbose=kwargs.get('verbose', True),
            )
        elif scheduler_type.lower() == "onecycle":
            return OneCycleLR(
                optimizer,
                max_lr=kwargs.get('max_lr', 0.01),
                total_steps=kwargs.get('total_steps', 100),
                pct_start=kwargs.get('pct_start', 0.3),
            )
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    @staticmethod
    def create_scheduler_from_config(
        optimizer: optim.Optimizer,
        config: Dict[str, Any]
    ) -> optim.lr_scheduler._LRScheduler:
        """
        Create scheduler from configuration dictionary
        
        Args:
            optimizer: Optimizer to schedule
            config: Scheduler configuration
            
        Returns:
            Scheduler instance
        """
        scheduler_type = config.pop('type', 'step')
        return SchedulerFactory.create_scheduler(
            optimizer,
            scheduler_type,
            **config
        )



