"""
Schedulers avanzados de learning rate
"""

import torch
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    ReduceLROnPlateau,
    OneCycleLR,
    LambdaLR
)
from typing import Optional, Callable
import logging

logger = logging.getLogger(__name__)


class SchedulerFactory:
    """Factory para crear schedulers avanzados"""
    
    @staticmethod
    def create_scheduler(
        scheduler_type: str,
        optimizer: torch.optim.Optimizer,
        **kwargs
    ):
        """
        Crea scheduler según tipo
        
        Args:
            scheduler_type: Tipo de scheduler
            optimizer: Optimizador
            **kwargs: Parámetros del scheduler
        """
        if scheduler_type == "cosine":
            return CosineAnnealingLR(
                optimizer,
                T_max=kwargs.get("T_max", 1000),
                eta_min=kwargs.get("eta_min", 1e-7)
            )
        
        elif scheduler_type == "cosine_restarts":
            return CosineAnnealingWarmRestarts(
                optimizer,
                T_0=kwargs.get("T_0", 10),
                T_mult=kwargs.get("T_mult", 2),
                eta_min=kwargs.get("eta_min", 1e-7)
            )
        
        elif scheduler_type == "reduce_on_plateau":
            return ReduceLROnPlateau(
                optimizer,
                mode=kwargs.get("mode", "min"),
                factor=kwargs.get("factor", 0.5),
                patience=kwargs.get("patience", 10),
                verbose=True
            )
        
        elif scheduler_type == "one_cycle":
            return OneCycleLR(
                optimizer,
                max_lr=kwargs.get("max_lr", 5e-5),
                epochs=kwargs.get("epochs", 10),
                steps_per_epoch=kwargs.get("steps_per_epoch", 100)
            )
        
        elif scheduler_type == "warmup_cosine":
            # Warmup + Cosine
            def lr_lambda(step):
                warmup_steps = kwargs.get("warmup_steps", 100)
                total_steps = kwargs.get("total_steps", 1000)
                
                if step < warmup_steps:
                    return step / warmup_steps
                else:
                    progress = (step - warmup_steps) / (total_steps - warmup_steps)
                    return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))
            
            return LambdaLR(optimizer, lr_lambda)
        
        else:
            raise ValueError(f"Scheduler type {scheduler_type} no soportado")




