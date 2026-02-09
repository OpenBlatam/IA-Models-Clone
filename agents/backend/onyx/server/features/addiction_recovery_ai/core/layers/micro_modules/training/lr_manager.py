"""
Learning Rate Manager - Ultra-Specific LR Management
Separated into its own file for maximum modularity
"""

import torch
import torch.optim as optim
import logging
from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class LRManagerBase(ABC):
    """Base class for learning rate management"""
    
    def __init__(self, name: str = "LRManager"):
        self.name = name
    
    @abstractmethod
    def get_scheduler(self, optimizer: optim.Optimizer, **kwargs) -> Any:
        """Get learning rate scheduler"""
        pass
    
    @abstractmethod
    def get_lr(self, optimizer: optim.Optimizer) -> float:
        """Get current learning rate"""
        pass


class StepLRManager(LRManagerBase):
    """Step learning rate scheduler"""
    
    def __init__(self):
        super().__init__("StepLRManager")
    
    def get_scheduler(self, optimizer: optim.Optimizer, step_size: int = 30, gamma: float = 0.1, **kwargs) -> optim.lr_scheduler.StepLR:
        """Get StepLR scheduler"""
        return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    def get_lr(self, optimizer: optim.Optimizer) -> float:
        """Get current learning rate"""
        return optimizer.param_groups[0]['lr']


class ExponentialLRManager(LRManagerBase):
    """Exponential learning rate scheduler"""
    
    def __init__(self):
        super().__init__("ExponentialLRManager")
    
    def get_scheduler(self, optimizer: optim.Optimizer, gamma: float = 0.95, **kwargs) -> optim.lr_scheduler.ExponentialLR:
        """Get ExponentialLR scheduler"""
        return optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    
    def get_lr(self, optimizer: optim.Optimizer) -> float:
        """Get current learning rate"""
        return optimizer.param_groups[0]['lr']


class CosineAnnealingLRManager(LRManagerBase):
    """Cosine annealing learning rate scheduler"""
    
    def __init__(self):
        super().__init__("CosineAnnealingLRManager")
    
    def get_scheduler(self, optimizer: optim.Optimizer, T_max: int = 10, eta_min: float = 0, **kwargs) -> optim.lr_scheduler.CosineAnnealingLR:
        """Get CosineAnnealingLR scheduler"""
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
    
    def get_lr(self, optimizer: optim.Optimizer) -> float:
        """Get current learning rate"""
        return optimizer.param_groups[0]['lr']


class ReduceLROnPlateauManager(LRManagerBase):
    """Reduce LR on plateau scheduler"""
    
    def __init__(self):
        super().__init__("ReduceLROnPlateauManager")
    
    def get_scheduler(self, optimizer: optim.Optimizer, mode: str = 'min', factor: float = 0.1, patience: int = 10, **kwargs) -> optim.lr_scheduler.ReduceLROnPlateau:
        """Get ReduceLROnPlateau scheduler"""
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=mode,
            factor=factor,
            patience=patience
        )
    
    def get_lr(self, optimizer: optim.Optimizer) -> float:
        """Get current learning rate"""
        return optimizer.param_groups[0]['lr']


class OneCycleLRManager(LRManagerBase):
    """One cycle learning rate scheduler"""
    
    def __init__(self):
        super().__init__("OneCycleLRManager")
    
    def get_scheduler(self, optimizer: optim.Optimizer, max_lr: float = 0.01, total_steps: int = 100, **kwargs) -> optim.lr_scheduler.OneCycleLR:
        """Get OneCycleLR scheduler"""
        return optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            total_steps=total_steps
        )
    
    def get_lr(self, optimizer: optim.Optimizer) -> float:
        """Get current learning rate"""
        return optimizer.param_groups[0]['lr']


class WarmupLRManager(LRManagerBase):
    """Warmup learning rate scheduler"""
    
    def __init__(self):
        super().__init__("WarmupLRManager")
    
    def get_scheduler(self, optimizer: optim.Optimizer, warmup_steps: int = 1000, **kwargs) -> Any:
        """Get warmup scheduler (custom implementation)"""
        # This would typically wrap another scheduler
        # For now, return a simple lambda scheduler
        from torch.optim.lr_scheduler import LambdaLR
        
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            return 1.0
        
        return LambdaLR(optimizer, lr_lambda)
    
    def get_lr(self, optimizer: optim.Optimizer) -> float:
        """Get current learning rate"""
        return optimizer.param_groups[0]['lr']


# Factory for LR managers
class LRManagerFactory:
    """Factory for creating learning rate managers"""
    
    _registry = {
        'step': StepLRManager,
        'exponential': ExponentialLRManager,
        'cosine': CosineAnnealingLRManager,
        'plateau': ReduceLROnPlateauManager,
        'onecycle': OneCycleLRManager,
        'warmup': WarmupLRManager,
    }
    
    @classmethod
    def create(cls, manager_type: str, **kwargs) -> LRManagerBase:
        """Create LR manager"""
        manager_type = manager_type.lower()
        if manager_type not in cls._registry:
            raise ValueError(f"Unknown LR manager type: {manager_type}")
        return cls._registry[manager_type](**kwargs)
    
    @classmethod
    def register(cls, name: str, manager_class: type):
        """Register custom LR manager"""
        cls._registry[name.lower()] = manager_class


__all__ = [
    "LRManagerBase",
    "StepLRManager",
    "ExponentialLRManager",
    "CosineAnnealingLRManager",
    "ReduceLROnPlateauManager",
    "OneCycleLRManager",
    "WarmupLRManager",
    "LRManagerFactory",
]



