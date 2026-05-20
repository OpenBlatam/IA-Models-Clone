"""
Loss Functions - Ultra-Specific Loss Calculation Components
Each loss function in its own focused implementation
"""

import torch
import torch.nn as nn
import logging
from typing import Dict, Any
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class LossBase(ABC):
    """Base class for all loss functions"""
    
    def __init__(self, name: str = "Loss"):
        self.name = name
    
    @abstractmethod
    def compute(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute loss"""
        pass


class MSELoss(LossBase):
    """Mean Squared Error loss"""
    
    def __init__(self):
        super().__init__("MSELoss")
        self.criterion = nn.MSELoss()
    
    def compute(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute MSE loss"""
        return self.criterion(predictions, targets)


class MAELoss(LossBase):
    """Mean Absolute Error loss"""
    
    def __init__(self):
        super().__init__("MAELoss")
        self.criterion = nn.L1Loss()
    
    def compute(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute MAE loss"""
        return self.criterion(predictions, targets)


class BCELoss(LossBase):
    """Binary Cross Entropy loss"""
    
    def __init__(self):
        super().__init__("BCELoss")
        self.criterion = nn.BCELoss()
    
    def compute(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute BCE loss"""
        return self.criterion(predictions, targets)


class CrossEntropyLoss(LossBase):
    """Cross Entropy loss"""
    
    def __init__(self):
        super().__init__("CrossEntropyLoss")
        self.criterion = nn.CrossEntropyLoss()
    
    def compute(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute Cross Entropy loss"""
        return self.criterion(predictions, targets)


class SmoothL1Loss(LossBase):
    """Smooth L1 loss"""
    
    def __init__(self, beta: float = 1.0):
        super().__init__("SmoothL1Loss")
        self.criterion = nn.SmoothL1Loss(beta=beta)
    
    def compute(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute Smooth L1 loss"""
        return self.criterion(predictions, targets)


class FocalLoss(LossBase):
    """Focal loss for imbalanced datasets"""
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0):
        super().__init__("FocalLoss")
        self.alpha = alpha
        self.gamma = gamma
    
    def compute(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute Focal loss"""
        ce_loss = nn.functional.cross_entropy(predictions, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


# Factory for losses
class LossFactory:
    """Factory for creating loss functions"""
    
    _registry = {
        'mse': MSELoss,
        'mae': MAELoss,
        'bce': BCELoss,
        'ce': CrossEntropyLoss,
        'smooth_l1': SmoothL1Loss,
        'focal': FocalLoss,
    }
    
    @classmethod
    def create(cls, loss_type: str, **kwargs) -> LossBase:
        """Create loss function"""
        loss_type = loss_type.lower()
        if loss_type not in cls._registry:
            raise ValueError(f"Unknown loss type: {loss_type}")
        return cls._registry[loss_type](**kwargs)
    
    @classmethod
    def register(cls, name: str, loss_class: type):
        """Register custom loss function"""
        cls._registry[name.lower()] = loss_class


__all__ = [
    "LossBase",
    "MSELoss",
    "MAELoss",
    "BCELoss",
    "CrossEntropyLoss",
    "SmoothL1Loss",
    "FocalLoss",
    "LossFactory",
]



