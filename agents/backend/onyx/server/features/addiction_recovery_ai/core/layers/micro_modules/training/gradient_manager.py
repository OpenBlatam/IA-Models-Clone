"""
Gradient Manager - Ultra-Specific Gradient Management
Separated into its own file for maximum modularity
"""

import torch
import torch.nn as nn
import logging
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class GradientManagerBase(ABC):
    """Base class for gradient management"""
    
    def __init__(self, name: str = "GradientManager"):
        self.name = name
    
    @abstractmethod
    def manage(self, model: nn.Module, loss: torch.Tensor) -> Dict[str, Any]:
        """Manage gradients"""
        pass


class GradientClipper(GradientManagerBase):
    """Clip gradients to prevent exploding gradients"""
    
    def __init__(self, max_norm: float = 1.0):
        super().__init__("GradientClipper")
        self.max_norm = max_norm
    
    def manage(self, model: nn.Module, loss: torch.Tensor) -> Dict[str, Any]:
        """Clip gradients"""
        # Compute gradients first
        loss.backward()
        
        # Clip gradients
        total_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=self.max_norm
        )
        
        return {
            'total_norm': total_norm.item(),
            'clipped': total_norm.item() > self.max_norm
        }


class GradientChecker(GradientManagerBase):
    """Check gradients for NaN/Inf values"""
    
    def __init__(self):
        super().__init__("GradientChecker")
    
    def manage(self, model: nn.Module, loss: torch.Tensor) -> Dict[str, Any]:
        """Check gradients"""
        loss.backward()
        
        has_nan = False
        has_inf = False
        param_count = 0
        nan_count = 0
        inf_count = 0
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                param_count += 1
                if torch.isnan(param.grad).any():
                    has_nan = True
                    nan_count += 1
                    logger.warning(f"NaN gradient in {name}")
                if torch.isinf(param.grad).any():
                    has_inf = True
                    inf_count += 1
                    logger.warning(f"Inf gradient in {name}")
        
        return {
            'has_nan': has_nan,
            'has_inf': has_inf,
            'param_count': param_count,
            'nan_count': nan_count,
            'inf_count': inf_count
        }


class GradientAccumulator(GradientManagerBase):
    """Accumulate gradients over multiple steps"""
    
    def __init__(self, accumulation_steps: int = 1):
        super().__init__("GradientAccumulator")
        self.accumulation_steps = accumulation_steps
        self.current_step = 0
    
    def manage(self, model: nn.Module, loss: torch.Tensor) -> Dict[str, Any]:
        """Accumulate gradients"""
        # Scale loss by accumulation steps
        scaled_loss = loss / self.accumulation_steps
        scaled_loss.backward()
        
        self.current_step += 1
        
        should_update = self.current_step >= self.accumulation_steps
        
        if should_update:
            self.current_step = 0
        
        return {
            'scaled_loss': scaled_loss.item(),
            'current_step': self.current_step,
            'should_update': should_update
        }
    
    def reset(self):
        """Reset accumulation counter"""
        self.current_step = 0


class GradientNormalizer(GradientManagerBase):
    """Normalize gradients"""
    
    def __init__(self, norm_type: float = 2.0):
        super().__init__("GradientNormalizer")
        self.norm_type = norm_type
    
    def manage(self, model: nn.Module, loss: torch.Tensor) -> Dict[str, Any]:
        """Normalize gradients"""
        loss.backward()
        
        total_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=float('inf'),
            norm_type=self.norm_type
        )
        
        # Normalize by total norm
        if total_norm > 0:
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.data.div_(total_norm)
        
        return {
            'total_norm': total_norm.item(),
            'normalized': True
        }


# Factory for gradient managers
class GradientManagerFactory:
    """Factory for creating gradient managers"""
    
    _registry = {
        'clip': GradientClipper,
        'check': GradientChecker,
        'accumulate': GradientAccumulator,
        'normalize': GradientNormalizer,
    }
    
    @classmethod
    def create(cls, manager_type: str, **kwargs) -> GradientManagerBase:
        """Create gradient manager"""
        manager_type = manager_type.lower()
        if manager_type not in cls._registry:
            raise ValueError(f"Unknown gradient manager type: {manager_type}")
        return cls._registry[manager_type](**kwargs)
    
    @classmethod
    def register(cls, name: str, manager_class: type):
        """Register custom gradient manager"""
        cls._registry[name.lower()] = manager_class


__all__ = [
    "GradientManagerBase",
    "GradientClipper",
    "GradientChecker",
    "GradientAccumulator",
    "GradientNormalizer",
    "GradientManagerFactory",
]



