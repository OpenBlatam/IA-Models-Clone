"""
Training Components - Ultra-Granular Training Management
Re-exports from specialized modules for backward compatibility
"""

from typing import Optional, Dict, Any, List
import torch
import torch.nn as nn
import torch.optim as optim
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


# ============================================================================
# Loss Calculation - Import from specialized module
# ============================================================================

from .losses import (
    LossBase,
    MSELoss,
    MAELoss,
    BCELoss,
    CrossEntropyLoss,
    SmoothL1Loss,
    FocalLoss,
    LossFactory
)

# Backward compatibility wrapper
class LossCalculator:
    """Calculate and manage loss functions (backward compatibility)"""
    
    @staticmethod
    def create(loss_type: str, **kwargs) -> nn.Module:
        """Create loss function"""
        loss = LossFactory.create(loss_type, **kwargs)
        return loss.criterion if hasattr(loss, 'criterion') else loss
    
    @staticmethod
    def calculate(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        criterion: Any
    ) -> torch.Tensor:
        """Calculate loss"""
        if isinstance(criterion, LossBase):
            return criterion.compute(predictions, targets)
        return criterion(predictions, targets)
    
    @staticmethod
    def check_loss(loss: torch.Tensor) -> Dict[str, bool]:
        """Check loss for issues"""
        return {
            'has_nan': torch.isnan(loss).any().item(),
            'has_inf': torch.isinf(loss).any().item(),
            'is_valid': not (torch.isnan(loss).any() or torch.isinf(loss).any())
        }


# ============================================================================
# Gradient Management - Import from specialized module
# ============================================================================

from .training.gradient_manager import (
    GradientManagerBase,
    GradientClipper,
    GradientChecker,
    GradientAccumulator,
    GradientNormalizer,
    GradientManagerFactory
)

# Backward compatibility wrapper
class GradientManager:
    """Manage gradients during training (backward compatibility)"""
    
    @staticmethod
    def clip_gradients(
        model: nn.Module,
        max_norm: float = 1.0,
        norm_type: float = 2.0
    ) -> float:
        """Clip gradients to prevent exploding"""
        # For full functionality, use GradientClipper directly
        total_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=max_norm,
            norm_type=norm_type
        )
        return total_norm.item()
    
    @staticmethod
    def zero_gradients(optimizer: optim.Optimizer):
        """Zero gradients"""
        optimizer.zero_grad()
    
    @staticmethod
    def check_gradients(model: nn.Module) -> Dict[str, Any]:
        """Check gradients for issues"""
        # For full functionality, use GradientChecker directly
        stats = {
            'has_nan': False,
            'has_inf': False,
            'max_grad': 0.0,
            'total_norm': 0.0
        }
        
        total_norm = 0.0
        for name, param in model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    stats['has_nan'] = True
                    logger.warning(f"NaN gradient in {name}")
                if torch.isinf(param.grad).any():
                    stats['has_inf'] = True
                    logger.warning(f"Inf gradient in {name}")
                
                param_norm = param.grad.data.norm(2.0)
                total_norm += param_norm.item() ** 2.0
                stats['max_grad'] = max(stats['max_grad'], param.grad.data.abs().max().item())
        
        stats['total_norm'] = total_norm ** (1.0 / 2.0)
        return stats


# ============================================================================
# Learning Rate Management - Import from specialized module
# ============================================================================

from .training.lr_manager import (
    LRManagerBase,
    StepLRManager,
    ExponentialLRManager,
    CosineAnnealingLRManager,
    ReduceLROnPlateauManager,
    OneCycleLRManager,
    WarmupLRManager,
    LRManagerFactory
)

# Backward compatibility wrapper
class LearningRateManager:
    """Manage learning rate scheduling (backward compatibility)"""
    
    @staticmethod
    def create_scheduler(
        optimizer: optim.Optimizer,
        scheduler_type: str = 'step',
        **kwargs
    ) -> Any:
        """Create learning rate scheduler"""
        manager = LRManagerFactory.create(scheduler_type)
        return manager.get_scheduler(optimizer, **kwargs)
    
    @staticmethod
    def get_lr(optimizer: optim.Optimizer) -> float:
        """Get current learning rate"""
        return optimizer.param_groups[0]['lr']
    
    @staticmethod
    def set_lr(optimizer: optim.Optimizer, lr: float):
        """Set learning rate"""
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


# ============================================================================
# Checkpoint Management - Import from specialized module
# ============================================================================

from .training.checkpoint_manager import (
    CheckpointManagerBase,
    FullCheckpointManager,
    StateDictCheckpointManager,
    BestModelCheckpointManager,
    PeriodicCheckpointManager,
    CheckpointManagerFactory
)

# Backward compatibility wrapper
class CheckpointManager:
    """Manage model checkpoints (backward compatibility)"""
    
    @staticmethod
    def save_checkpoint(
        model: nn.Module,
        path: str,
        optimizer: Optional[optim.Optimizer] = None,
        epoch: Optional[int] = None,
        loss: Optional[float] = None,
        **kwargs
    ):
        """Save model checkpoint"""
        manager = FullCheckpointManager()
        manager.save(model, path, optimizer=optimizer, epoch=epoch, loss=loss, **kwargs)
    
    @staticmethod
    def load_checkpoint(
        model: nn.Module,
        path: str,
        optimizer: Optional[optim.Optimizer] = None,
        map_location: str = 'cpu'
    ) -> Dict[str, Any]:
        """Load model checkpoint"""
        manager = FullCheckpointManager()
        return manager.load(model, path, optimizer=optimizer, map_location=map_location)


# Export all components
__all__ = [
    # Loss Calculation
    "LossCalculator",
    # Gradient Management
    "GradientManager",
    # Learning Rate Management
    "LearningRateManager",
    # Checkpoint Management
    "CheckpointManager",
]
