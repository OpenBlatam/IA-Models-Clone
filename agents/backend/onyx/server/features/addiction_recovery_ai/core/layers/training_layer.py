"""
Training Layer - Ultra Modular Training Components
Separates training logic, optimizers, schedulers, and configuration
"""

from typing import Optional, Dict, Any, List
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
from abc import ABC, abstractmethod

from .interfaces import ITrainer, IOptimizer, IScheduler

logger = logging.getLogger(__name__)


# ============================================================================
# Training Configuration
# ============================================================================

class TrainingConfig:
    """Training configuration container"""
    
    def __init__(
        self,
        num_epochs: int = 10,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        gradient_clip_val: float = 1.0,
        accumulate_grad_batches: int = 1,
        use_mixed_precision: bool = True,
        **kwargs
    ):
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.gradient_clip_val = gradient_clip_val
        self.accumulate_grad_batches = accumulate_grad_batches
        self.use_mixed_precision = use_mixed_precision
        self.extra_config = kwargs
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'num_epochs': self.num_epochs,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'gradient_clip_val': self.gradient_clip_val,
            'accumulate_grad_batches': self.accumulate_grad_batches,
            'use_mixed_precision': self.use_mixed_precision,
            **self.extra_config
        }


# ============================================================================
# Optimizer Factory
# ============================================================================

class OptimizerFactory:
    """Factory for creating optimizers"""
    
    _registry: Dict[str, type] = {
        'adam': optim.Adam,
        'adamw': optim.AdamW,
        'sgd': optim.SGD,
        'rmsprop': optim.RMSprop,
    }
    
    @classmethod
    def create(
        cls,
        optimizer_type: str,
        model: nn.Module,
        learning_rate: float = 1e-3,
        **kwargs
    ) -> optim.Optimizer:
        """Create optimizer"""
        optimizer_type = optimizer_type.lower()
        
        if optimizer_type not in cls._registry:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
        
        optimizer_class = cls._registry[optimizer_type]
        return optimizer_class(model.parameters(), lr=learning_rate, **kwargs)
    
    @classmethod
    def register(cls, name: str, optimizer_class: type):
        """Register custom optimizer"""
        cls._registry[name.lower()] = optimizer_class


# ============================================================================
# Scheduler Factory
# ============================================================================

class SchedulerFactory:
    """Factory for creating learning rate schedulers"""
    
    @staticmethod
    def create(
        scheduler_type: str,
        optimizer: optim.Optimizer,
        **kwargs
    ) -> optim.lr_scheduler._LRScheduler:
        """Create scheduler"""
        scheduler_type = scheduler_type.lower()
        
        if scheduler_type == 'reduce_on_plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(optimizer, **kwargs)
        elif scheduler_type == 'step':
            return optim.lr_scheduler.StepLR(optimizer, **kwargs)
        elif scheduler_type == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(optimizer, **kwargs)
        elif scheduler_type == 'exponential':
            return optim.lr_scheduler.ExponentialLR(optimizer, **kwargs)
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")


# ============================================================================
# Training Pipeline - Composable training components
# ============================================================================

class TrainingPipeline:
    """
    Composable training pipeline
    Separates training logic into modular components
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Setup optimizer and scheduler
        self.optimizer = OptimizerFactory.create(
            'adamw',
            model,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        self.scheduler = None
        self.criterion = None
        self.callbacks: List[Any] = []
    
    def set_criterion(self, criterion: nn.Module):
        """Set loss criterion"""
        self.criterion = criterion
        return self
    
    def set_scheduler(self, scheduler_type: str, **kwargs):
        """Set learning rate scheduler"""
        self.scheduler = SchedulerFactory.create(scheduler_type, self.optimizer, **kwargs)
        return self
    
    def add_callback(self, callback: Any):
        """Add training callback"""
        self.callbacks.append(callback)
        return self
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Training step logic here
            # This is a simplified version - actual implementation would use BaseTrainer
            pass
        
        return {'loss': total_loss / num_batches if num_batches > 0 else 0.0}
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate model"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                # Validation logic here
                pass
        
        return {'loss': total_loss / num_batches if num_batches > 0 else 0.0}


# ============================================================================
# Trainer Factory
# ============================================================================

class TrainerFactory:
    """Factory for creating trainers"""
    
    @staticmethod
    def create(
        model: nn.Module,
        config: TrainingConfig,
        trainer_type: str = "base"
    ) -> TrainingPipeline:
        """Create trainer from config"""
        if trainer_type == "base":
            return TrainingPipeline(model, config)
        else:
            raise ValueError(f"Unknown trainer type: {trainer_type}")


# Export main components
__all__ = [
    "TrainingConfig",
    "OptimizerFactory",
    "SchedulerFactory",
    "TrainingPipeline",
    "TrainerFactory",
]



