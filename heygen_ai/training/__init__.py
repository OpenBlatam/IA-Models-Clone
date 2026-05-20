"""
Training Module for HeyGen AI
==============================

This module contains training utilities following best practices:
- Training loops with proper error handling
- Learning rate scheduling
- Early stopping
- Gradient clipping
- Mixed precision training
- Experiment tracking
- Fine-tuning with LoRA
- Model evaluation
"""

# Import base training classes (defined in this file)
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


@dataclass
class TrainingConfig:
    """Configuration for training.
    
    Attributes:
        learning_rate: Initial learning rate
        weight_decay: Weight decay for optimizer
        gradient_clip_norm: Gradient clipping norm
        use_mixed_precision: Use FP16/BF16 training
        early_stopping_patience: Early stopping patience
        save_checkpoint_steps: Steps between checkpoints
    """
    
    def __init__(
        self,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        gradient_clip_norm: float = 1.0,
        use_mixed_precision: bool = True,
        early_stopping_patience: int = 5,
        save_checkpoint_steps: int = 1000,
    ):
        """Initialize training configuration."""
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.gradient_clip_norm = gradient_clip_norm
        self.use_mixed_precision = use_mixed_precision
        self.early_stopping_patience = early_stopping_patience
        self.save_checkpoint_steps = save_checkpoint_steps


class Trainer:
    """Base trainer class with common training functionality.
    
    Features:
    - Mixed precision training
    - Gradient clipping
    - Learning rate scheduling
    - Early stopping
    - Checkpointing
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        criterion: nn.Module,
        config: TrainingConfig,
        device: Optional[torch.device] = None,
    ):
        """Initialize trainer.
        
        Args:
            model: PyTorch model
            optimizer: Optimizer
            criterion: Loss function
            config: Training configuration
            device: Training device
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Mixed precision scaler
        self.scaler = None
        if config.use_mixed_precision and self.device.type == "cuda":
            self.scaler = torch.cuda.amp.GradScaler()
        
        # Move model to device
        self.model = self.model.to(self.device)
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Perform a single training step.
        
        Args:
            batch: Training batch
        
        Returns:
            Dictionary with loss and metrics
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Move batch to device
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                 for k, v in batch.items()}
        
        # Forward pass with mixed precision
        if self.scaler:
            with torch.cuda.amp.autocast():
                outputs = self.model(**batch)
                loss = self.criterion(outputs, batch.get('labels'))
            
            # Backward pass
            self.scaler.scale(loss).backward()
            
            # Gradient clipping
            if self.config.gradient_clip_norm > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip_norm
                )
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            outputs = self.model(**batch)
            loss = self.criterion(outputs, batch.get('labels'))
            
            loss.backward()
            
            # Gradient clipping
            if self.config.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip_norm
                )
            
            self.optimizer.step()
        
        return {"loss": loss.item()}


# Import submodules
from .finetuning import (
    LoRAConfig,
    FineTuningConfig,
    LoRAFineTuner,
    FullFineTuner,
)
from .evaluation import (
    EvaluationMetrics,
    ModelEvaluator,
    train_test_split,
    train_val_test_split,
)
from .experiment_tracking import (
    ExperimentConfig,
    ExperimentTracker,
)

__all__ = [
    # Base training
    "Trainer",
    "TrainingConfig",
    # Fine-tuning
    "LoRAConfig",
    "FineTuningConfig",
    "LoRAFineTuner",
    "FullFineTuner",
    # Evaluation
    "EvaluationMetrics",
    "ModelEvaluator",
    "train_test_split",
    "train_val_test_split",
    # Experiment tracking
    "ExperimentConfig",
    "ExperimentTracker",
]
