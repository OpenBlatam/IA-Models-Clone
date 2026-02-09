"""
Training Pipeline

Functional pipeline for model training with proper composition.
"""

import logging
from typing import Optional, Dict, Any, Callable
import torch
import torch.nn as nn

from ..training.enhanced_training import EnhancedTrainingPipeline

logger = logging.getLogger(__name__)


class TrainingPipeline:
    """
    Functional training pipeline.
    
    Composes training steps in a clear, functional way.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: Optional[torch.utils.data.DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        criterion: Optional[nn.Module] = None,
        scheduler: Optional[Any] = None
    ):
        """
        Initialize training pipeline.
        
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer: Optimizer
            criterion: Loss function
            scheduler: Learning rate scheduler
        """
        self.pipeline = EnhancedTrainingPipeline(
            model=model,
            train_dataset=train_loader.dataset,
            val_dataset=val_loader.dataset if val_loader else None,
            batch_size=train_loader.batch_size,
            num_workers=train_loader.num_workers
        )
        
        if optimizer and criterion:
            self.pipeline.setup_training(
                optimizer=optimizer,
                criterion=criterion,
                scheduler=scheduler
            )
    
    def train(self, num_epochs: int, **kwargs) -> Dict[str, Any]:
        """
        Train model.
        
        Args:
            num_epochs: Number of epochs
            **kwargs: Additional training parameters
            
        Returns:
            Training history
        """
        return self.pipeline.train(num_epochs=num_epochs, **kwargs)



