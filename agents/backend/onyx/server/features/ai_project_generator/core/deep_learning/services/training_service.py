"""
Training Service - High-level Training Management
=================================================

Service for managing training workflows:
- Training orchestration
- Configuration management
- Progress tracking
- Result management
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional
import torch
from torch.utils.data import DataLoader

from ..core.base import BaseComponent
from ..training import Trainer, TrainingConfig, EarlyStopping, create_optimizer, create_scheduler
from ..utils import get_device, set_seed, ExperimentTracker
from ..architecture.observer import EventPublisher
from ..architecture.strategy import TrainingStrategy, StandardTrainingStrategy

logger = logging.getLogger(__name__)


class TrainingService(BaseComponent):
    """
    High-level service for training management.
    
    Orchestrates the entire training workflow.
    """
    
    def _initialize(self) -> None:
        """Initialize service."""
        self.device = get_device()
        self.tracker: Optional[ExperimentTracker] = None
        self.event_publisher = EventPublisher()
        self.strategy: Optional[TrainingStrategy] = None
    
    def setup(
        self,
        experiment_name: str,
        log_dir: Optional[Path] = None,
        use_tensorboard: bool = True,
        use_wandb: bool = False
    ) -> 'TrainingService':
        """
        Setup training service.
        
        Args:
            experiment_name: Experiment name
            log_dir: Logging directory
            use_tensorboard: Use TensorBoard
            use_wandb: Use W&B
            
        Returns:
            Self for method chaining
        """
        if log_dir is None:
            log_dir = Path("logs") / experiment_name
        
        self.tracker = ExperimentTracker(
            experiment_name=experiment_name,
            log_dir=log_dir,
            use_tensorboard=use_tensorboard,
            use_wandb=use_wandb
        )
        
        # Subscribe tracker to events
        if self.tracker:
            from ..architecture.observer import TrainingObserver
            observer = TrainingObserver(tracker=self.tracker)
            self.event_publisher.subscribe('epoch_end', observer)
            self.event_publisher.subscribe('training_end', observer)
        
        return self
    
    def set_strategy(self, strategy: TrainingStrategy) -> 'TrainingService':
        """
        Set training strategy.
        
        Args:
            strategy: Training strategy
            
        Returns:
            Self for method chaining
        """
        self.strategy = strategy
        return self
    
    def train(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute training.
        
        Args:
            model: PyTorch model
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            config: Training configuration
            **kwargs: Additional arguments
            
        Returns:
            Training results
        """
        if self.strategy is None:
            self.strategy = StandardTrainingStrategy()
        
        # Publish training start
        self.event_publisher.publish('training_start', {
            'model_params': sum(p.numel() for p in model.parameters())
        })
        
        # Execute strategy
        results = self.strategy.train(
            model,
            train_loader,
            val_loader,
            config=config or {},
            **kwargs
        )
        
        # Publish training end
        self.event_publisher.publish('training_end', results)
        
        # Close tracker
        if self.tracker:
            self.tracker.close()
        
        return results



