"""
Base Callback
Abstract base class for training callbacks
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class BaseCallback(ABC):
    """
    Abstract base class for training callbacks
    """
    
    def __init__(self):
        """Initialize callback"""
        pass
    
    @abstractmethod
    def on_train_begin(self, trainer: Any, **kwargs):
        """Called at the start of training"""
        pass
    
    @abstractmethod
    def on_train_end(self, trainer: Any, **kwargs):
        """Called at the end of training"""
        pass
    
    def on_epoch_begin(self, trainer: Any, epoch: int, **kwargs):
        """Called at the start of each epoch"""
        pass
    
    def on_epoch_end(self, trainer: Any, epoch: int, metrics: Dict[str, float], **kwargs):
        """Called at the end of each epoch"""
        pass
    
    def on_batch_begin(self, trainer: Any, batch: Any, **kwargs):
        """Called at the start of each batch"""
        pass
    
    def on_batch_end(self, trainer: Any, batch: Any, loss: float, **kwargs):
        """Called at the end of each batch"""
        pass


class EarlyStoppingCallback(BaseCallback):
    """
    Early stopping callback
    """
    
    def __init__(
        self,
        monitor: str = "loss",
        patience: int = 10,
        min_delta: float = 0.001,
        mode: str = "min"
    ):
        """
        Initialize early stopping
        
        Args:
            monitor: Metric to monitor
            patience: Patience
            min_delta: Minimum change
            mode: 'min' or 'max'
        """
        super().__init__()
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.patience_counter = 0
        self.should_stop = False
    
    def on_epoch_end(self, trainer: Any, epoch: int, metrics: Dict[str, float], **kwargs):
        """Check for early stopping"""
        current_value = metrics.get(self.monitor, float('inf'))
        
        if self.mode == 'min':
            improved = current_value < (self.best_value - self.min_delta)
        else:
            improved = current_value > (self.best_value + self.min_delta)
        
        if improved:
            self.best_value = current_value
            self.patience_counter = 0
        else:
            self.patience_counter += 1
        
        if self.patience_counter >= self.patience:
            self.should_stop = True
            logger.info(f"Early stopping triggered at epoch {epoch}")
    
    def on_train_begin(self, trainer: Any, **kwargs):
        """Reset on train begin"""
        self.should_stop = False
        self.patience_counter = 0
    
    def on_train_end(self, trainer: Any, **kwargs):
        """Called at train end"""
        pass


class LearningRateSchedulerCallback(BaseCallback):
    """
    Learning rate scheduler callback
    """
    
    def __init__(self, scheduler: Any):
        """
        Initialize scheduler callback
        
        Args:
            scheduler: Learning rate scheduler
        """
        super().__init__()
        self.scheduler = scheduler
    
    def on_epoch_end(self, trainer: Any, epoch: int, metrics: Dict[str, float], **kwargs):
        """Update learning rate"""
        if hasattr(self.scheduler, 'step'):
            # For ReduceLROnPlateau
            if hasattr(self.scheduler, 'mode'):
                monitor_value = metrics.get('loss', 0)
                self.scheduler.step(monitor_value)
            else:
                self.scheduler.step()
    
    def on_train_begin(self, trainer: Any, **kwargs):
        """Called at train begin"""
        pass
    
    def on_train_end(self, trainer: Any, **kwargs):
        """Called at train end"""
        pass


class CheckpointCallback(BaseCallback):
    """
    Checkpoint callback
    """
    
    def __init__(self, checkpoint_manager: Any, save_every: int = 1):
        """
        Initialize checkpoint callback
        
        Args:
            checkpoint_manager: Checkpoint manager
            save_every: Save every N epochs
        """
        super().__init__()
        self.checkpoint_manager = checkpoint_manager
        self.save_every = save_every
    
    def on_epoch_end(
        self,
        trainer: Any,
        epoch: int,
        metrics: Dict[str, float],
        **kwargs
    ):
        """Save checkpoint"""
        if epoch % self.save_every == 0:
            is_best = metrics.get('loss', float('inf')) < self.checkpoint_manager.best_metric
            self.checkpoint_manager.save(
                model=trainer.model,
                epoch=epoch,
                metrics=metrics,
                optimizer=kwargs.get('optimizer'),
                is_best=is_best
            )
    
    def on_train_begin(self, trainer: Any, **kwargs):
        """Called at train begin"""
        pass
    
    def on_train_end(self, trainer: Any, **kwargs):
        """Called at train end"""
        pass













