"""
Training Callbacks - Advanced Callback System
==============================================

Provides callback system for training hooks:
- Model checkpointing
- Learning rate scheduling
- Early stopping
- Metrics logging
- Custom callbacks
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from pathlib import Path
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class Callback(ABC):
    """Abstract base class for training callbacks."""
    
    @abstractmethod
    def on_train_begin(self, trainer: Any, **kwargs) -> None:
        """Called at the beginning of training."""
        pass
    
    @abstractmethod
    def on_train_end(self, trainer: Any, **kwargs) -> None:
        """Called at the end of training."""
        pass
    
    @abstractmethod
    def on_epoch_begin(self, epoch: int, **kwargs) -> None:
        """Called at the beginning of each epoch."""
        pass
    
    @abstractmethod
    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], **kwargs) -> None:
        """Called at the end of each epoch."""
        pass
    
    @abstractmethod
    def on_batch_begin(self, batch_idx: int, **kwargs) -> None:
        """Called at the beginning of each batch."""
        pass
    
    @abstractmethod
    def on_batch_end(self, batch_idx: int, loss: float, **kwargs) -> None:
        """Called at the end of each batch."""
        pass


class ModelCheckpoint(Callback):
    """Callback for saving model checkpoints."""
    
    def __init__(
        self,
        save_dir: Path,
        monitor: str = 'val_loss',
        mode: str = 'min',
        save_best_only: bool = True,
        save_frequency: int = 1
    ):
        """
        Initialize model checkpoint callback.
        
        Args:
            save_dir: Directory to save checkpoints
            monitor: Metric to monitor
            mode: 'min' or 'max'
            save_best_only: Only save best model
            save_frequency: Save every N epochs
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_frequency = save_frequency
        self.best_score = float('inf') if mode == 'min' else float('-inf')
        self.model = None
    
    def on_train_begin(self, trainer: Any, **kwargs) -> None:
        """Store model reference."""
        self.model = trainer.model
    
    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], **kwargs) -> None:
        """Save checkpoint if conditions are met."""
        if epoch % self.save_frequency != 0:
            return
        
        if self.monitor not in metrics:
            logger.warning(f"Metric {self.monitor} not found in metrics")
            return
        
        current_score = metrics[self.monitor]
        should_save = False
        
        if self.save_best_only:
            if (self.mode == 'min' and current_score < self.best_score) or \
               (self.mode == 'max' and current_score > self.best_score):
                self.best_score = current_score
                should_save = True
        else:
            should_save = True
        
        if should_save:
            checkpoint_path = self.save_dir / f"checkpoint_epoch_{epoch+1}.pt"
            model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
            model_to_save.save_checkpoint(
                checkpoint_path,
                optimizer=kwargs.get('optimizer'),
                scheduler=kwargs.get('scheduler'),
                epoch=epoch,
                metrics=metrics
            )
            logger.info(f"Checkpoint saved: {checkpoint_path}")


class LearningRateScheduler(Callback):
    """Callback for learning rate scheduling."""
    
    def __init__(self, scheduler: Any):
        """
        Initialize LR scheduler callback.
        
        Args:
            scheduler: Learning rate scheduler
        """
        self.scheduler = scheduler
    
    def on_batch_end(self, batch_idx: int, **kwargs) -> None:
        """Step scheduler after batch."""
        if self.scheduler is not None:
            self.scheduler.step()
    
    def on_epoch_end(self, epoch: int, **kwargs) -> None:
        """Step scheduler after epoch (if needed)."""
        # Some schedulers step per epoch
        pass


class MetricsLogger(Callback):
    """Callback for logging metrics."""
    
    def __init__(self, tracker: Optional[Any] = None):
        """
        Initialize metrics logger.
        
        Args:
            tracker: Experiment tracker (TensorBoard/W&B)
        """
        self.tracker = tracker
    
    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], **kwargs) -> None:
        """Log metrics."""
        if self.tracker:
            for name, value in metrics.items():
                self.tracker.log_scalar(name, value, epoch)


class CallbackList:
    """Container for multiple callbacks."""
    
    def __init__(self, callbacks: Optional[List[Callback]] = None):
        """
        Initialize callback list.
        
        Args:
            callbacks: List of callbacks
        """
        self.callbacks = callbacks or []
    
    def append(self, callback: Callback) -> None:
        """Add a callback."""
        self.callbacks.append(callback)
    
    def on_train_begin(self, trainer: Any, **kwargs) -> None:
        """Call on_train_begin for all callbacks."""
        for callback in self.callbacks:
            callback.on_train_begin(trainer, **kwargs)
    
    def on_train_end(self, trainer: Any, **kwargs) -> None:
        """Call on_train_end for all callbacks."""
        for callback in self.callbacks:
            callback.on_train_end(trainer, **kwargs)
    
    def on_epoch_begin(self, epoch: int, **kwargs) -> None:
        """Call on_epoch_begin for all callbacks."""
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, **kwargs)
    
    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], **kwargs) -> None:
        """Call on_epoch_end for all callbacks."""
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, metrics, **kwargs)
    
    def on_batch_begin(self, batch_idx: int, **kwargs) -> None:
        """Call on_batch_begin for all callbacks."""
        for callback in self.callbacks:
            callback.on_batch_begin(batch_idx, **kwargs)
    
    def on_batch_end(self, batch_idx: int, loss: float, **kwargs) -> None:
        """Call on_batch_end for all callbacks."""
        for callback in self.callbacks:
            callback.on_batch_end(batch_idx, loss, **kwargs)



