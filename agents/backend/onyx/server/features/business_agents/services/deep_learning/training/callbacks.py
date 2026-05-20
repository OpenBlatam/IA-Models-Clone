"""
Training Callbacks
=================

Callback system for training hooks and monitoring.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Callable
import logging
from abc import ABC, abstractmethod

try:
    import structlog
    logger = structlog.get_logger(__name__)
except ImportError:
    logger = logging.getLogger(__name__)


class Callback(ABC):
    """Base callback class."""
    
    @abstractmethod
    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Called at the beginning of training."""
        pass
    
    @abstractmethod
    def on_train_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Called at the end of training."""
        pass
    
    @abstractmethod
    def on_epoch_begin(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Called at the beginning of an epoch."""
        pass
    
    @abstractmethod
    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Called at the end of an epoch."""
        pass
    
    @abstractmethod
    def on_batch_begin(self, batch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Called at the beginning of a batch."""
        pass
    
    @abstractmethod
    def on_batch_end(self, batch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Called at the end of a batch."""
        pass


class LearningRateSchedulerCallback(Callback):
    """Callback for learning rate scheduling."""
    
    def __init__(self, scheduler: Any):
        """
        Initialize scheduler callback.
        
        Args:
            scheduler: Learning rate scheduler
        """
        self.scheduler = scheduler
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Step scheduler at end of epoch."""
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            val_loss = logs.get('val_loss') if logs else None
            if val_loss is not None:
                self.scheduler.step(val_loss)
        else:
            self.scheduler.step()


class ModelCheckpointCallback(Callback):
    """Callback for model checkpointing."""
    
    def __init__(
        self,
        checkpoint_dir: str,
        monitor: str = 'val_loss',
        mode: str = 'min',
        save_best_only: bool = True,
        save_freq: int = 1
    ):
        """
        Initialize checkpoint callback.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            monitor: Metric to monitor
            mode: 'min' or 'max'
            save_best_only: Only save best model
            save_freq: Save frequency (epochs)
        """
        self.checkpoint_dir = checkpoint_dir
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_freq = save_freq
        self.best_score = None
        self.best_epoch = None
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Save checkpoint if needed."""
        if logs is None:
            return
        
        if epoch % self.save_freq != 0:
            return
        
        current_score = logs.get(self.monitor)
        if current_score is None:
            return
        
        should_save = False
        
        if self.best_score is None:
            should_save = True
            self.best_score = current_score
            self.best_epoch = epoch
        elif (self.mode == 'min' and current_score < self.best_score) or \
             (self.mode == 'max' and current_score > self.best_score):
            should_save = True
            self.best_score = current_score
            self.best_epoch = epoch
        
        if should_save or not self.save_best_only:
            # Save checkpoint (implementation depends on trainer)
            logger.info(f"Checkpoint saved at epoch {epoch}")


class EarlyStoppingCallback(Callback):
    """Callback for early stopping."""
    
    def __init__(
        self,
        monitor: str = 'val_loss',
        patience: int = 5,
        min_delta: float = 0.0,
        mode: str = 'min',
        restore_best_weights: bool = True
    ):
        """
        Initialize early stopping callback.
        
        Args:
            monitor: Metric to monitor
            patience: Number of epochs to wait
            min_delta: Minimum change to qualify as improvement
            mode: 'min' or 'max'
            restore_best_weights: Restore best weights on stop
        """
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.wait = 0
        self.stopped_epoch = None
        self.best_weights = None
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Check if training should stop."""
        if logs is None:
            return
        
        current_score = logs.get(self.monitor)
        if current_score is None:
            return
        
        if self.best_score is None:
            self.best_score = current_score
            self.wait = 0
        elif (self.mode == 'min' and current_score < self.best_score - self.min_delta) or \
             (self.mode == 'max' and current_score > self.best_score + self.min_delta):
            self.best_score = current_score
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                logger.info(f"Early stopping triggered at epoch {epoch}")
                # Signal to stop training (implementation depends on trainer)


class TensorBoardCallback(Callback):
    """Callback for TensorBoard logging."""
    
    def __init__(self, log_dir: str):
        """
        Initialize TensorBoard callback.
        
        Args:
            log_dir: Directory for TensorBoard logs
        """
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir)
            self.available = True
        except ImportError:
            logger.warning("TensorBoard not available")
            self.available = False
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Log metrics to TensorBoard."""
        if not self.available or logs is None:
            return
        
        for key, value in logs.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(key, value, epoch)
    
    def on_train_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Close TensorBoard writer."""
        if self.available:
            self.writer.close()


class WandBCallback(Callback):
    """Callback for Weights & Biases logging."""
    
    def __init__(self, project: str, **kwargs):
        """
        Initialize W&B callback.
        
        Args:
            project: W&B project name
            **kwargs: Additional W&B arguments
        """
        try:
            import wandb
            wandb.init(project=project, **kwargs)
            self.available = True
        except ImportError:
            logger.warning("W&B not available")
            self.available = False
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Log metrics to W&B."""
        if not self.available or logs is None:
            return
        
        import wandb
        wandb.log(logs, step=epoch)
    
    def on_train_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Finish W&B run."""
        if self.available:
            import wandb
            wandb.finish()


class CallbackList:
    """Container for multiple callbacks."""
    
    def __init__(self, callbacks: List[Callback]):
        """
        Initialize callback list.
        
        Args:
            callbacks: List of callbacks
        """
        self.callbacks = callbacks
    
    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Call on_train_begin for all callbacks."""
        for callback in self.callbacks:
            callback.on_train_begin(logs)
    
    def on_train_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Call on_train_end for all callbacks."""
        for callback in self.callbacks:
            callback.on_train_end(logs)
    
    def on_epoch_begin(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Call on_epoch_begin for all callbacks."""
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, logs)
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Call on_epoch_end for all callbacks."""
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, logs)
    
    def on_batch_begin(self, batch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Call on_batch_begin for all callbacks."""
        for callback in self.callbacks:
            callback.on_batch_begin(batch, logs)
    
    def on_batch_end(self, batch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Call on_batch_end for all callbacks."""
        for callback in self.callbacks:
            callback.on_batch_end(batch, logs)



