"""
Training Callbacks
==================

Callbacks for training hooks and monitoring.
"""

import torch
import logging
from typing import Dict, Any, Optional, Callable, List
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class Callback(ABC):
    """Base callback class."""
    
    @abstractmethod
    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None):
        """Called at the beginning of training."""
        pass
    
    @abstractmethod
    def on_train_end(self, logs: Optional[Dict[str, Any]] = None):
        """Called at the end of training."""
        pass
    
    @abstractmethod
    def on_epoch_begin(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        """Called at the beginning of an epoch."""
        pass
    
    @abstractmethod
    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        """Called at the end of an epoch."""
        pass
    
    @abstractmethod
    def on_batch_begin(self, batch: int, logs: Optional[Dict[str, Any]] = None):
        """Called at the beginning of a batch."""
        pass
    
    @abstractmethod
    def on_batch_end(self, batch: int, logs: Optional[Dict[str, Any]] = None):
        """Called at the end of a batch."""
        pass


class EarlyStoppingCallback(Callback):
    """Early stopping callback."""
    
    def __init__(
        self,
        monitor: str = "val_loss",
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = "min"
    ):
        """
        Initialize early stopping.
        
        Args:
            monitor: Metric to monitor
            patience: Number of epochs to wait
            min_delta: Minimum change to qualify as improvement
            mode: "min" or "max"
        """
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_value = float('inf') if mode == "min" else float('-inf')
        self.wait = 0
        self.stopped_epoch = 0
        self.best_weights = None
    
    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None):
        """Reset state."""
        self.wait = 0
        self.best_value = float('inf') if self.mode == "min" else float('-inf')
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        """Check if should stop."""
        if logs is None or self.monitor not in logs:
            return
        
        current = logs[self.monitor]
        
        if self.mode == "min":
            improved = current < (self.best_value - self.min_delta)
        else:
            improved = current > (self.best_value + self.min_delta)
        
        if improved:
            self.best_value = current
            self.wait = 0
        else:
            self.wait += 1
        
        if self.wait >= self.patience:
            self.stopped_epoch = epoch
            logger.info(f"Early stopping at epoch {epoch}")
    
    def on_train_end(self, logs: Optional[Dict[str, Any]] = None):
        """Called at end of training."""
        pass
    
    def on_epoch_begin(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        """Called at beginning of epoch."""
        pass
    
    def on_batch_begin(self, batch: int, logs: Optional[Dict[str, Any]] = None):
        """Called at beginning of batch."""
        pass
    
    def on_batch_end(self, batch: int, logs: Optional[Dict[str, Any]] = None):
        """Called at end of batch."""
        pass


class ModelCheckpointCallback(Callback):
    """Model checkpoint callback."""
    
    def __init__(
        self,
        filepath: str,
        monitor: str = "val_loss",
        save_best_only: bool = True,
        mode: str = "min"
    ):
        """
        Initialize checkpoint callback.
        
        Args:
            filepath: Path to save checkpoints
            monitor: Metric to monitor
            save_best_only: Only save best model
            mode: "min" or "max"
        """
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.mode = mode
        self.best_value = float('inf') if mode == "min" else float('-inf')
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        """Save checkpoint if needed."""
        if logs is None or self.monitor not in logs:
            return
        
        current = logs[self.monitor]
        
        if self.mode == "min":
            improved = current < self.best_value
        else:
            improved = current > self.best_value
        
        if improved or not self.save_best_only:
            self.best_value = current
            # Save would be handled by trainer
            logger.info(f"Checkpoint saved at epoch {epoch}")


class LearningRateSchedulerCallback(Callback):
    """Learning rate scheduler callback."""
    
    def __init__(self, scheduler: torch.optim.lr_scheduler._LRScheduler):
        """
        Initialize LR scheduler callback.
        
        Args:
            scheduler: Learning rate scheduler
        """
        self.scheduler = scheduler
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        """Step scheduler."""
        self.scheduler.step()
        current_lr = self.scheduler.get_last_lr()[0]
        logger.debug(f"Learning rate: {current_lr}")


class CallbackList:
    """List of callbacks."""
    
    def __init__(self, callbacks: Optional[List[Callback]] = None):
        """
        Initialize callback list.
        
        Args:
            callbacks: List of callbacks
        """
        self.callbacks = callbacks or []
    
    def append(self, callback: Callback):
        """Add callback."""
        self.callbacks.append(callback)
    
    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None):
        """Call on_train_begin for all callbacks."""
        for callback in self.callbacks:
            callback.on_train_begin(logs)
    
    def on_train_end(self, logs: Optional[Dict[str, Any]] = None):
        """Call on_train_end for all callbacks."""
        for callback in self.callbacks:
            callback.on_train_end(logs)
    
    def on_epoch_begin(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        """Call on_epoch_begin for all callbacks."""
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, logs)
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        """Call on_epoch_end for all callbacks."""
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, logs)
    
    def on_batch_begin(self, batch: int, logs: Optional[Dict[str, Any]] = None):
        """Call on_batch_begin for all callbacks."""
        for callback in self.callbacks:
            callback.on_batch_begin(batch, logs)
    
    def on_batch_end(self, batch: int, logs: Optional[Dict[str, Any]] = None):
        """Call on_batch_end for all callbacks."""
        for callback in self.callbacks:
            callback.on_batch_end(batch, logs)




