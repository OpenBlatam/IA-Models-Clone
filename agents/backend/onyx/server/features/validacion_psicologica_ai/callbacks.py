"""
Training Callbacks
==================
Callbacks for training monitoring and control
"""

from typing import Dict, Any, List, Optional, Callable
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import structlog
from abc import ABC, abstractmethod
from pathlib import Path

logger = structlog.get_logger()


class Callback(ABC):
    """Base callback class"""
    
    @abstractmethod
    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Called at the beginning of training"""
        pass
    
    @abstractmethod
    def on_train_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Called at the end of training"""
        pass
    
    def on_epoch_begin(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Called at the beginning of an epoch"""
        pass
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Called at the end of an epoch"""
        pass
    
    def on_batch_begin(self, batch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Called at the beginning of a batch"""
        pass
    
    def on_batch_end(self, batch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Called at the end of a batch"""
        pass


class EarlyStoppingCallback(Callback):
    """Early stopping callback"""
    
    def __init__(
        self,
        monitor: str = "val_loss",
        patience: int = 5,
        min_delta: float = 0.0,
        mode: str = "min"
    ):
        """
        Initialize early stopping
        
        Args:
            monitor: Metric to monitor
            patience: Number of epochs to wait
            min_delta: Minimum change to qualify as improvement
            mode: 'min' or 'max'
        """
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = float('inf') if mode == "min" else float('-inf')
        self.wait = 0
        self.stopped_epoch = 0
        self.best_weights = None
    
    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Reset counters"""
        self.wait = 0
        self.best_score = float('inf') if self.mode == "min" else float('-inf')
    
    def on_epoch_end(
        self,
        epoch: int,
        logs: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Check if training should stop
        
        Returns:
            True if training should stop
        """
        if logs is None:
            return False
        
        current = logs.get(self.monitor)
        if current is None:
            return False
        
        if self.mode == "min":
            improved = current < (self.best_score - self.min_delta)
        else:
            improved = current > (self.best_score + self.min_delta)
        
        if improved:
            self.best_score = current
            self.wait = 0
            return False
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                logger.info(f"Early stopping triggered at epoch {epoch}")
                return True
        
        return False


class ModelCheckpointCallback(Callback):
    """Model checkpoint callback"""
    
    def __init__(
        self,
        filepath: str,
        monitor: str = "val_loss",
        save_best_only: bool = True,
        mode: str = "min"
    ):
        """
        Initialize checkpoint callback
        
        Args:
            filepath: Path to save checkpoints
            monitor: Metric to monitor
            save_best_only: Save only best model
            mode: 'min' or 'max'
        """
        self.filepath = Path(filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.mode = mode
        self.best_score = float('inf') if mode == "min" else float('-inf')
    
    def on_epoch_end(
        self,
        epoch: int,
        logs: Optional[Dict[str, Any]] = None,
        model: Optional[nn.Module] = None
    ) -> None:
        """Save checkpoint if needed"""
        if model is None or logs is None:
            return
        
        current = logs.get(self.monitor)
        if current is None:
            return
        
        should_save = False
        
        if self.save_best_only:
            if self.mode == "min":
                if current < self.best_score:
                    self.best_score = current
                    should_save = True
            else:
                if current > self.best_score:
                    self.best_score = current
                    should_save = True
        else:
            should_save = True
        
        if should_save:
            checkpoint_path = self.filepath.parent / f"{self.filepath.stem}_epoch_{epoch}.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "monitor_value": current,
                "logs": logs
            }, checkpoint_path)
            logger.info(f"Checkpoint saved: {checkpoint_path}")


class LearningRateSchedulerCallback(Callback):
    """Learning rate scheduler callback"""
    
    def __init__(
        self,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        monitor: str = "val_loss"
    ):
        """
        Initialize LR scheduler callback
        
        Args:
            scheduler: Learning rate scheduler
            monitor: Metric to monitor
        """
        self.scheduler = scheduler
        self.monitor = monitor
    
    def on_epoch_end(
        self,
        epoch: int,
        logs: Optional[Dict[str, Any]] = None
    ) -> None:
        """Step scheduler"""
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            if logs and self.monitor in logs:
                self.scheduler.step(logs[self.monitor])
        else:
            self.scheduler.step()


class TensorBoardCallback(Callback):
    """TensorBoard logging callback"""
    
    def __init__(self, log_dir: str = "./logs"):
        """
        Initialize TensorBoard callback
        
        Args:
            log_dir: Log directory
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=str(self.log_dir))
            self.enabled = True
        except ImportError:
            logger.warning("TensorBoard not available")
            self.writer = None
            self.enabled = False
    
    def on_epoch_end(
        self,
        epoch: int,
        logs: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log metrics to TensorBoard"""
        if not self.enabled or logs is None:
            return
        
        for key, value in logs.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(key, value, epoch)
    
    def on_train_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Close writer"""
        if self.enabled and self.writer:
            self.writer.close()


class CallbackList:
    """List of callbacks"""
    
    def __init__(self, callbacks: List[Callback]):
        """
        Initialize callback list
        
        Args:
            callbacks: List of callbacks
        """
        self.callbacks = callbacks
    
    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Call on_train_begin for all callbacks"""
        for callback in self.callbacks:
            callback.on_train_begin(logs)
    
    def on_train_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Call on_train_end for all callbacks"""
        for callback in self.callbacks:
            callback.on_train_end(logs)
    
    def on_epoch_begin(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Call on_epoch_begin for all callbacks"""
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, logs)
    
    def on_epoch_end(
        self,
        epoch: int,
        logs: Optional[Dict[str, Any]] = None,
        model: Optional[nn.Module] = None
    ) -> bool:
        """
        Call on_epoch_end for all callbacks
        
        Returns:
            True if training should stop
        """
        should_stop = False
        for callback in self.callbacks:
            if isinstance(callback, EarlyStoppingCallback):
                if callback.on_epoch_end(epoch, logs):
                    should_stop = True
            elif isinstance(callback, ModelCheckpointCallback):
                callback.on_epoch_end(epoch, logs, model)
            else:
                callback.on_epoch_end(epoch, logs)
        
        return should_stop
    
    def on_batch_begin(self, batch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Call on_batch_begin for all callbacks"""
        for callback in self.callbacks:
            callback.on_batch_begin(batch, logs)
    
    def on_batch_end(self, batch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Call on_batch_end for all callbacks"""
        for callback in self.callbacks:
            callback.on_batch_end(batch, logs)




