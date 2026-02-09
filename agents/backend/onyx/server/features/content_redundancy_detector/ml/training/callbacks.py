"""
Training Callbacks
Modular callback system for training hooks
"""

import logging
import torch
from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod
from pathlib import Path

logger = logging.getLogger(__name__)


class Callback(ABC):
    """
    Base callback class
    All callbacks should inherit from this
    """
    
    def on_train_begin(self, trainer: Any, logs: Optional[Dict[str, Any]] = None) -> None:
        """Called at the beginning of training"""
        pass
    
    def on_train_end(self, trainer: Any, logs: Optional[Dict[str, Any]] = None) -> None:
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
    """
    Early stopping callback
    Stops training when monitored metric stops improving
    """
    
    def __init__(
        self,
        monitor: str = "val_loss",
        patience: int = 5,
        min_delta: float = 0.0,
        mode: str = "min",
        restore_best_weights: bool = True,
    ):
        """
        Initialize early stopping
        
        Args:
            monitor: Metric to monitor
            patience: Number of epochs to wait
            min_delta: Minimum change to qualify as improvement
            mode: 'min' or 'max'
            restore_best_weights: Restore best weights when stopping
        """
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.wait = 0
        self.best_weights = None
        self.stopped_epoch = 0
    
    def on_train_begin(self, trainer: Any, logs: Optional[Dict[str, Any]] = None) -> None:
        """Initialize at training start"""
        self.wait = 0
        self.best_value = float('inf') if self.mode == 'min' else float('-inf')
        self.best_weights = None
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Check if training should stop"""
        if logs is None or self.monitor not in logs:
            return
        
        current_value = logs[self.monitor]
        
        if self.mode == 'min':
            is_better = current_value < (self.best_value - self.min_delta)
        else:
            is_better = current_value > (self.best_value + self.min_delta)
        
        if is_better:
            self.best_value = current_value
            self.wait = 0
            if self.restore_best_weights and hasattr(logs, 'model'):
                self.best_weights = logs['model'].state_dict().copy()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                if self.restore_best_weights and self.best_weights is not None:
                    logs['model'].load_state_dict(self.best_weights)
                logger.info(f"Early stopping triggered at epoch {epoch}")
                return True  # Signal to stop training
        
        return False


class ModelCheckpointCallback(Callback):
    """
    Model checkpoint callback
    Saves model checkpoints during training
    """
    
    def __init__(
        self,
        checkpoint_dir: Path,
        monitor: str = "val_loss",
        mode: str = "min",
        save_best_only: bool = True,
        save_freq: int = 1,
    ):
        """
        Initialize checkpoint callback
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            monitor: Metric to monitor
            mode: 'min' or 'max'
            save_best_only: Only save best model
            save_freq: Save every N epochs
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_freq = save_freq
        
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.epochs_since_last_save = 0
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Save checkpoint if needed"""
        if logs is None:
            return
        
        self.epochs_since_last_save += 1
        
        # Check if we should save
        should_save = False
        if self.save_best_only:
            if self.monitor in logs:
                current_value = logs[self.monitor]
                if self.mode == 'min':
                    is_better = current_value < self.best_value
                else:
                    is_better = current_value > self.best_value
                
                if is_better:
                    self.best_value = current_value
                    should_save = True
        else:
            if self.epochs_since_last_save >= self.save_freq:
                should_save = True
                self.epochs_since_last_save = 0
        
        if should_save and 'model' in logs and 'optimizer' in logs:
            checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': logs['model'].state_dict(),
                'optimizer_state_dict': logs['optimizer'].state_dict(),
                'best_value': self.best_value,
                'logs': logs,
            }, checkpoint_path)
            logger.info(f"Saved checkpoint: {checkpoint_path}")


class LearningRateSchedulerCallback(Callback):
    """
    Learning rate scheduler callback
    Updates learning rate based on schedule
    """
    
    def __init__(self, scheduler: torch.optim.lr_scheduler._LRScheduler):
        """
        Initialize scheduler callback
        
        Args:
            scheduler: Learning rate scheduler
        """
        self.scheduler = scheduler
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Update learning rate"""
        self.scheduler.step()
        if logs is not None:
            current_lr = self.scheduler.get_last_lr()[0]
            logs['learning_rate'] = current_lr
            logger.debug(f"Learning rate updated to {current_lr:.6f}")


class ExperimentTrackingCallback(Callback):
    """
    Experiment tracking callback
    Logs metrics to wandb or tensorboard
    """
    
    def __init__(
        self,
        tracker_type: str = "wandb",
        project_name: Optional[str] = None,
        run_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize experiment tracking
        
        Args:
            tracker_type: 'wandb' or 'tensorboard'
            project_name: Project name
            run_name: Run name
            config: Configuration to log
        """
        self.tracker_type = tracker_type
        self.project_name = project_name
        self.run_name = run_name
        self.config = config or {}
        self.tracker = None
        
        self._initialize_tracker()
    
    def _initialize_tracker(self) -> None:
        """Initialize the tracking backend"""
        if self.tracker_type == "wandb":
            try:
                import wandb
                wandb.init(
                    project=self.project_name,
                    name=self.run_name,
                    config=self.config
                )
                self.tracker = wandb
                logger.info("Initialized wandb tracking")
            except ImportError:
                logger.warning("wandb not available, skipping tracking")
        elif self.tracker_type == "tensorboard":
            try:
                from torch.utils.tensorboard import SummaryWriter
                log_dir = f"runs/{self.run_name or 'experiment'}"
                self.tracker = SummaryWriter(log_dir=log_dir)
                logger.info(f"Initialized tensorboard tracking: {log_dir}")
            except ImportError:
                logger.warning("tensorboard not available, skipping tracking")
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Log metrics"""
        if self.tracker is None or logs is None:
            return
        
        if self.tracker_type == "wandb":
            self.tracker.log(logs, step=epoch)
        elif self.tracker_type == "tensorboard":
            for key, value in logs.items():
                if isinstance(value, (int, float)):
                    self.tracker.add_scalar(key, value, epoch)
    
    def on_train_end(self, trainer: Any, logs: Optional[Dict[str, Any]] = None) -> None:
        """Close tracker"""
        if self.tracker is not None:
            if self.tracker_type == "wandb":
                self.tracker.finish()
            elif self.tracker_type == "tensorboard":
                self.tracker.close()


class CallbackList:
    """
    Container for managing multiple callbacks
    """
    
    def __init__(self, callbacks: Optional[List[Callback]] = None):
        """
        Initialize callback list
        
        Args:
            callbacks: List of callbacks
        """
        self.callbacks = callbacks or []
    
    def append(self, callback: Callback) -> None:
        """Add a callback"""
        self.callbacks.append(callback)
    
    def on_train_begin(self, trainer: Any, logs: Optional[Dict[str, Any]] = None) -> None:
        """Call on_train_begin for all callbacks"""
        for callback in self.callbacks:
            callback.on_train_begin(trainer, logs)
    
    def on_train_end(self, trainer: Any, logs: Optional[Dict[str, Any]] = None) -> None:
        """Call on_train_end for all callbacks"""
        for callback in self.callbacks:
            callback.on_train_end(trainer, logs)
    
    def on_epoch_begin(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Call on_epoch_begin for all callbacks"""
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, logs)
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> bool:
        """Call on_epoch_end for all callbacks, return True if should stop"""
        should_stop = False
        for callback in self.callbacks:
            result = callback.on_epoch_end(epoch, logs)
            if result is True:
                should_stop = True
        return should_stop
    
    def on_batch_begin(self, batch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Call on_batch_begin for all callbacks"""
        for callback in self.callbacks:
            callback.on_batch_begin(batch, logs)
    
    def on_batch_end(self, batch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Call on_batch_end for all callbacks"""
        for callback in self.callbacks:
            callback.on_batch_end(batch, logs)



