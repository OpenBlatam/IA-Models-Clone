"""
Training Callbacks
Modular callbacks for training events
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import torch
import logging

logger = logging.getLogger(__name__)


class TrainingCallback(ABC):
    """Abstract base class for training callbacks"""
    
    @abstractmethod
    def on_epoch_start(self, epoch: int, trainer: Any):
        """Called at the start of each epoch"""
        pass
    
    @abstractmethod
    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], trainer: Any):
        """Called at the end of each epoch"""
        pass
    
    def on_batch_start(self, batch_idx: int, trainer: Any):
        """Called at the start of each batch"""
        pass
    
    def on_batch_end(self, batch_idx: int, loss: float, trainer: Any):
        """Called at the end of each batch"""
        pass
    
    def on_training_start(self, trainer: Any):
        """Called at the start of training"""
        pass
    
    def on_training_end(self, trainer: Any):
        """Called at the end of training"""
        pass


class EarlyStoppingCallback(TrainingCallback):
    """Early stopping callback"""
    
    def __init__(
        self,
        monitor: str = "val_loss",
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = "min"
    ):
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_value = float('inf') if mode == "min" else float('-inf')
        self.patience_counter = 0
        self.stopped = False
    
    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], trainer: Any):
        """Check for early stopping"""
        if self.monitor not in metrics:
            return
        
        current_value = metrics[self.monitor]
        
        if self.mode == "min":
            is_better = current_value < (self.best_value - self.min_delta)
        else:
            is_better = current_value > (self.best_value + self.min_delta)
        
        if is_better:
            self.best_value = current_value
            self.patience_counter = 0
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.patience:
                self.stopped = True
                logger.info(f"Early stopping triggered at epoch {epoch}")
    
    def on_epoch_start(self, epoch: int, trainer: Any):
        """Check if training should stop"""
        if self.stopped:
            trainer.should_stop = True
    
    def on_training_start(self, trainer: Any):
        """Initialize"""
        self.stopped = False
        self.patience_counter = 0


class ModelCheckpointCallback(TrainingCallback):
    """Model checkpointing callback"""
    
    def __init__(
        self,
        checkpoint_dir: str,
        save_best: bool = True,
        save_frequency: int = 10,
        monitor: str = "val_loss",
        mode: str = "min"
    ):
        from pathlib import Path
        
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_best = save_best
        self.save_frequency = save_frequency
        self.monitor = monitor
        self.mode = mode
        self.best_value = float('inf') if mode == "min" else float('-inf')
        self.best_epoch = 0
    
    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], trainer: Any):
        """Save checkpoint"""
        # Save best model
        if self.save_best and self.monitor in metrics:
            current_value = metrics[self.monitor]
            is_best = (
                (self.mode == "min" and current_value < self.best_value) or
                (self.mode == "max" and current_value > self.best_value)
            )
            
            if is_best:
                self.best_value = current_value
                self.best_epoch = epoch
                self._save_checkpoint(trainer, epoch, is_best=True)
        
        # Save periodic checkpoints
        if (epoch + 1) % self.save_frequency == 0:
            self._save_checkpoint(trainer, epoch, is_best=False)
    
    def _save_checkpoint(self, trainer: Any, epoch: int, is_best: bool = False):
        """Save checkpoint"""
        suffix = "best" if is_best else f"epoch_{epoch + 1}"
        checkpoint_path = self.checkpoint_dir / f"checkpoint_{suffix}.pt"
        trainer.save_checkpoint(checkpoint_path, trainer.optimizer, epoch)
    
    def on_epoch_start(self, epoch: int, trainer: Any):
        """Called at epoch start"""
        pass
    
    def on_training_start(self, trainer: Any):
        """Initialize"""
        pass
    
    def on_training_end(self, trainer: Any):
        """Save final checkpoint"""
        pass


class LearningRateSchedulerCallback(TrainingCallback):
    """Learning rate scheduling callback"""
    
    def __init__(self, scheduler: torch.optim.lr_scheduler._LRScheduler):
        self.scheduler = scheduler
    
    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], trainer: Any):
        """Step scheduler"""
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            monitor_value = metrics.get("val_loss", metrics.get("train_loss", 0))
            self.scheduler.step(monitor_value)
        else:
            self.scheduler.step()
    
    def on_epoch_start(self, epoch: int, trainer: Any):
        """Called at epoch start"""
        pass
    
    def on_training_start(self, trainer: Any):
        """Initialize"""
        pass
    
    def on_training_end(self, trainer: Any):
        """Called at training end"""
        pass


class MetricsLoggingCallback(TrainingCallback):
    """Logging callback for metrics"""
    
    def __init__(self, log_frequency: int = 1):
        self.log_frequency = log_frequency
    
    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], trainer: Any):
        """Log metrics"""
        if (epoch + 1) % self.log_frequency == 0:
            logger.info(f"Epoch {epoch + 1} Metrics: {metrics}")
    
    def on_epoch_start(self, epoch: int, trainer: Any):
        """Called at epoch start"""
        pass
    
    def on_training_start(self, trainer: Any):
        """Initialize"""
        pass
    
    def on_training_end(self, trainer: Any):
        """Called at training end"""
        pass













