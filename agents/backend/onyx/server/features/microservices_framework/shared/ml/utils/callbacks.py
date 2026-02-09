"""
Callbacks
Callback system for training and inference.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import torch
import logging

logger = logging.getLogger(__name__)


class Callback(ABC):
    """Base callback class."""
    
    @abstractmethod
    def on_epoch_start(self, epoch: int, **kwargs):
        """Called at the start of an epoch."""
        pass
    
    @abstractmethod
    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], **kwargs):
        """Called at the end of an epoch."""
        pass
    
    def on_batch_start(self, batch_idx: int, **kwargs):
        """Called at the start of a batch."""
        pass
    
    def on_batch_end(self, batch_idx: int, loss: float, **kwargs):
        """Called at the end of a batch."""
        pass
    
    def on_training_start(self, **kwargs):
        """Called at the start of training."""
        pass
    
    def on_training_end(self, **kwargs):
        """Called at the end of training."""
        pass


class EarlyStoppingCallback(Callback):
    """Early stopping callback."""
    
    def __init__(
        self,
        monitor: str = "val_loss",
        patience: int = 5,
        min_delta: float = 0.0,
        mode: str = "min",
        restore_best_weights: bool = True,
    ):
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        self.early_stopped = False
    
    def on_epoch_start(self, epoch: int, **kwargs):
        pass
    
    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], model: torch.nn.Module, **kwargs):
        if self.monitor not in metrics:
            logger.warning(f"Monitor metric {self.monitor} not found in metrics")
            return
        
        current_score = metrics[self.monitor]
        
        if self.best_score is None:
            self.best_score = current_score
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        elif self._is_better(current_score, self.best_score):
            self.best_score = current_score
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stopped = True
                if self.restore_best_weights and self.best_weights:
                    model.load_state_dict(self.best_weights)
                    logger.info("Early stopping triggered. Restored best weights.")
    
    def _is_better(self, current: float, best: float) -> bool:
        if self.mode == "min":
            return current < best - self.min_delta
        else:
            return current > best + self.min_delta
    
    def on_training_start(self, **kwargs):
        pass
    
    def on_training_end(self, **kwargs):
        pass


class ModelCheckpointCallback(Callback):
    """Model checkpointing callback."""
    
    def __init__(
        self,
        save_dir: str,
        monitor: str = "val_loss",
        mode: str = "min",
        save_best: bool = True,
        save_last: bool = True,
        save_frequency: int = 1,
    ):
        self.save_dir = save_dir
        self.monitor = monitor
        self.mode = mode
        self.save_best = save_best
        self.save_last = save_last
        self.save_frequency = save_frequency
        
        self.best_score = None
        self.last_epoch = 0
        
        from pathlib import Path
        Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    def on_epoch_end(
        self,
        epoch: int,
        metrics: Dict[str, float],
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        **kwargs
    ):
        self.last_epoch = epoch
        
        # Save last checkpoint
        if self.save_last and epoch % self.save_frequency == 0:
            self._save_checkpoint(
                model,
                optimizer,
                epoch,
                metrics,
                f"checkpoint_epoch_{epoch}.pt"
            )
        
        # Save best checkpoint
        if self.save_best and self.monitor in metrics:
            current_score = metrics[self.monitor]
            
            if self.best_score is None:
                self.best_score = current_score
                self._save_checkpoint(
                    model,
                    optimizer,
                    epoch,
                    metrics,
                    "best_model.pt"
                )
            elif self._is_better(current_score, self.best_score):
                self.best_score = current_score
                self._save_checkpoint(
                    model,
                    optimizer,
                    epoch,
                    metrics,
                    "best_model.pt"
                )
    
    def _is_better(self, current: float, best: float) -> bool:
        if self.mode == "min":
            return current < best
        else:
            return current > best
    
    def _save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        epoch: int,
        metrics: Dict[str, float],
        filename: str,
    ):
        from pathlib import Path
        checkpoint_path = Path(self.save_dir) / filename
        
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "metrics": metrics,
        }
        
        if optimizer:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def on_epoch_start(self, epoch: int, **kwargs):
        pass
    
    def on_training_start(self, **kwargs):
        pass
    
    def on_training_end(self, **kwargs):
        pass


class LoggingCallback(Callback):
    """Logging callback."""
    
    def __init__(self, log_interval: int = 10):
        self.log_interval = log_interval
    
    def on_epoch_start(self, epoch: int, **kwargs):
        logger.info(f"Starting epoch {epoch}")
    
    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], **kwargs):
        logger.info(f"Epoch {epoch} completed. Metrics: {metrics}")
    
    def on_batch_end(self, batch_idx: int, loss: float, **kwargs):
        if batch_idx % self.log_interval == 0:
            logger.info(f"Batch {batch_idx}, Loss: {loss:.4f}")
    
    def on_training_start(self, **kwargs):
        logger.info("Training started")
    
    def on_training_end(self, **kwargs):
        logger.info("Training completed")


class CallbackManager:
    """Manager for multiple callbacks."""
    
    def __init__(self, callbacks: Optional[list] = None):
        self.callbacks = callbacks or []
    
    def add_callback(self, callback: Callback):
        """Add a callback."""
        self.callbacks.append(callback)
    
    def on_epoch_start(self, epoch: int, **kwargs):
        for callback in self.callbacks:
            callback.on_epoch_start(epoch, **kwargs)
    
    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], **kwargs):
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, metrics, **kwargs)
    
    def on_batch_start(self, batch_idx: int, **kwargs):
        for callback in self.callbacks:
            callback.on_batch_start(batch_idx, **kwargs)
    
    def on_batch_end(self, batch_idx: int, loss: float, **kwargs):
        for callback in self.callbacks:
            callback.on_batch_end(batch_idx, loss, **kwargs)
    
    def on_training_start(self, **kwargs):
        for callback in self.callbacks:
            callback.on_training_start(**kwargs)
    
    def on_training_end(self, **kwargs):
        for callback in self.callbacks:
            callback.on_training_end(**kwargs)



