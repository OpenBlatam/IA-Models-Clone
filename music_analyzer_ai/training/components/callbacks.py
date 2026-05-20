"""
Modular Training Callbacks
Callback system for training hooks
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class TrainingCallback(ABC):
    """Base class for training callbacks"""
    
    @abstractmethod
    def on_epoch_start(self, epoch: int, **kwargs):
        """Called at the start of each epoch"""
        pass
    
    @abstractmethod
    def on_epoch_end(self, epoch: int, metrics: Dict[str, Any], **kwargs):
        """Called at the end of each epoch"""
        pass
    
    @abstractmethod
    def on_batch_end(self, batch_idx: int, metrics: Dict[str, Any], **kwargs):
        """Called at the end of each batch"""
        pass
    
    def on_training_start(self, **kwargs):
        """Called at the start of training"""
        pass
    
    def on_training_end(self, **kwargs):
        """Called at the end of training"""
        pass


class EarlyStoppingCallback(TrainingCallback):
    """Early stopping callback"""
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        monitor: str = "val_loss",
        mode: str = "min"
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.mode = mode
        self.best_score = float('inf') if mode == "min" else float('-inf')
        self.patience_counter = 0
        self.should_stop = False
    
    def on_epoch_start(self, epoch: int, **kwargs):
        """Reset at epoch start"""
        pass
    
    def on_epoch_end(self, epoch: int, metrics: Dict[str, Any], **kwargs):
        """Check for early stopping"""
        if self.monitor not in metrics:
            return
        
        current_score = metrics[self.monitor]
        
        if self.mode == "min":
            is_better = current_score < (self.best_score - self.min_delta)
        else:
            is_better = current_score > (self.best_score + self.min_delta)
        
        if is_better:
            self.best_score = current_score
            self.patience_counter = 0
        else:
            self.patience_counter += 1
        
        if self.patience_counter >= self.patience:
            self.should_stop = True
            logger.info(f"Early stopping triggered at epoch {epoch}")
    
    def on_batch_end(self, batch_idx: int, metrics: Dict[str, Any], **kwargs):
        """Not used for early stopping"""
        pass


class CheckpointCallback(TrainingCallback):
    """Checkpoint saving callback"""
    
    def __init__(
        self,
        checkpoint_dir: str = "./checkpoints",
        save_best: bool = True,
        save_every: Optional[int] = None,
        monitor: str = "val_loss",
        mode: str = "min"
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_best = save_best
        self.save_every = save_every
        self.monitor = monitor
        self.mode = mode
        self.best_score = float('inf') if mode == "min" else float('-inf')
    
    def on_epoch_start(self, epoch: int, **kwargs):
        """Not used"""
        pass
    
    def on_epoch_end(self, epoch: int, metrics: Dict[str, Any], **kwargs):
        """Save checkpoint if needed"""
        model = kwargs.get("model")
        optimizer = kwargs.get("optimizer")
        scheduler = kwargs.get("scheduler")
        
        if model is None:
            return
        
        # Save best model
        if self.save_best and self.monitor in metrics:
            current_score = metrics[self.monitor]
            is_better = (
                current_score < self.best_score if self.mode == "min"
                else current_score > self.best_score
            )
            
            if is_better:
                self.best_score = current_score
                self._save_checkpoint(
                    model, optimizer, scheduler, epoch, metrics, "best"
                )
        
        # Save periodic checkpoints
        if self.save_every and epoch % self.save_every == 0:
            self._save_checkpoint(
                model, optimizer, scheduler, epoch, metrics, f"epoch_{epoch}"
            )
    
    def on_batch_end(self, batch_idx: int, metrics: Dict[str, Any], **kwargs):
        """Not used"""
        pass
    
    def _save_checkpoint(
        self,
        model,
        optimizer,
        scheduler,
        epoch: int,
        metrics: Dict[str, Any],
        suffix: str
    ):
        """Save checkpoint to disk"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "metrics": metrics
        }
        
        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        
        if scheduler is not None:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()
        
        checkpoint_path = self.checkpoint_dir / f"checkpoint_{suffix}.pt"
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")


class LearningRateCallback(TrainingCallback):
    """Learning rate logging callback"""
    
    def __init__(self, log_every: int = 10):
        self.log_every = log_every
    
    def on_epoch_start(self, epoch: int, **kwargs):
        """Not used"""
        pass
    
    def on_epoch_end(self, epoch: int, metrics: Dict[str, Any], **kwargs):
        """Log learning rate"""
        optimizer = kwargs.get("optimizer")
        if optimizer is not None:
            lr = optimizer.param_groups[0]["lr"]
            logger.info(f"Epoch {epoch} - Learning Rate: {lr:.6f}")
    
    def on_batch_end(self, batch_idx: int, metrics: Dict[str, Any], **kwargs):
        """Log learning rate periodically"""
        if batch_idx % self.log_every == 0:
            optimizer = kwargs.get("optimizer")
            if optimizer is not None:
                lr = optimizer.param_groups[0]["lr"]
                logger.debug(f"Batch {batch_idx} - Learning Rate: {lr:.6f}")


class MetricsCallback(TrainingCallback):
    """Metrics logging callback"""
    
    def __init__(self, log_every: int = 10):
        self.log_every = log_every
    
    def on_epoch_start(self, epoch: int, **kwargs):
        """Not used"""
        pass
    
    def on_epoch_end(self, epoch: int, metrics: Dict[str, Any], **kwargs):
        """Log metrics"""
        metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        logger.info(f"Epoch {epoch} - {metrics_str}")
    
    def on_batch_end(self, batch_idx: int, metrics: Dict[str, Any], **kwargs):
        """Log metrics periodically"""
        if batch_idx % self.log_every == 0:
            metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
            logger.debug(f"Batch {batch_idx} - {metrics_str}")



