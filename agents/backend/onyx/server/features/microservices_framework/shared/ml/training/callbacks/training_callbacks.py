"""
Training Callbacks
Specialized callbacks for training operations.
"""

from typing import Dict, Any, Optional
import torch
import logging
from ...utils.callbacks import Callback

logger = logging.getLogger(__name__)


class GradientMonitorCallback(Callback):
    """Monitor gradients during training."""
    
    def __init__(self, log_interval: int = 100):
        self.log_interval = log_interval
        self.gradient_norms = []
    
    def on_batch_end(self, batch_idx: int, loss: float, model: torch.nn.Module, **kwargs):
        """Monitor gradients."""
        if batch_idx % self.log_interval == 0:
            total_norm = 0.0
            param_count = 0
            
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                    param_count += 1
            
            total_norm = total_norm ** (1. / 2)
            self.gradient_norms.append(total_norm)
            
            logger.info(f"Gradient norm: {total_norm:.4f}")
    
    def on_epoch_start(self, epoch: int, **kwargs):
        pass
    
    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], **kwargs):
        pass
    
    def on_training_start(self, **kwargs):
        pass
    
    def on_training_end(self, **kwargs):
        pass


class LearningRateMonitorCallback(Callback):
    """Monitor learning rate during training."""
    
    def __init__(self, optimizer: torch.optim.Optimizer):
        self.optimizer = optimizer
    
    def on_batch_end(self, batch_idx: int, loss: float, **kwargs):
        """Log learning rate."""
        if batch_idx % 100 == 0:
            lr = self.optimizer.param_groups[0]['lr']
            logger.info(f"Learning rate: {lr:.2e}")
    
    def on_epoch_start(self, epoch: int, **kwargs):
        pass
    
    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], **kwargs):
        pass
    
    def on_training_start(self, **kwargs):
        pass
    
    def on_training_end(self, **kwargs):
        pass


class ModelCheckpointCallback(Callback):
    """Save model checkpoints."""
    
    def __init__(
        self,
        save_dir: str,
        save_frequency: int = 1,
        save_best: bool = True,
        monitor: str = "val_loss",
    ):
        self.save_dir = save_dir
        self.save_frequency = save_frequency
        self.save_best = save_best
        self.monitor = monitor
        self.best_score = None
        
        from pathlib import Path
        Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    def on_epoch_end(
        self,
        epoch: int,
        metrics: Dict[str, float],
        model: torch.nn.Module,
        **kwargs
    ):
        """Save checkpoint."""
        if epoch % self.save_frequency == 0:
            checkpoint_path = f"{self.save_dir}/checkpoint_epoch_{epoch}.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "metrics": metrics,
            }, checkpoint_path)
            logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        if self.save_best and self.monitor in metrics:
            current_score = metrics[self.monitor]
            if self.best_score is None or current_score < self.best_score:
                self.best_score = current_score
                best_path = f"{self.save_dir}/best_model.pt"
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "metrics": metrics,
                }, best_path)
                logger.info(f"Best model saved: {best_path}")
    
    def on_epoch_start(self, epoch: int, **kwargs):
        pass
    
    def on_training_start(self, **kwargs):
        pass
    
    def on_training_end(self, **kwargs):
        pass



