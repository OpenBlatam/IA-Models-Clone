"""
Early Stopping Callback Module

Implements early stopping callback.
"""

from typing import Dict, Any
import logging

from .base import TrainingCallback

logger = logging.getLogger(__name__)


class EarlyStoppingCallback(TrainingCallback):
    """
    Early stopping callback.
    
    Args:
        patience: Number of epochs to wait before stopping.
        min_delta: Minimum change to qualify as improvement.
        monitor: Metric to monitor.
        mode: "min" or "max".
    """
    
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
        logger.debug(f"Initialized EarlyStoppingCallback with patience={patience}, monitor='{monitor}'")
    
    def on_epoch_start(self, epoch: int, **kwargs):
        """Reset at epoch start."""
        pass
    
    def on_epoch_end(self, epoch: int, metrics: Dict[str, Any], **kwargs):
        """Check for early stopping."""
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
        """Not used for early stopping."""
        pass



