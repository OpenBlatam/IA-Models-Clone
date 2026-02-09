"""
Learning Rate Callback Module

Implements learning rate logging callback.
"""

from typing import Dict, Any
import logging

from .base import TrainingCallback

logger = logging.getLogger(__name__)


class LearningRateCallback(TrainingCallback):
    """
    Learning rate logging callback.
    
    Args:
        log_every: Log learning rate every N batches.
    """
    
    def __init__(self, log_every: int = 10):
        self.log_every = log_every
        logger.debug(f"Initialized LearningRateCallback with log_every={log_every}")
    
    def on_epoch_start(self, epoch: int, **kwargs):
        """Not used."""
        pass
    
    def on_epoch_end(self, epoch: int, metrics: Dict[str, Any], **kwargs):
        """Log learning rate."""
        optimizer = kwargs.get("optimizer")
        if optimizer is not None:
            lr = optimizer.param_groups[0]["lr"]
            logger.info(f"Epoch {epoch} - Learning Rate: {lr:.6f}")
    
    def on_batch_end(self, batch_idx: int, metrics: Dict[str, Any], **kwargs):
        """Log learning rate periodically."""
        if batch_idx % self.log_every == 0:
            optimizer = kwargs.get("optimizer")
            if optimizer is not None:
                lr = optimizer.param_groups[0]["lr"]
                logger.debug(f"Batch {batch_idx} - Learning Rate: {lr:.6f}")



