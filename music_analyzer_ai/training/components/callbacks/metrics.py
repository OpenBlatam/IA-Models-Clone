"""
Metrics Callback Module

Implements metrics logging callback.
"""

from typing import Dict, Any
import logging

from .base import TrainingCallback

logger = logging.getLogger(__name__)


class MetricsCallback(TrainingCallback):
    """
    Metrics logging callback.
    
    Args:
        log_every: Log metrics every N batches.
    """
    
    def __init__(self, log_every: int = 10):
        self.log_every = log_every
        logger.debug(f"Initialized MetricsCallback with log_every={log_every}")
    
    def on_epoch_start(self, epoch: int, **kwargs):
        """Not used."""
        pass
    
    def on_epoch_end(self, epoch: int, metrics: Dict[str, Any], **kwargs):
        """Log metrics."""
        metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        logger.info(f"Epoch {epoch} - {metrics_str}")
    
    def on_batch_end(self, batch_idx: int, metrics: Dict[str, Any], **kwargs):
        """Log metrics periodically."""
        if batch_idx % self.log_every == 0:
            metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
            logger.debug(f"Batch {batch_idx} - {metrics_str}")



