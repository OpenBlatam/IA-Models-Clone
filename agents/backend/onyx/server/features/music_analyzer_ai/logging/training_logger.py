"""
Training Logger
Specialized logger for training
"""

from typing import Dict, Any, Optional
import logging
from datetime import datetime


class TrainingLogger:
    """Logger for training progress"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.start_time = datetime.now()
    
    def log_epoch_start(self, epoch: int, total_epochs: int):
        """Log epoch start"""
        self.logger.info(f"Epoch {epoch}/{total_epochs} started")
    
    def log_epoch_end(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Optional[Dict[str, float]] = None
    ):
        """Log epoch end"""
        self.logger.info(f"Epoch {epoch} completed")
        self.logger.info(f"Train metrics: {train_metrics}")
        if val_metrics:
            self.logger.info(f"Val metrics: {val_metrics}")
    
    def log_batch(
        self,
        epoch: int,
        batch_idx: int,
        metrics: Dict[str, float]
    ):
        """Log batch progress"""
        self.logger.debug(
            f"Epoch {epoch}, Batch {batch_idx}: {metrics}"
        )
    
    def log_checkpoint(self, checkpoint_path: str):
        """Log checkpoint save"""
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def log_best_model(self, epoch: int, metric: str, value: float):
        """Log best model"""
        self.logger.info(
            f"New best model at epoch {epoch}: {metric} = {value}"
        )
    
    def log_training_complete(self, total_time: float):
        """Log training completion"""
        self.logger.info(f"Training completed in {total_time:.2f} seconds")



