"""
Training Logger

Specialized logger for training progress.
"""

import logging
from typing import Dict, Any, Optional
from .structured_logger import StructuredLogger

logger = logging.getLogger(__name__)


class TrainingLogger:
    """Logger specialized for training."""
    
    def __init__(
        self,
        log_dir: Optional[str] = None,
        use_json: bool = False
    ):
        """
        Initialize training logger.
        
        Args:
            log_dir: Log directory
            use_json: Use JSON format
        """
        self.logger = StructuredLogger(
            "training",
            log_dir=log_dir,
            use_json=use_json
        )
    
    def log_step(
        self,
        step: int,
        epoch: int,
        loss: float,
        metrics: Optional[Dict[str, float]] = None,
        **kwargs
    ) -> None:
        """
        Log training step.
        
        Args:
            step: Step number
            epoch: Epoch number
            loss: Loss value
            metrics: Additional metrics
            **kwargs: Additional context
        """
        log_data = {
            'step': step,
            'epoch': epoch,
            'loss': loss,
            **(metrics or {}),
            **kwargs
        }
        
        self.logger.info(
            f"Step {step} (Epoch {epoch}): Loss={loss:.4f}",
            **log_data
        )
    
    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: Optional[float] = None,
        metrics: Optional[Dict[str, float]] = None,
        **kwargs
    ) -> None:
        """
        Log epoch summary.
        
        Args:
            epoch: Epoch number
            train_loss: Training loss
            val_loss: Validation loss
            metrics: Additional metrics
            **kwargs: Additional context
        """
        log_data = {
            'epoch': epoch,
            'train_loss': train_loss,
            **(metrics or {}),
            **kwargs
        }
        
        if val_loss is not None:
            log_data['val_loss'] = val_loss
            message = f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}"
        else:
            message = f"Epoch {epoch}: Train Loss={train_loss:.4f}"
        
        self.logger.info(message, **log_data)
    
    def log_checkpoint(
        self,
        epoch: int,
        checkpoint_path: str,
        metrics: Optional[Dict[str, float]] = None
    ) -> None:
        """
        Log checkpoint save.
        
        Args:
            epoch: Epoch number
            checkpoint_path: Path to checkpoint
            metrics: Metrics at checkpoint
        """
        log_data = {
            'epoch': epoch,
            'checkpoint_path': checkpoint_path,
            **(metrics or {})
        }
        
        self.logger.info(
            f"Checkpoint saved at epoch {epoch}: {checkpoint_path}",
            **log_data
        )


def log_training_step(
    step: int,
    epoch: int,
    loss: float,
    **kwargs
) -> None:
    """Convenience function to log training step."""
    logger = TrainingLogger()
    logger.log_step(step, epoch, loss, **kwargs)


def log_epoch_summary(
    epoch: int,
    train_loss: float,
    val_loss: Optional[float] = None,
    **kwargs
) -> None:
    """Convenience function to log epoch summary."""
    logger = TrainingLogger()
    logger.log_epoch(epoch, train_loss, val_loss, **kwargs)



