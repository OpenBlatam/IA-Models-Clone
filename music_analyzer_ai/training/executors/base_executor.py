"""
Base Training Executor

Separates training execution logic from strategy, providing
clean separation of concerns.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import logging
import torch
from torch.utils.data import DataLoader

from ...interfaces.base import ITrainingStrategy, ITrainingCallback

logger = logging.getLogger(__name__)


class BaseTrainingExecutor(ABC):
    """
    Abstract base for training executors.
    
    Executors handle the orchestration of training loops,
    while strategies handle the actual training logic.
    """
    
    def __init__(
        self,
        strategy: ITrainingStrategy,
        callbacks: Optional[List[ITrainingCallback]] = None
    ):
        """
        Initialize training executor.
        
        Args:
            strategy: Training strategy
            callbacks: List of training callbacks
        """
        self.strategy = strategy
        self.callbacks = callbacks or []
        self.current_epoch = 0
        self.current_step = 0
    
    @abstractmethod
    def train(
        self,
        train_loader: DataLoader,
        num_epochs: int,
        val_loader: Optional[DataLoader] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute training.
        
        Args:
            train_loader: Training data loader
            num_epochs: Number of epochs
            val_loader: Optional validation data loader
            **kwargs: Additional arguments
            
        Returns:
            Training history and metrics
        """
        pass
    
    def _call_callbacks(self, event: str, **kwargs):
        """Call all callbacks for an event."""
        for callback in self.callbacks:
            try:
                if event == "epoch_start":
                    callback.on_epoch_start(kwargs.get("epoch", 0))
                elif event == "epoch_end":
                    callback.on_epoch_end(
                        kwargs.get("epoch", 0),
                        kwargs.get("metrics", {})
                    )
                elif event == "batch_start":
                    callback.on_batch_start(kwargs.get("batch_idx", 0))
                elif event == "batch_end":
                    callback.on_batch_end(
                        kwargs.get("batch_idx", 0),
                        kwargs.get("metrics", {})
                    )
            except Exception as e:
                logger.warning(f"Callback {callback} failed: {e}")


class StandardTrainingExecutor(BaseTrainingExecutor):
    """
    Standard training executor with epoch and batch management.
    """
    
    def train(
        self,
        train_loader: DataLoader,
        num_epochs: int,
        val_loader: Optional[DataLoader] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute standard training loop.
        
        Args:
            train_loader: Training data loader
            num_epochs: Number of epochs
            val_loader: Optional validation data loader
            **kwargs: Additional arguments
            
        Returns:
            Training history
        """
        history = {
            "train_loss": [],
            "val_loss": [],
            "train_metrics": [],
            "val_metrics": []
        }
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Epoch start callbacks
            self._call_callbacks("epoch_start", epoch=epoch)
            
            # Training epoch
            train_metrics = self.strategy.train_epoch(train_loader, epoch)
            history["train_loss"].append(train_metrics.get("loss", 0.0))
            history["train_metrics"].append(train_metrics)
            
            # Validation epoch
            val_metrics = {}
            if val_loader is not None:
                val_metrics = self._validate_epoch(val_loader)
                history["val_loss"].append(val_metrics.get("loss", 0.0))
                history["val_metrics"].append(val_metrics)
            
            # Epoch end callbacks
            epoch_metrics = {**train_metrics, **val_metrics}
            self._call_callbacks("epoch_end", epoch=epoch, metrics=epoch_metrics)
            
            logger.info(
                f"Epoch {epoch+1}/{num_epochs} - "
                f"Train Loss: {train_metrics.get('loss', 0.0):.4f} - "
                f"Val Loss: {val_metrics.get('loss', 0.0):.4f}"
            )
        
        return history
    
    def _validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """Execute validation epoch."""
        all_metrics = []
        
        for batch_idx, batch in enumerate(val_loader):
            metrics = self.strategy.validate_step(batch)
            all_metrics.append(metrics)
        
        # Aggregate metrics
        aggregated = {}
        for key in all_metrics[0].keys():
            aggregated[key] = sum(m[key] for m in all_metrics) / len(all_metrics)
        
        return aggregated


__all__ = [
    "BaseTrainingExecutor",
    "StandardTrainingExecutor",
]



