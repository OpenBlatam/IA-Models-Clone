"""
Base Trainer - Shared base implementation for all trainers
"""

from typing import Dict, Any, Optional
from torch.utils.data import DataLoader
import logging
import torch

from ..interfaces.trainer_interface import ITrainer, ITrainingCallback

logger = logging.getLogger(__name__)


class BaseTrainer(ITrainer):
    """
    Base class for all trainers with common functionality
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = model.to(device)
        self.device = device
        self.callbacks: list[ITrainingCallback] = []
        self.current_epoch = 0
        self.history: Dict[str, list] = {}
    
    def add_callback(self, callback: ITrainingCallback):
        """Add training callback"""
        self.callbacks.append(callback)
        logger.info(f"Added callback: {type(callback).__name__}")
    
    def remove_callback(self, callback: ITrainingCallback):
        """Remove training callback"""
        if callback in self.callbacks:
            self.callbacks.remove(callback)
    
    def _notify_callbacks(self, event: str, **kwargs):
        """Notify all callbacks of an event"""
        for callback in self.callbacks:
            try:
                if event == "epoch_start":
                    callback.on_epoch_start(kwargs.get("epoch", 0))
                elif event == "epoch_end":
                    callback.on_epoch_end(
                        kwargs.get("epoch", 0),
                        kwargs.get("metrics", {})
                    )
                elif event == "batch_end":
                    callback.on_batch_end(
                        kwargs.get("batch_idx", 0),
                        kwargs.get("loss", 0.0)
                    )
            except Exception as e:
                logger.warning(f"Callback error: {str(e)}")
    
    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        """Base train epoch implementation"""
        self.current_epoch = epoch
        self._notify_callbacks("epoch_start", epoch=epoch)
        
        # Subclasses should implement actual training logic
        metrics = {"loss": 0.0}
        
        self._notify_callbacks("epoch_end", epoch=epoch, metrics=metrics)
        return metrics
    
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Base evaluate implementation"""
        # Subclasses should implement actual evaluation logic
        return {"loss": 0.0}
    
    def save_checkpoint(self, path: str, epoch: int, metrics: Dict[str, float]) -> None:
        """Base save checkpoint implementation"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "metrics": metrics
        }
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str) -> tuple:
        """Base load checkpoint implementation"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        epoch = checkpoint.get("epoch", 0)
        metrics = checkpoint.get("metrics", {})
        logger.info(f"Checkpoint loaded from {path}")
        return epoch, metrics













