"""
Base Training Strategy
Abstract base for different training strategies
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

try:
    import torch
    from torch.utils.data import DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available")


class BaseTrainingStrategy(ABC):
    """
    Abstract base class for training strategies
    Different strategies for different training scenarios
    """
    
    def __init__(
        self,
        model,
        optimizer,
        loss_fn,
        device: str = "cuda"
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        
        # Move model to device
        self.model = self.model.to(device)
    
    @abstractmethod
    def train_step(
        self,
        batch: Dict[str, Any],
        batch_idx: int
    ) -> Dict[str, float]:
        """Execute one training step"""
        pass
    
    @abstractmethod
    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """Execute one training epoch"""
        pass
    
    def validate_step(
        self,
        batch: Dict[str, Any]
    ) -> Dict[str, float]:
        """Execute one validation step"""
        self.model.eval()
        
        with torch.no_grad():
            batch = self._move_to_device(batch)
            outputs = self.model(batch)
            loss = self.loss_fn(outputs, batch)
        
        return {"loss": loss.item()}
    
    def validate_epoch(
        self,
        dataloader: DataLoader
    ) -> Dict[str, float]:
        """Execute validation epoch"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                metrics = self.validate_step(batch)
                total_loss += metrics["loss"]
                num_batches += 1
        
        return {"val_loss": total_loss / num_batches if num_batches > 0 else 0.0}
    
    def _move_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move batch to device"""
        device_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                device_batch[key] = value.to(self.device, non_blocking=True)
            else:
                device_batch[key] = value
        return device_batch



