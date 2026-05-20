"""
Base Training Loop
Abstract base class for training loops
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


class BaseTrainingLoop(ABC):
    """
    Abstract base class for training loops
    Defines interface for all training loop implementations
    """
    
    def __init__(
        self,
        model,
        optimizer,
        loss_fn,
        device: str = "cuda",
        use_mixed_precision: bool = True
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.use_mixed_precision = use_mixed_precision and device == "cuda"
        
        # Move model to device
        self.model = self.model.to(device)
        
        # Mixed precision scaler
        if self.use_mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
    
    @abstractmethod
    def train_step(
        self,
        batch: Dict[str, Any],
        batch_idx: int
    ) -> Dict[str, float]:
        """
        Execute one training step
        
        Args:
            batch: Batch of data
            batch_idx: Batch index
        
        Returns:
            Dictionary of metrics for this step
        """
        pass
    
    @abstractmethod
    def train_epoch(
        self,
        dataloader,
        epoch: int
    ) -> Dict[str, float]:
        """
        Execute one training epoch
        
        Args:
            dataloader: DataLoader for training data
            epoch: Current epoch number
        
        Returns:
            Dictionary of epoch metrics
        """
        pass
    
    def validate_step(
        self,
        batch: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Execute one validation step
        
        Args:
            batch: Batch of data
        
        Returns:
            Dictionary of metrics for this step
        """
        self.model.eval()
        
        with torch.no_grad():
            # Move batch to device
            batch = self._move_to_device(batch)
            
            # Forward pass
            if self.use_mixed_precision:
                with torch.cuda.amp.autocast():
                    outputs = self.model(batch)
                    loss = self.loss_fn(outputs, batch)
            else:
                outputs = self.model(batch)
                loss = self.loss_fn(outputs, batch)
        
        return {
            "loss": loss.item(),
            "outputs": outputs
        }
    
    def validate_epoch(
        self,
        dataloader
    ) -> Dict[str, float]:
        """
        Execute validation epoch
        
        Args:
            dataloader: DataLoader for validation data
        
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                metrics = self.validate_step(batch)
                total_loss += metrics["loss"]
                num_batches += 1
        
        return {
            "val_loss": total_loss / num_batches if num_batches > 0 else 0.0
        }
    
    def _move_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move batch to device"""
        device_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                device_batch[key] = value.to(self.device, non_blocking=True)
            else:
                device_batch[key] = value
        return device_batch
    
    def _check_nan_inf(self, tensor: torch.Tensor, name: str = "tensor") -> bool:
        """Check for NaN/Inf values"""
        if torch.isnan(tensor).any():
            logger.warning(f"NaN detected in {name}")
            return True
        if torch.isinf(tensor).any():
            logger.warning(f"Inf detected in {name}")
            return True
        return False



