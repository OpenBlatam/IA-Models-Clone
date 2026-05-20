"""
Standard Training Strategy
Single GPU training with standard optimizations
"""

from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

try:
    import torch
    from torch.utils.data import DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available")

from .base_strategy import BaseTrainingStrategy


class StandardTrainingStrategy(BaseTrainingStrategy):
    """
    Standard training strategy for single GPU
    """
    
    def __init__(
        self,
        model,
        optimizer,
        loss_fn,
        device: str = "cuda",
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0
    ):
        super().__init__(model, optimizer, loss_fn, device)
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
    
    def train_step(
        self,
        batch: Dict[str, Any],
        batch_idx: int
    ) -> Dict[str, float]:
        """Execute one training step"""
        self.model.train()
        batch = self._move_to_device(batch)
        
        # Forward pass
        outputs = self.model(batch)
        loss = self.loss_fn(outputs, batch)
        loss = loss / self.gradient_accumulation_steps
        
        # Backward pass
        loss.backward()
        
        # Gradient accumulation
        if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.max_grad_norm
            )
            self.optimizer.step()
            self.optimizer.zero_grad()
        
        return {"loss": loss.item() * self.gradient_accumulation_steps}
    
    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """Execute one training epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(dataloader):
            try:
                metrics = self.train_step(batch, batch_idx)
                total_loss += metrics["loss"]
                num_batches += 1
            except Exception as e:
                logger.error(f"Error in batch {batch_idx}: {str(e)}")
                self.optimizer.zero_grad()
                continue
        
        return {
            "train_loss": total_loss / num_batches if num_batches > 0 else 0.0
        }



