"""
Standard Training Loop
Standard single-GPU training loop implementation
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

from .base_loop import BaseTrainingLoop


class StandardTrainingLoop(BaseTrainingLoop):
    """
    Standard training loop for single GPU
    """
    
    def __init__(
        self,
        model,
        optimizer,
        loss_fn,
        device: str = "cuda",
        use_mixed_precision: bool = True,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0
    ):
        super().__init__(model, optimizer, loss_fn, device, use_mixed_precision)
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
    
    def train_step(
        self,
        batch: Dict[str, Any],
        batch_idx: int
    ) -> Dict[str, float]:
        """Execute one training step"""
        self.model.train()
        
        # Move batch to device
        batch = self._move_to_device(batch)
        
        # Forward pass
        if self.use_mixed_precision and self.scaler:
            with torch.cuda.amp.autocast():
                outputs = self.model(batch)
                loss = self.loss_fn(outputs, batch)
                loss = loss / self.gradient_accumulation_steps
        else:
            outputs = self.model(batch)
            loss = self.loss_fn(outputs, batch)
            loss = loss / self.gradient_accumulation_steps
        
        # Check for NaN/Inf
        if self._check_nan_inf(loss, "loss"):
            return {"loss": 0.0, "skipped": True}
        
        # Backward pass
        if self.scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Gradient accumulation
        if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
            # Check gradients
            has_nan_grad = False
            for param in self.model.parameters():
                if param.grad is not None:
                    if self._check_nan_inf(param.grad, f"grad_{param.shape}"):
                        has_nan_grad = True
                        param.grad.zero_()
            
            if not has_nan_grad:
                # Gradient clipping
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=self.max_grad_norm
                    )
                    if not self._check_nan_inf(grad_norm, "grad_norm"):
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=self.max_grad_norm
                    )
                    if not self._check_nan_inf(grad_norm, "grad_norm"):
                        self.optimizer.step()
                
                self.optimizer.zero_grad()
        
        return {
            "loss": loss.item() * self.gradient_accumulation_steps,
            "skipped": False
        }
    
    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """Execute one training epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        skipped_batches = 0
        
        for batch_idx, batch in enumerate(dataloader):
            try:
                metrics = self.train_step(batch, batch_idx)
                
                if not metrics.get("skipped", False):
                    total_loss += metrics["loss"]
                    num_batches += 1
                else:
                    skipped_batches += 1
            
            except Exception as e:
                logger.error(f"Error in batch {batch_idx}: {str(e)}")
                skipped_batches += 1
                self.optimizer.zero_grad()
                continue
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        return {
            "train_loss": avg_loss,
            "num_batches": num_batches,
            "skipped_batches": skipped_batches
        }



