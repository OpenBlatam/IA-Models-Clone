"""
Mixed Precision Training Strategy
FP16 training with automatic mixed precision
"""

from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

try:
    import torch
    from torch.utils.data import DataLoader
    from torch.cuda.amp import autocast, GradScaler
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available")

from .standard_strategy import StandardTrainingStrategy


class MixedPrecisionStrategy(StandardTrainingStrategy):
    """
    Training strategy with mixed precision (FP16)
    """
    
    def __init__(
        self,
        model,
        optimizer,
        loss_fn,
        device: str = "cuda",
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        init_scale: float = 2.**16,
        growth_factor: float = 2.0
    ):
        super().__init__(
            model, optimizer, loss_fn, device,
            gradient_accumulation_steps, max_grad_norm
        )
        
        # Mixed precision scaler
        if device == "cuda":
            self.scaler = GradScaler(
                init_scale=init_scale,
                growth_factor=growth_factor
            )
        else:
            self.scaler = None
    
    def train_step(
        self,
        batch: Dict[str, Any],
        batch_idx: int
    ) -> Dict[str, float]:
        """Execute one training step with mixed precision"""
        self.model.train()
        batch = self._move_to_device(batch)
        
        # Forward pass with autocast
        if self.scaler:
            with autocast():
                outputs = self.model(batch)
                loss = self.loss_fn(outputs, batch)
                loss = loss / self.gradient_accumulation_steps
            
            # Backward pass with scaler
            self.scaler.scale(loss).backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                # Unscale gradients
                self.scaler.unscale_(self.optimizer)
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.max_grad_norm
                )
                
                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
        else:
            # Fallback to standard training
            return super().train_step(batch, batch_idx)
        
        return {"loss": loss.item() * self.gradient_accumulation_steps}



