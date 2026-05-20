"""
Improved Training Utilities

Following best practices for deep learning training:
- Proper data loading with DataLoader
- Mixed precision training
- Gradient accumulation
- Learning rate scheduling
- Early stopping
- Proper evaluation metrics
- Gradient clipping
"""

import logging
from typing import Optional, Dict, Any, List, Callable
import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.cuda.amp import GradScaler
import numpy as np

logger = logging.getLogger(__name__)


class ImprovedTrainer:
    """
    Improved trainer following best practices.
    
    Features:
    - Mixed precision training
    - Gradient accumulation
    - Learning rate scheduling
    - Early stopping
    - Gradient clipping
    - Comprehensive logging
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: Optimizer,
        criterion: Callable,
        device: torch.device,
        use_mixed_precision: bool = True,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: Optional[float] = 1.0,
        scheduler: Optional[_LRScheduler] = None
    ):
        """
        Initialize improved trainer.
        
        Args:
            model: PyTorch model
            optimizer: Optimizer
            criterion: Loss function
            device: Device to train on
            use_mixed_precision: Whether to use mixed precision
            gradient_accumulation_steps: Steps for gradient accumulation
            max_grad_norm: Maximum gradient norm for clipping
            scheduler: Learning rate scheduler
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.use_mixed_precision = use_mixed_precision and device.type == "cuda"
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.scheduler = scheduler
        
        # Mixed precision scaler
        self.scaler = GradScaler() if self.use_mixed_precision else None
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')
        
        logger.info(
            f"Trainer initialized: mixed_precision={self.use_mixed_precision}, "
            f"gradient_accumulation={gradient_accumulation_steps}, "
            f"max_grad_norm={max_grad_norm}"
        )
    
    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            dataloader: Training dataloader
            epoch: Current epoch number
            
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        self.current_epoch = epoch
        
        total_loss = 0.0
        num_batches = 0
        
        # Reset optimizer
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(dataloader):
            # Move batch to device
            batch = self._move_to_device(batch)
            
            # Forward pass with mixed precision
            with torch.cuda.amp.autocast(enabled=self.use_mixed_precision):
                loss = self._compute_loss(batch)
                
                # Scale loss for gradient accumulation
                loss = loss / self.gradient_accumulation_steps
            
            # Backward pass
            if self.use_mixed_precision:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Update weights every gradient_accumulation_steps
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.max_grad_norm is not None:
                    if self.use_mixed_precision:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.max_grad_norm
                    )
                
                # Optimizer step
                if self.use_mixed_precision:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                # Learning rate scheduler step
                if self.scheduler:
                    self.scheduler.step()
                
                # Reset gradients
                self.optimizer.zero_grad()
                
                self.global_step += 1
            
            total_loss += loss.item() * self.gradient_accumulation_steps
            num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        metrics = {
            "loss": avg_loss,
            "learning_rate": self.optimizer.param_groups[0]['lr']
        }
        
        return metrics
    
    def evaluate(
        self,
        dataloader: DataLoader
    ) -> Dict[str, float]:
        """
        Evaluate model on validation set.
        
        Args:
            dataloader: Validation dataloader
            
        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                batch = self._move_to_device(batch)
                
                with torch.cuda.amp.autocast(enabled=self.use_mixed_precision):
                    loss = self._compute_loss(batch)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        return {"loss": avg_loss}
    
    def _compute_loss(self, batch: Dict[str, Any]) -> torch.Tensor:
        """
        Compute loss for a batch.
        
        Args:
            batch: Batch of data
            
        Returns:
            Loss tensor
        """
        # This should be implemented by subclasses
        # Example:
        # outputs = self.model(batch['input'])
        # loss = self.criterion(outputs, batch['target'])
        # return loss
        raise NotImplementedError("Subclasses must implement _compute_loss")
    
    def _move_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Move batch to device.
        
        Args:
            batch: Batch of data
            
        Returns:
            Batch moved to device
        """
        if isinstance(batch, dict):
            return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                   for k, v in batch.items()}
        elif isinstance(batch, (list, tuple)):
            return type(batch)(v.to(self.device) if isinstance(v, torch.Tensor) else v 
                             for v in batch)
        elif isinstance(batch, torch.Tensor):
            return batch.to(self.device)
        else:
            return batch
    
    def save_checkpoint(
        self,
        path: str,
        additional_info: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Save training checkpoint.
        
        Args:
            path: Path to save checkpoint
            additional_info: Additional information to save
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_loss': self.best_loss,
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        if additional_info:
            checkpoint.update(additional_info)
        
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        """
        Load training checkpoint.
        
        Args:
            path: Path to checkpoint
            
        Returns:
            Checkpoint dictionary
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if 'scaler_state_dict' in checkpoint and self.scaler:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.current_epoch = checkpoint.get('epoch', 0)
        self.global_step = checkpoint.get('global_step', 0)
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        
        logger.info(f"Checkpoint loaded from {path}")
        
        return checkpoint


class EarlyStopping:
    """
    Early stopping callback.
    
    Stops training when validation loss stops improving.
    """
    
    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.0,
        mode: str = "min"
    ):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: "min" for loss, "max" for accuracy
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current metric score
            
        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = score
        elif self._is_better(score, self.best_score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop
    
    def _is_better(self, current: float, best: float) -> bool:
        """Check if current score is better than best."""
        if self.mode == "min":
            return current < (best - self.min_delta)
        else:
            return current > (best + self.min_delta)













