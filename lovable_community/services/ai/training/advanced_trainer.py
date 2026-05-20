"""
Advanced Trainer with Best Practices

Following best practices for deep learning training:
- Proper learning rate scheduling
- Gradient accumulation
- Mixed precision
- Gradient clipping
- Early stopping
- Checkpointing
- Metrics tracking
"""

import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.cuda.amp import GradScaler, autocast
from typing import Optional, Dict, Any, List, Callable
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


class AdvancedTrainer:
    """
    Advanced trainer with all best practices.
    
    Features:
    - Learning rate scheduling
    - Gradient accumulation
    - Mixed precision
    - Gradient clipping
    - Early stopping
    - Comprehensive metrics
    - Model checkpointing
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        criterion: nn.Module,
        device: torch.device,
        scheduler: Optional[_LRScheduler] = None,
        use_mixed_precision: bool = True,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        clip_gradients: bool = True,
        save_dir: Optional[Path] = None,
        save_best: bool = True,
        save_last: bool = True,
        save_every_n_epochs: Optional[int] = None
    ):
        """
        Initialize advanced trainer.
        
        Args:
            model: PyTorch model
            optimizer: Optimizer
            criterion: Loss function
            device: Device to train on
            scheduler: Learning rate scheduler
            use_mixed_precision: Use mixed precision training
            gradient_accumulation_steps: Steps for gradient accumulation
            max_grad_norm: Maximum gradient norm for clipping
            clip_gradients: Whether to clip gradients
            save_dir: Directory to save checkpoints
            save_best: Save best model
            save_last: Save last model
            save_every_n_epochs: Save every N epochs
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scheduler = scheduler
        self.use_mixed_precision = use_mixed_precision and device.type == "cuda"
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.clip_gradients = clip_gradients
        self.save_dir = Path(save_dir) if save_dir else None
        self.save_best = save_best
        self.save_last = save_last
        self.save_every_n_epochs = save_every_n_epochs
        
        # Mixed precision scaler
        self.scaler = GradScaler() if self.use_mixed_precision else None
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = float('inf')
        self.train_metrics_history = []
        self.val_metrics_history = []
        
        # Create save directory
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(
            f"Advanced trainer initialized: mixed_precision={self.use_mixed_precision}, "
            f"gradient_accumulation={gradient_accumulation_steps}, "
            f"max_grad_norm={max_grad_norm}"
        )
    
    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int,
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, float]:
        """
        Train for one epoch with all optimizations.
        
        Args:
            dataloader: Training dataloader
            epoch: Current epoch
            progress_callback: Optional callback for progress
            
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        self.current_epoch = epoch
        
        total_loss = 0.0
        num_batches = 0
        all_predictions = []
        all_targets = []
        
        # Reset optimizer
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(dataloader):
            # Move batch to device
            batch = self._move_to_device(batch)
            
            # Forward pass with mixed precision
            with autocast(enabled=self.use_mixed_precision):
                outputs = self._forward_pass(batch)
                loss = self._compute_loss(outputs, batch)
                
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
                if self.clip_gradients:
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
            
            # Accumulate metrics
            total_loss += loss.item() * self.gradient_accumulation_steps
            num_batches += 1
            
            # Collect predictions and targets for metrics
            if hasattr(batch, 'targets'):
                all_predictions.append(outputs.detach().cpu())
                all_targets.append(batch['targets'].detach().cpu())
            
            # Progress callback
            if progress_callback:
                progress_callback(batch_idx, len(dataloader), loss.item())
        
        # Compute metrics
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        metrics = {
            "loss": avg_loss,
            "learning_rate": self.optimizer.param_groups[0]['lr']
        }
        
        # Add additional metrics if available
        if all_predictions and all_targets:
            predictions = torch.cat(all_predictions)
            targets = torch.cat(all_targets)
            metrics.update(self._compute_metrics(predictions, targets))
        
        self.train_metrics_history.append(metrics)
        
        return metrics
    
    def validate(
        self,
        dataloader: DataLoader
    ) -> Dict[str, float]:
        """
        Validate model.
        
        Args:
            dataloader: Validation dataloader
            
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        num_batches = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in dataloader:
                batch = self._move_to_device(batch)
                
                with autocast(enabled=self.use_mixed_precision):
                    outputs = self._forward_pass(batch)
                    loss = self._compute_loss(outputs, batch)
                
                total_loss += loss.item()
                num_batches += 1
                
                if hasattr(batch, 'targets'):
                    all_predictions.append(outputs.detach().cpu())
                    all_targets.append(batch['targets'].detach().cpu())
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        metrics = {"loss": avg_loss}
        
        if all_predictions and all_targets:
            predictions = torch.cat(all_predictions)
            targets = torch.cat(all_targets)
            metrics.update(self._compute_metrics(predictions, targets))
        
        self.val_metrics_history.append(metrics)
        
        return metrics
    
    def _forward_pass(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Forward pass (to be implemented by subclasses)."""
        return self.model(batch['input'])
    
    def _compute_loss(
        self,
        outputs: torch.Tensor,
        batch: Dict[str, Any]
    ) -> torch.Tensor:
        """Compute loss."""
        return self.criterion(outputs, batch['targets'])
    
    def _compute_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> Dict[str, float]:
        """Compute additional metrics."""
        # Default: accuracy for classification
        if predictions.dim() > 1 and predictions.size(1) > 1:
            pred_classes = predictions.argmax(dim=1)
            accuracy = (pred_classes == targets).float().mean().item()
            return {"accuracy": accuracy}
        return {}
    
    def _move_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move batch to device."""
        return {
            k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
    
    def save_checkpoint(
        self,
        filename: str,
        is_best: bool = False,
        additional_info: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Save training checkpoint.
        
        Args:
            filename: Checkpoint filename
            is_best: Whether this is the best model
            additional_info: Additional information to save
        """
        if not self.save_dir:
            return
        
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_metric': self.best_metric,
            'train_metrics': self.train_metrics_history,
            'val_metrics': self.val_metrics_history,
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        if additional_info:
            checkpoint.update(additional_info)
        
        # Save checkpoint
        checkpoint_path = self.save_dir / filename
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")
        
        # Save as best if applicable
        if is_best and self.save_best:
            best_path = self.save_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"Best model saved to {best_path}")
    
    def load_checkpoint(self, checkpoint_path: Path) -> Dict[str, Any]:
        """
        Load training checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint
            
        Returns:
            Checkpoint dictionary
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if 'scaler_state_dict' in checkpoint and self.scaler:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.current_epoch = checkpoint.get('epoch', 0)
        self.global_step = checkpoint.get('global_step', 0)
        self.best_metric = checkpoint.get('best_metric', float('inf'))
        self.train_metrics_history = checkpoint.get('train_metrics', [])
        self.val_metrics_history = checkpoint.get('val_metrics', [])
        
        logger.info(f"Checkpoint loaded from {checkpoint_path}")
        
        return checkpoint
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: int = 10,
        early_stopping_patience: Optional[int] = None
    ) -> Dict[str, List[Dict[str, float]]]:
        """
        Full training loop.
        
        Args:
            train_loader: Training dataloader
            val_loader: Validation dataloader (optional)
            num_epochs: Number of epochs
            early_stopping_patience: Early stopping patience
            
        Returns:
            Dictionary with training history
        """
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(num_epochs):
            # Train
            train_metrics = self.train_epoch(train_loader, epoch)
            logger.info(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_metrics['loss']:.4f}")
            
            # Validate
            if val_loader:
                val_metrics = self.validate(val_loader)
                val_loss = val_metrics['loss']
                logger.info(f"Epoch {epoch+1}/{num_epochs} - Val Loss: {val_loss:.4f}")
                
                # Check if best
                is_best = val_loss < best_val_loss
                if is_best:
                    best_val_loss = val_loss
                    self.best_metric = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                # Save checkpoint
                if is_best:
                    self.save_checkpoint(f"epoch_{epoch+1}.pt", is_best=True)
                elif self.save_last:
                    self.save_checkpoint(f"last.pt", is_best=False)
                elif self.save_every_n_epochs and (epoch + 1) % self.save_every_n_epochs == 0:
                    self.save_checkpoint(f"epoch_{epoch+1}.pt", is_best=False)
                
                # Early stopping
                if early_stopping_patience and patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
            else:
                # No validation, just save periodically
                if self.save_every_n_epochs and (epoch + 1) % self.save_every_n_epochs == 0:
                    self.save_checkpoint(f"epoch_{epoch+1}.pt", is_best=False)
        
        return {
            "train": self.train_metrics_history,
            "val": self.val_metrics_history
        }













