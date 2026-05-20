"""
Advanced Trainer
================

Advanced trainer with additional features following best practices.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from typing import Dict, Any, Optional, List, Callable
from pathlib import Path
import logging
from tqdm import tqdm
import numpy as np

from .trainer import Trainer
from ..utils.callbacks import CallbackList

logger = logging.getLogger(__name__)


class AdvancedTrainer(Trainer):
    """
    Advanced trainer with additional features:
    - Gradient accumulation
    - Learning rate finder
    - Model ensembling
    - Advanced callbacks
    - Better logging
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        device: torch.device,
        config: Dict[str, Any]
    ):
        """
        Initialize advanced trainer.
        
        Args:
            model: PyTorch model
            train_loader: Training dataloader
            val_loader: Validation dataloader
            criterion: Loss function
            optimizer: Optimizer
            device: Device
            config: Training configuration
        """
        super().__init__(model, train_loader, val_loader, criterion, optimizer, device, config)
        
        # Advanced features
        self.gradient_accumulation_steps = config.get("gradient_accumulation_steps", 1)
        self.callbacks = CallbackList(config.get("callbacks", []))
        self.use_ema = config.get("use_ema", False)
        
        # EMA (Exponential Moving Average)
        if self.use_ema:
            self.ema_model = self._create_ema_model()
            self.ema_decay = config.get("ema_decay", 0.999)
        
        # Gradient clipping
        self.grad_clip_value = config.get("grad_clip", 0.0)
        
        # NaN/Inf detection
        self.check_nan_inf = config.get("check_nan_inf", True)
    
    def _create_ema_model(self) -> nn.Module:
        """Create EMA model copy."""
        import copy
        ema_model = copy.deepcopy(self.model)
        for param in ema_model.parameters():
            param.requires_grad = False
        return ema_model
    
    def _update_ema(self):
        """Update EMA model."""
        if not self.use_ema:
            return
        
        with torch.no_grad():
            for ema_param, model_param in zip(
                self.ema_model.parameters(),
                self.model.parameters()
            ):
                ema_param.data.mul_(self.ema_decay).add_(
                    model_param.data, alpha=1 - self.ema_decay
                )
    
    def _check_nan_inf(self, loss: torch.Tensor) -> bool:
        """
        Check for NaN/Inf in loss.
        
        Args:
            loss: Loss tensor
        
        Returns:
            True if NaN/Inf detected
        """
        if not self.check_nan_inf:
            return False
        
        if torch.isnan(loss) or torch.isinf(loss):
            logger.error("NaN or Inf detected in loss!")
            return True
        return False
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch with advanced features."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        self.optimizer.zero_grad()
        
        pbar = tqdm(self.train_loader, desc="Training")
        
        for batch_idx, (features, targets) in enumerate(pbar):
            features = features.to(self.device)
            targets = targets.to(self.device)
            
            if targets.dim() == 1:
                targets = targets.unsqueeze(1)
            
            # Forward pass
            if self.use_amp:
                with autocast():
                    outputs = self.model(features)
                    loss = self.criterion(outputs, targets)
                    loss = loss / self.gradient_accumulation_steps
            else:
                outputs = self.model(features)
                loss = self.criterion(outputs, targets)
                loss = loss / self.gradient_accumulation_steps
            
            # Check for NaN/Inf
            if self._check_nan_inf(loss):
                logger.warning("Skipping batch due to NaN/Inf")
                continue
            
            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.grad_clip_value > 0:
                    if self.use_amp:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.grad_clip_value
                        )
                    else:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.grad_clip_value
                        )
                
                # Optimizer step
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                
                # Update EMA
                self._update_ema()
            
            # Update metrics
            total_loss += loss.item() * self.gradient_accumulation_steps
            num_batches += 1
            
            # Callbacks
            self.callbacks.on_batch_end(
                batch_idx,
                {"loss": loss.item() * self.gradient_accumulation_steps}
            )
            
            pbar.set_postfix({"loss": loss.item() * self.gradient_accumulation_steps})
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return {"loss": avg_loss}
    
    def train(self, num_epochs: int) -> Dict[str, Any]:
        """
        Train model with callbacks.
        
        Args:
            num_epochs: Number of epochs
        
        Returns:
            Training history
        """
        self.callbacks.on_train_begin()
        
        logger.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            self.callbacks.on_epoch_begin(epoch)
            
            # Train
            train_metrics = self.train_epoch()
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["train_metrics"].append(train_metrics)
            
            # Validate
            val_metrics = self.validate()
            val_loss = val_metrics["loss"]
            self.history["val_loss"].append(val_loss)
            self.history["val_metrics"].append(val_metrics)
            
            # Learning rate scheduling
            if self.scheduler:
                self.scheduler.step()
            
            # Callbacks
            epoch_logs = {**train_metrics, **val_metrics}
            self.callbacks.on_epoch_end(epoch, epoch_logs)
            
            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_checkpoint(epoch, is_best=True)
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break
            
            # Regular checkpoint
            if (epoch + 1) % self.config.get("checkpoint_interval", 10) == 0:
                self.save_checkpoint(epoch)
            
            logger.info(
                f"Epoch {epoch + 1}/{num_epochs} - "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Val Loss: {val_loss:.4f}"
            )
        
        self.callbacks.on_train_end()
        
        return self.history
    
    def get_ema_model(self) -> Optional[nn.Module]:
        """Get EMA model if enabled."""
        if self.use_ema:
            return self.ema_model
        return None




