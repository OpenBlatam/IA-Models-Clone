"""
Training Module
Implements training with proper error handling, gradient clipping, and mixed precision.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any, Callable
from tqdm import tqdm
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class Trainer:
    """
    Trainer class following best practices for deep learning training.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        criterion: Optional[nn.Module] = None,
        device: Optional[str] = None,
        use_amp: bool = True,
        max_grad_norm: float = 1.0,
        gradient_accumulation_steps: int = 1,
        log_interval: int = 10,
        save_dir: Optional[str] = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_amp = use_amp and self.device == "cuda"
        self.max_grad_norm = max_grad_norm
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.log_interval = log_interval
        self.save_dir = Path(save_dir) if save_dir else None
        
        # Move model to device
        self.model.to(self.device)
        
        # Setup optimizer
        if optimizer is None:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=5e-5,
                weight_decay=0.01,
            )
        else:
            self.optimizer = optimizer
        
        # Setup loss function
        if criterion is None:
            self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        else:
            self.criterion = criterion
        
        # Setup mixed precision scaler
        self.scaler = None
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")
        
        # Enable anomaly detection in debug mode
        if logger.level == logging.DEBUG:
            torch.autograd.set_detect_anomaly(True)
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch + 1}",
            leave=False,
        )
        
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = self._move_to_device(batch)
            
            # Forward pass with mixed precision
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(**batch)
                    loss = self._compute_loss(outputs, batch)
                    loss = loss / self.gradient_accumulation_steps
            else:
                outputs = self.model(**batch)
                loss = self._compute_loss(outputs, batch)
                loss = loss / self.gradient_accumulation_steps
            
            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.max_grad_norm
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.max_grad_norm
                    )
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                self.global_step += 1
            
            # Update metrics
            total_loss += loss.item() * self.gradient_accumulation_steps
            num_batches += 1
            
            # Logging
            if self.global_step % self.log_interval == 0:
                avg_loss = total_loss / num_batches
                progress_bar.set_postfix({"loss": f"{avg_loss:.4f}"})
                logger.info(
                    f"Step {self.global_step}: loss={avg_loss:.4f}",
                    extra={"step": self.global_step, "loss": avg_loss},
                )
            
            # Check for NaN/Inf
            if torch.isnan(loss) or torch.isinf(loss):
                logger.error(f"NaN/Inf loss detected at step {self.global_step}")
                raise ValueError("Training diverged: NaN/Inf loss")
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return {"loss": avg_loss, "steps": self.global_step}
    
    def validate(self) -> Dict[str, float]:
        """
        Validate model.
        
        Returns:
            Dictionary with validation metrics
        """
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation", leave=False):
                batch = self._move_to_device(batch)
                
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(**batch)
                        loss = self._compute_loss(outputs, batch)
                else:
                    outputs = self.model(**batch)
                    loss = self._compute_loss(outputs, batch)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return {"val_loss": avg_loss}
    
    def _compute_loss(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute loss from model outputs."""
        if "loss" in outputs:
            return outputs["loss"]
        
        if "logits" in outputs and "labels" in batch:
            logits = outputs["logits"]
            labels = batch["labels"]
            # Reshape for cross entropy
            if len(logits.shape) == 3:
                logits = logits.view(-1, logits.size(-1))
                labels = labels.view(-1)
            return self.criterion(logits, labels)
        
        raise ValueError("Cannot compute loss: missing logits or labels")
    
    def _move_to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Move batch to device."""
        return {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
    
    def train(self, num_epochs: int, save_best: bool = True):
        """
        Train model for specified number of epochs.
        
        Args:
            num_epochs: Number of epochs to train
            save_best: Whether to save best model based on validation loss
        """
        logger.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Train epoch
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Log metrics
            metrics = {**train_metrics, **val_metrics}
            logger.info(f"Epoch {epoch + 1}/{num_epochs}: {metrics}")
            
            # Save checkpoint
            if self.save_dir:
                self.save_checkpoint(epoch, metrics)
            
            # Save best model
            if save_best and "val_loss" in val_metrics:
                if val_metrics["val_loss"] < self.best_val_loss:
                    self.best_val_loss = val_metrics["val_loss"]
                    if self.save_dir:
                        self.save_checkpoint(epoch, metrics, is_best=True)
        
        logger.info("Training completed")
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """Save model checkpoint."""
        if self.save_dir is None:
            return
        
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
        }
        
        if self.scaler:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()
        
        # Save regular checkpoint
        checkpoint_path = self.save_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.save_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model to {best_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        if "scaler_state_dict" in checkpoint and self.scaler:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        
        self.current_epoch = checkpoint.get("epoch", 0)
        self.best_val_loss = checkpoint.get("metrics", {}).get("val_loss", float("inf"))
        
        logger.info(f"Loaded checkpoint from {checkpoint_path}")



