"""
Model Trainer
=============

Training utilities following PyTorch best practices.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from typing import Dict, Any, Optional, Callable
from pathlib import Path
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


class Trainer:
    """
    Trainer class following PyTorch best practices.
    
    Features:
    - Mixed precision training
    - Gradient clipping
    - Early stopping
    - Learning rate scheduling
    - Model checkpointing
    - Experiment tracking
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
        Initialize trainer.
        
        Args:
            model: PyTorch model
            train_loader: Training dataloader
            val_loader: Validation dataloader
            criterion: Loss function
            optimizer: Optimizer
            device: Device (CPU/GPU)
            config: Training configuration
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.config = config
        
        # Mixed precision
        self.use_amp = config.get("use_mixed_precision", False)
        self.scaler = GradScaler() if self.use_amp else None
        
        # Learning rate scheduler
        scheduler_type = config.get("scheduler_type", "step")
        if scheduler_type == "step":
            self.scheduler = optim.lr_scheduler.StepLR(
                optimizer,
                step_size=config.get("scheduler_step_size", 10),
                gamma=config.get("scheduler_gamma", 0.1)
            )
        elif scheduler_type == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=config.get("max_epochs", 100)
            )
        else:
            self.scheduler = None
        
        # Early stopping
        self.early_stopping_patience = config.get("early_stopping_patience", 10)
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # Checkpointing
        self.checkpoint_dir = Path(config.get("checkpoint_dir", "checkpoints"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Training history
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_metrics": [],
            "val_metrics": []
        }
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for batch_idx, (features, targets) in enumerate(pbar):
            features = features.to(self.device)
            targets = targets.to(self.device)
            
            # Ensure targets have correct shape
            if targets.dim() == 1:
                targets = targets.unsqueeze(1)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.use_amp:
                with autocast():
                    outputs = self.model(features)
                    loss = self.criterion(outputs, targets)
                
                # Backward pass with scaling
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.config.get("grad_clip", 0) > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config["grad_clip"]
                    )
                
                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(features)
                loss = self.criterion(outputs, targets)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                if self.config.get("grad_clip", 0) > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config["grad_clip"]
                    )
                
                # Optimizer step
                self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({"loss": loss.item()})
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return {"loss": avg_loss}
    
    def validate(self) -> Dict[str, float]:
        """Validate model."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for features, targets in tqdm(self.val_loader, desc="Validating"):
                features = features.to(self.device)
                targets = targets.to(self.device)
                
                if targets.dim() == 1:
                    targets = targets.unsqueeze(1)
                
                # Forward pass
                if self.use_amp:
                    with autocast():
                        outputs = self.model(features)
                        loss = self.criterion(outputs, targets)
                else:
                    outputs = self.model(features)
                    loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return {"loss": avg_loss}
    
    def train(self, num_epochs: int) -> Dict[str, Any]:
        """
        Train model.
        
        Args:
            num_epochs: Number of epochs
        
        Returns:
            Training history
        """
        logger.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
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
            
            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                # Save best model
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
        
        return self.history
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch
            is_best: Whether this is the best model
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "history": self.history
        }
        
        if self.scheduler:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model with val_loss: {self.best_val_loss:.4f}")
    
    def load_checkpoint(self, checkpoint_path: Path):
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        if self.scheduler and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        self.best_val_loss = checkpoint.get("best_val_loss", float('inf'))
        self.history = checkpoint.get("history", self.history)
        
        logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch'] + 1}")




