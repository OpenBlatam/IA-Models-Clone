"""
Training Pipeline for Quality Control AI Models
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from typing import Optional, Dict, List, Callable
import logging
from tqdm import tqdm
import os
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class ModelTrainer:
    """
    Trainer for quality control AI models with mixed precision, 
    gradient accumulation, and experiment tracking
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        device: Optional[torch.device] = None,
        use_mixed_precision: bool = True,
        gradient_accumulation_steps: int = 1,
        log_dir: str = "./logs",
        use_wandb: bool = False,
        wandb_project: str = "quality-control-ai"
    ):
        """
        Initialize trainer
        
        Args:
            model: PyTorch model to train
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            device: Device to train on
            use_mixed_precision: Whether to use mixed precision training
            gradient_accumulation_steps: Number of steps to accumulate gradients
            log_dir: Directory for TensorBoard logs
            use_wandb: Whether to use Weights & Biases
            wandb_project: W&B project name
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_mixed_precision = use_mixed_precision and torch.cuda.is_available()
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Mixed precision scaler
        self.scaler = GradScaler() if self.use_mixed_precision else None
        
        # TensorBoard writer
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)
        
        # W&B
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        if self.use_wandb:
            wandb.init(project=wandb_project, reinit=True)
        
        logger.info(f"ModelTrainer initialized on device: {self.device}")
        logger.info(f"Mixed precision: {self.use_mixed_precision}")
        logger.info(f"Gradient accumulation steps: {gradient_accumulation_steps}")
    
    def train_epoch(
        self,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        epoch: int,
        clip_grad_norm: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Train for one epoch
        
        Args:
            optimizer: Optimizer
            criterion: Loss function
            epoch: Current epoch number
            clip_grad_norm: Gradient clipping norm (optional)
            
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            if isinstance(batch, (list, tuple)):
                inputs = batch[0].to(self.device)
                targets = batch[1].to(self.device)
            else:
                inputs = batch.to(self.device)
                targets = None
            
            # Forward pass with mixed precision
            if self.use_mixed_precision:
                with autocast():
                    if targets is not None:
                        outputs = self.model(inputs)
                        loss = criterion(outputs, targets)
                    else:
                        # For autoencoders
                        reconstructed, _ = self.model(inputs)
                        loss = criterion(reconstructed, inputs)
                    
                    # Scale loss for gradient accumulation
                    loss = loss / self.gradient_accumulation_steps
                
                # Backward pass
                self.scaler.scale(loss).backward()
            else:
                if targets is not None:
                    outputs = self.model(inputs)
                    loss = criterion(outputs, targets)
                else:
                    reconstructed, _ = self.model(inputs)
                    loss = criterion(reconstructed, inputs)
                
                loss = loss / self.gradient_accumulation_steps
                loss.backward()
            
            total_loss += loss.item() * self.gradient_accumulation_steps
            num_batches += 1
            
            # Update weights
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                if self.use_mixed_precision:
                    # Gradient clipping
                    if clip_grad_norm is not None:
                        self.scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), clip_grad_norm
                        )
                    
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    if clip_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), clip_grad_norm
                        )
                    optimizer.step()
                
                optimizer.zero_grad()
            
            # Update progress bar
            pbar.set_postfix({"loss": loss.item() * self.gradient_accumulation_steps})
        
        avg_loss = total_loss / num_batches
        
        # Log metrics
        self.writer.add_scalar("Train/Loss", avg_loss, epoch)
        if self.use_wandb:
            wandb.log({"train/loss": avg_loss, "epoch": epoch})
        
        return {"loss": avg_loss}
    
    def validate(
        self,
        criterion: nn.Module,
        epoch: int
    ) -> Dict[str, float]:
        """
        Validate model
        
        Args:
            criterion: Loss function
            epoch: Current epoch number
            
        Returns:
            Dictionary with validation metrics
        """
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                if isinstance(batch, (list, tuple)):
                    inputs = batch[0].to(self.device)
                    targets = batch[1].to(self.device)
                else:
                    inputs = batch.to(self.device)
                    targets = None
                
                if self.use_mixed_precision:
                    with autocast():
                        if targets is not None:
                            outputs = self.model(inputs)
                            loss = criterion(outputs, targets)
                        else:
                            reconstructed, _ = self.model(inputs)
                            loss = criterion(reconstructed, inputs)
                else:
                    if targets is not None:
                        outputs = self.model(inputs)
                        loss = criterion(outputs, targets)
                    else:
                        reconstructed, _ = self.model(inputs)
                        loss = criterion(reconstructed, inputs)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        
        # Log metrics
        self.writer.add_scalar("Val/Loss", avg_loss, epoch)
        if self.use_wandb:
            wandb.log({"val/loss": avg_loss, "epoch": epoch})
        
        return {"loss": avg_loss}
    
    def train(
        self,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        num_epochs: int,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        clip_grad_norm: Optional[float] = None,
        save_dir: Optional[str] = None,
        save_best: bool = True
    ):
        """
        Full training loop
        
        Args:
            optimizer: Optimizer
            criterion: Loss function
            num_epochs: Number of epochs
            scheduler: Learning rate scheduler (optional)
            clip_grad_norm: Gradient clipping norm (optional)
            save_dir: Directory to save checkpoints
            save_best: Whether to save best model
        """
        best_val_loss = float('inf')
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        for epoch in range(1, num_epochs + 1):
            logger.info(f"Epoch {epoch}/{num_epochs}")
            
            # Train
            train_metrics = self.train_epoch(optimizer, criterion, epoch, clip_grad_norm)
            
            # Validate
            val_metrics = self.validate(criterion, epoch)
            
            # Learning rate scheduling
            if scheduler is not None:
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_metrics.get("loss", train_metrics["loss"]))
                else:
                    scheduler.step()
            
            # Save checkpoint
            if save_dir:
                checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_metrics["loss"],
                    "val_loss": val_metrics.get("loss", float('inf'))
                }
                
                if scheduler is not None:
                    checkpoint["scheduler_state_dict"] = scheduler.state_dict()
                
                # Save latest
                torch.save(checkpoint, os.path.join(save_dir, "latest.pth"))
                
                # Save best
                if save_best and val_metrics.get("loss", float('inf')) < best_val_loss:
                    best_val_loss = val_metrics["loss"]
                    torch.save(checkpoint, os.path.join(save_dir, "best.pth"))
                    logger.info(f"Saved best model (val_loss: {best_val_loss:.4f})")
        
        self.writer.close()
        if self.use_wandb:
            wandb.finish()
    
    def save_checkpoint(self, path: str, optimizer: optim.Optimizer, epoch: int):
        """Save training checkpoint"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str, optimizer: Optional[optim.Optimizer] = None):
        """Load training checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        logger.info(f"Checkpoint loaded from {path}")
        return checkpoint.get("epoch", 0)

