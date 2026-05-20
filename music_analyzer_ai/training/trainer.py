"""
Training System for Music Analysis Models
Complete training pipeline with optimization, logging, and checkpointing
"""

from typing import Dict, Any, Optional, List, Callable
import logging
import time
from pathlib import Path
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.cuda.amp import GradScaler, autocast
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available")


@dataclass
class TrainingConfig:
    """Configuration for training"""
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    optimizer: str = "adam"  # "adam", "sgd", "adamw"
    scheduler: str = "cosine"  # "cosine", "step", "plateau"
    early_stopping_patience: int = 10
    gradient_clip: float = 1.0
    use_mixed_precision: bool = True
    save_best_only: bool = True
    checkpoint_dir: str = "./checkpoints"
    log_interval: int = 10


class MusicModelTrainer:
    """
    Complete training system with:
    - Optimized training loop
    - Mixed precision training
    - Learning rate scheduling
    - Early stopping
    - Gradient clipping
    - Checkpointing
    - Experiment tracking
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        device: str = "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu",
        experiment_tracker: Optional[Any] = None
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.experiment_tracker = experiment_tracker
        
        # Optimizer
        self.optimizer = self._create_optimizer()
        
        # Scheduler
        self.scheduler = self._create_scheduler()
        
        # Mixed precision
        self.scaler = GradScaler() if config.use_mixed_precision and device == "cuda" else None
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.training_history: List[Dict[str, float]] = []
        
        # Checkpoint directory
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer based on config"""
        if self.config.optimizer.lower() == "adam":
            return optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer.lower() == "adamw":
            return optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer.lower() == "sgd":
            return optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                momentum=0.9,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
    
    def _create_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler"""
        if self.config.scheduler.lower() == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.epochs
            )
        elif self.config.scheduler.lower() == "step":
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.epochs // 3,
                gamma=0.1
            )
        elif self.config.scheduler.lower() == "plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=0.5,
                patience=5
            )
        else:
            return None
    
    def train_epoch(
        self,
        train_loader,
        criterion: nn.Module,
        epoch: int
    ) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, batch in enumerate(train_loader):
            features = batch["features"].to(self.device)
            labels = batch["label"].squeeze().to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.scaler:
                with autocast():
                    outputs = self.model(features)
                    loss = criterion(outputs, labels)
                
                # Backward pass
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.config.gradient_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip
                    )
                
                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(features)
                loss = criterion(outputs, labels)
                loss.backward()
                
                if self.config.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip
                    )
                
                self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Logging
            if batch_idx % self.config.log_interval == 0:
                logger.info(
                    f"Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, "
                    f"Loss: {loss.item():.4f}, Acc: {100*correct/total:.2f}%"
                )
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100 * correct / total
        
        return {
            "loss": avg_loss,
            "accuracy": accuracy
        }
    
    def validate(
        self,
        val_loader,
        criterion: nn.Module
    ) -> Dict[str, float]:
        """Validate model"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                features = batch["features"].to(self.device)
                labels = batch["label"].squeeze().to(self.device)
                
                if self.scaler:
                    with autocast():
                        outputs = self.model(features)
                        loss = criterion(outputs, labels)
                else:
                    outputs = self.model(features)
                    loss = criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100 * correct / total
        
        return {
            "loss": avg_loss,
            "accuracy": accuracy
        }
    
    def train(
        self,
        train_loader,
        val_loader,
        criterion: nn.Module,
        num_epochs: Optional[int] = None
    ) -> Dict[str, Any]:
        """Complete training loop"""
        num_epochs = num_epochs or self.config.epochs
        start_time = time.time()
        
        logger.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(1, num_epochs + 1):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch(train_loader, criterion, epoch)
            
            # Validate
            val_metrics = self.validate(val_loader, criterion)
            
            # Update scheduler
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics["loss"])
                else:
                    self.scheduler.step()
            
            # Log metrics
            current_lr = self.optimizer.param_groups[0]["lr"]
            metrics = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_acc": train_metrics["accuracy"],
                "val_loss": val_metrics["loss"],
                "val_acc": val_metrics["accuracy"],
                "learning_rate": current_lr
            }
            
            self.training_history.append(metrics)
            
            # Experiment tracking
            if self.experiment_tracker:
                try:
                    self.experiment_tracker.log(metrics)
                except Exception as e:
                    logger.warning(f"Experiment tracking error: {str(e)}")
            
            logger.info(
                f"Epoch {epoch}/{num_epochs} - "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Train Acc: {train_metrics['accuracy']:.2f}%, "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Val Acc: {val_metrics['accuracy']:.2f}%, "
                f"LR: {current_lr:.6f}"
            )
            
            # Checkpointing
            is_best = val_metrics["loss"] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics["loss"]
                self.patience_counter = 0
                self.save_checkpoint(epoch, is_best=True)
            else:
                self.patience_counter += 1
                if not self.config.save_best_only:
                    self.save_checkpoint(epoch, is_best=False)
            
            # Early stopping
            if self.patience_counter >= self.config.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
        
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f}s")
        
        return {
            "training_history": self.training_history,
            "best_val_loss": self.best_val_loss,
            "total_epochs": self.current_epoch,
            "training_time": training_time
        }
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "best_val_loss": self.best_val_loss,
            "training_history": self.training_history
        }
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model (val_loss: {self.best_val_loss:.4f})")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.scheduler and checkpoint.get("scheduler_state_dict"):
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.best_val_loss = checkpoint["best_val_loss"]
        self.training_history = checkpoint.get("training_history", [])
        logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")

