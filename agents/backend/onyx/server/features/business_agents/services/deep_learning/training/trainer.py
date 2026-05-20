"""
Training Manager - Comprehensive Training Loop
==============================================

Training manager with mixed precision, gradient accumulation,
learning rate scheduling, and experiment tracking.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import (
    CosineAnnealingLR, ReduceLROnPlateau, StepLR,
    ExponentialLR, OneCycleLR
)
from typing import Optional, Dict, List, Any, Callable
import logging
from pathlib import Path

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    import structlog
    logger = structlog.get_logger(__name__)
except ImportError:
    logger = logging.getLogger(__name__)

from ..config.config_loader import TrainingConfig
from .early_stopping import EarlyStopping
from .checkpoint import CheckpointManager


class TrainingManager:
    """
    Comprehensive training manager with best practices.
    
    Features:
    - Mixed precision training (AMP)
    - Gradient accumulation
    - Learning rate scheduling
    - Early stopping
    - Checkpointing
    - Experiment tracking (TensorBoard/W&B)
    - Gradient clipping
    - NaN/Inf detection
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        config: Optional[TrainingConfig] = None,
        device: Optional[torch.device] = None,
        criterion: Optional[nn.Module] = None,
        experiment_tracker: Optional[Any] = None
    ):
        """
        Initialize training manager.
        
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            config: Training configuration
            device: Target device
            criterion: Loss function (defaults to CrossEntropyLoss)
            experiment_tracker: TensorBoard writer or wandb
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config or TrainingConfig()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.experiment_tracker = experiment_tracker
        
        # Move model to device
        self.model.to(self.device)
        
        # Mixed precision
        self.use_amp = (
            self.config.use_mixed_precision and 
            self.device.type == "cuda"
        )
        self.scaler = GradScaler() if self.use_amp else None
        
        # Optimizer
        self.optimizer = self._create_optimizer()
        
        # Learning rate scheduler
        self.scheduler = self._create_scheduler()
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=self.config.early_stopping_patience,
            min_delta=self.config.early_stopping_min_delta,
            mode=self.config.early_stopping_mode
        )
        
        # Checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=self.config.checkpoint_dir
        )
        
        # Training history
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "learning_rate": []
        }
        
        # Debugging
        if self.config.detect_anomaly:
            torch.autograd.set_detect_anomaly(True)
            logger.warning("Anomaly detection enabled - may slow down training")
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer based on config."""
        params = self.model.parameters()
        lr = self.config.learning_rate
        weight_decay = self.config.weight_decay
        
        optimizer_name = self.config.optimizer.lower()
        optimizer_params = self.config.optimizer_params or {}
        
        if optimizer_name == "adamw":
            return optim.AdamW(
                params,
                lr=lr,
                weight_decay=weight_decay,
                betas=optimizer_params.get("betas", (0.9, 0.999)),
                eps=optimizer_params.get("eps", 1e-8)
            )
        elif optimizer_name == "adam":
            return optim.Adam(
                params,
                lr=lr,
                weight_decay=weight_decay,
                **optimizer_params
            )
        elif optimizer_name == "sgd":
            return optim.SGD(
                params,
                lr=lr,
                momentum=optimizer_params.get("momentum", 0.9),
                weight_decay=weight_decay,
                **{k: v for k, v in optimizer_params.items() if k != "momentum"}
            )
        elif optimizer_name == "rmsprop":
            return optim.RMSprop(
                params,
                lr=lr,
                weight_decay=weight_decay,
                **optimizer_params
            )
        else:
            logger.warning(f"Unknown optimizer: {optimizer_name}, using AdamW")
            return optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    
    def _create_scheduler(self) -> Optional[Any]:
        """Create learning rate scheduler."""
        scheduler_type = self.config.learning_rate_scheduler.lower()
        scheduler_params = self.config.scheduler_params or {}
        
        if scheduler_type == "cosine":
            return CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.epochs,
                eta_min=scheduler_params.get("eta_min", 1e-6)
            )
        elif scheduler_type == "step":
            return StepLR(
                self.optimizer,
                step_size=scheduler_params.get("step_size", self.config.epochs // 3),
                gamma=scheduler_params.get("gamma", 0.1)
            )
        elif scheduler_type == "reduce_on_plateau":
            return ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=scheduler_params.get("factor", 0.5),
                patience=scheduler_params.get("patience", 3),
                verbose=True
            )
        elif scheduler_type == "exponential":
            return ExponentialLR(
                self.optimizer,
                gamma=scheduler_params.get("gamma", 0.95)
            )
        elif scheduler_type == "onecycle":
            return OneCycleLR(
                self.optimizer,
                max_lr=self.config.learning_rate,
                epochs=self.config.epochs,
                steps_per_epoch=len(self.train_loader)
            )
        return None
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch with mixed precision."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Gradient accumulation
            if batch_idx % self.config.gradient_accumulation_steps == 0:
                self.optimizer.zero_grad()
            
            # Mixed precision forward pass
            if self.use_amp:
                with autocast(dtype=getattr(torch, self.config.mixed_precision_dtype)):
                    output = self.model(data)
                    loss = self.criterion(output, target) / self.config.gradient_accumulation_steps
                
                # Backward pass with scaling
                self.scaler.scale(loss).backward()
                
                # Gradient accumulation
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    self.scaler.unscale_(self.optimizer)
                    clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip_norm
                    )
                    
                    # Optimizer step
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
            else:
                output = self.model(data)
                loss = self.criterion(output, target) / self.config.gradient_accumulation_steps
                loss.backward()
                
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip_norm
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
            # Metrics
            total_loss += loss.item() * self.config.gradient_accumulation_steps
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
            # Check for NaN/Inf
            if torch.isnan(loss) or torch.isinf(loss):
                logger.error("❌ NaN or Inf loss detected!")
                raise ValueError("Training diverged: NaN/Inf loss")
            
            # Logging
            if (batch_idx + 1) % self.config.log_interval == 0:
                logger.info(
                    f"Batch {batch_idx + 1}/{len(self.train_loader)}, "
                    f"Loss: {loss.item():.4f}, "
                    f"Acc: {100.0 * correct / total:.2f}%"
                )
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100.0 * correct / total
        
        return {"loss": avg_loss, "accuracy": accuracy}
    
    def validate(self) -> Dict[str, float]:
        """Validate model."""
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                if self.use_amp:
                    with autocast(dtype=getattr(torch, self.config.mixed_precision_dtype)):
                        output = self.model(data)
                        loss = self.criterion(output, target)
                else:
                    output = self.model(data)
                    loss = self.criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100.0 * correct / total
        
        return {"loss": avg_loss, "accuracy": accuracy}
    
    def train(self) -> Dict[str, List[float]]:
        """Full training loop."""
        logger.info(f"🚀 Starting training on {self.device}")
        logger.info(f"Mixed precision: {self.use_amp}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        best_val_loss = float('inf')
        
        for epoch in range(self.config.epochs):
            # Train
            train_metrics = self.train_epoch()
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["train_acc"].append(train_metrics["accuracy"])
            
            # Validate
            val_metrics = {}
            if self.val_loader is not None:
                val_metrics = self.validate()
                self.history["val_loss"].append(val_metrics.get("loss", 0.0))
                self.history["val_acc"].append(val_metrics.get("accuracy", 0.0))
            
            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_metrics.get("loss", train_metrics["loss"]))
                else:
                    self.scheduler.step()
            
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history["learning_rate"].append(current_lr)
            
            # Logging
            logger.info(
                f"Epoch {epoch+1}/{self.config.epochs} - "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Train Acc: {train_metrics['accuracy']:.2f}%"
            )
            if val_metrics:
                logger.info(
                    f"Val Loss: {val_metrics.get('loss', 0.0):.4f}, "
                    f"Val Acc: {val_metrics.get('accuracy', 0.0):.2f}%"
                )
            
            # Experiment tracking
            if self.experiment_tracker:
                log_dict = {
                    "epoch": epoch + 1,
                    "train_loss": train_metrics["loss"],
                    "train_accuracy": train_metrics["accuracy"],
                    "learning_rate": current_lr,
                }
                if val_metrics:
                    log_dict.update({
                        "val_loss": val_metrics.get("loss", 0.0),
                        "val_accuracy": val_metrics.get("accuracy", 0.0),
                    })
                
                if TENSORBOARD_AVAILABLE and isinstance(self.experiment_tracker, SummaryWriter):
                    for key, value in log_dict.items():
                        self.experiment_tracker.add_scalar(key, value, epoch + 1)
                elif WANDB_AVAILABLE:
                    wandb.log(log_dict)
            
            # Checkpointing
            is_best = False
            if val_metrics:
                val_loss = val_metrics.get("loss", float('inf'))
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    is_best = True
            
            if self.config.save_checkpoints:
                if (epoch + 1) % self.config.checkpoint_frequency == 0:
                    self.checkpoint_manager.save_checkpoint(
                        model=self.model,
                        optimizer=self.optimizer,
                        epoch=epoch + 1,
                        metrics={**train_metrics, **val_metrics},
                        is_best=is_best
                    )
            
            # Early stopping
            if self.val_loader is not None:
                val_loss = val_metrics.get("loss", float('inf'))
                if self.early_stopping(val_loss, self.model):
                    logger.info(f"⏹️ Early stopping at epoch {epoch+1}")
                    break
        
        logger.info("✅ Training completed")
        return self.history



