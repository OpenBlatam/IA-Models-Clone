#!/usr/bin/env python3
"""
Advanced Trainer - State-of-the-art training implementation
Provides comprehensive training with mixed precision, gradient accumulation, and advanced features
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import logging
import time
import math
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
import numpy as np
from pathlib import Path
import json
import uuid
from datetime import datetime, timezone
from tqdm import tqdm
import wandb
import tensorboard
from torch.utils.tensorboard import SummaryWriter

@dataclass
class TrainingConfig:
    """Configuration for training."""
    # Basic training parameters
    num_epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    gradient_clip_norm: float = 1.0
    
    # Advanced training features
    use_mixed_precision: bool = True
    use_gradient_accumulation: bool = True
    gradient_accumulation_steps: int = 4
    use_gradient_checkpointing: bool = True
    use_data_parallel: bool = False
    use_distributed: bool = False
    
    # Optimization
    optimizer: str = "adamw"  # adamw, adam, sgd, rmsprop
    scheduler: str = "cosine"  # cosine, linear, step, exponential
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0
    
    # Early stopping
    use_early_stopping: bool = True
    patience: int = 10
    min_delta: float = 1e-4
    
    # Model checkpointing
    save_best_model: bool = True
    save_last_model: bool = True
    checkpoint_dir: str = "checkpoints"
    save_every_n_epochs: int = 10
    
    # Logging and monitoring
    use_wandb: bool = False
    use_tensorboard: bool = True
    log_every_n_steps: int = 100
    eval_every_n_epochs: int = 1
    
    # Evaluation
    eval_metrics: List[str] = field(default_factory=lambda: ["accuracy", "loss"])
    use_validation: bool = True
    
    # Advanced features
    use_ema: bool = False
    ema_decay: float = 0.999
    use_swa: bool = False
    swa_start_epoch: int = 50
    use_progressive_resizing: bool = False
    progressive_resize_schedule: List[Tuple[int, int]] = field(default_factory=list)

@dataclass
class TrainingResult:
    """Result of training process."""
    best_epoch: int
    best_metrics: Dict[str, float]
    final_metrics: Dict[str, float]
    training_history: Dict[str, List[float]]
    training_time: float
    total_steps: int
    model_path: Optional[str] = None
    config: Optional[TrainingConfig] = None

class Trainer:
    """Advanced trainer with state-of-the-art features."""
    
    def __init__(self, model: nn.Module, config: TrainingConfig, 
                 train_loader: DataLoader, val_loader: Optional[DataLoader] = None,
                 device: str = "auto", logger: logging.Logger = None):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.logger = logger or logging.getLogger(__name__)
        
        # Device setup
        self.device = self._setup_device(device)
        self.model = self.model.to(self.device)
        
        # Training state
        self.current_epoch = 0
        self.current_step = 0
        self.best_metrics = {}
        self.training_history = {"train_loss": [], "val_loss": [], "val_metrics": []}
        
        # Initialize components
        self._initialize_optimizer()
        self._initialize_scheduler()
        self._initialize_scaler()
        self._initialize_callbacks()
        self._initialize_logging()
        
        # Advanced features
        self.ema_model = None
        self.swa_model = None
        self._initialize_advanced_features()
        
        self.logger.info(f"Trainer initialized on device: {self.device}")
    
    def _setup_device(self, device: str) -> torch.device:
        """Setup training device."""
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        
        return torch.device(device)
    
    def _initialize_optimizer(self):
        """Initialize optimizer."""
        if self.config.optimizer.lower() == "adamw":
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer.lower() == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer.lower() == "sgd":
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                momentum=0.9
            )
        elif self.config.optimizer.lower() == "rmsprop":
            self.optimizer = optim.RMSprop(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
        
        self.logger.info(f"Initialized optimizer: {self.config.optimizer}")
    
    def _initialize_scheduler(self):
        """Initialize learning rate scheduler."""
        if self.config.scheduler.lower() == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.config.num_epochs
            )
        elif self.config.scheduler.lower() == "linear":
            self.scheduler = optim.lr_scheduler.LinearLR(
                self.optimizer, start_factor=1.0, end_factor=0.0,
                total_iters=self.config.num_epochs
            )
        elif self.config.scheduler.lower() == "step":
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=30, gamma=0.1
            )
        elif self.config.scheduler.lower() == "exponential":
            self.scheduler = optim.lr_scheduler.ExponentialLR(
                self.optimizer, gamma=0.95
            )
        else:
            self.scheduler = None
        
        if self.scheduler:
            self.logger.info(f"Initialized scheduler: {self.config.scheduler}")
    
    def _initialize_scaler(self):
        """Initialize gradient scaler for mixed precision."""
        if self.config.use_mixed_precision:
            self.scaler = GradScaler()
            self.logger.info("Initialized gradient scaler for mixed precision")
        else:
            self.scaler = None
    
    def _initialize_callbacks(self):
        """Initialize training callbacks."""
        self.callbacks = []
        
        # Early stopping
        if self.config.use_early_stopping:
            from .callbacks import EarlyStopping
            self.callbacks.append(EarlyStopping(
                patience=self.config.patience,
                min_delta=self.config.min_delta
            ))
        
        # Model checkpointing
        if self.config.save_best_model or self.config.save_last_model:
            from .callbacks import ModelCheckpoint
            self.callbacks.append(ModelCheckpoint(
                save_best=self.config.save_best_model,
                save_last=self.config.save_last_model,
                checkpoint_dir=self.config.checkpoint_dir
            ))
        
        # Learning rate scheduler
        if self.scheduler:
            from .callbacks import LearningRateScheduler
            self.callbacks.append(LearningRateScheduler(self.scheduler))
        
        # Progress bar
        from .callbacks import ProgressBar
        self.callbacks.append(ProgressBar())
        
        self.logger.info(f"Initialized {len(self.callbacks)} callbacks")
    
    def _initialize_logging(self):
        """Initialize logging and monitoring."""
        # TensorBoard
        if self.config.use_tensorboard:
            self.tb_writer = SummaryWriter(log_dir="runs")
            self.logger.info("Initialized TensorBoard logging")
        else:
            self.tb_writer = None
        
        # Weights & Biases
        if self.config.use_wandb:
            try:
                wandb.init(project="advanced-training", config=self.config.__dict__)
                self.logger.info("Initialized Weights & Biases logging")
            except Exception as e:
                self.logger.warning(f"Failed to initialize W&B: {e}")
                self.config.use_wandb = False
    
    def _initialize_advanced_features(self):
        """Initialize advanced training features."""
        # EMA (Exponential Moving Average)
        if self.config.use_ema:
            self.ema_model = self._create_ema_model()
            self.logger.info("Initialized EMA model")
        
        # SWA (Stochastic Weight Averaging)
        if self.config.use_swa:
            self.swa_model = self._create_swa_model()
            self.logger.info("Initialized SWA model")
    
    def _create_ema_model(self):
        """Create EMA model."""
        # Create a copy of the model for EMA
        ema_model = type(self.model)(**self.model.config.__dict__ if hasattr(self.model, 'config') else {})
        ema_model.load_state_dict(self.model.state_dict())
        ema_model.eval()
        return ema_model
    
    def _create_swa_model(self):
        """Create SWA model."""
        # Create a copy of the model for SWA
        swa_model = type(self.model)(**self.model.config.__dict__ if hasattr(self.model, 'config') else {})
        swa_model.load_state_dict(self.model.state_dict())
        return swa_model
    
    def train(self) -> TrainingResult:
        """Main training loop."""
        self.logger.info("Starting training...")
        start_time = time.time()
        
        try:
            for epoch in range(self.config.num_epochs):
                self.current_epoch = epoch
                
                # Train for one epoch
                train_metrics = self._train_epoch()
                
                # Validate if validation loader is available
                val_metrics = {}
                if self.val_loader and epoch % self.config.eval_every_n_epochs == 0:
                    val_metrics = self._validate_epoch()
                
                # Update training history
                self._update_training_history(train_metrics, val_metrics)
                
                # Log metrics
                self._log_metrics(epoch, train_metrics, val_metrics)
                
                # Call callbacks
                self._call_callbacks(epoch, train_metrics, val_metrics)
                
                # Check for early stopping
                if self._should_stop_early():
                    self.logger.info(f"Early stopping at epoch {epoch}")
                    break
                
                # Update learning rate
                if self.scheduler:
                    self.scheduler.step()
            
            # Finalize training
            training_time = time.time() - start_time
            result = self._finalize_training(training_time)
            
            self.logger.info(f"Training completed in {training_time:.2f} seconds")
            return result
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            raise
    
    def _train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = self._move_batch_to_device(batch)
            
            # Forward pass with mixed precision
            if self.config.use_mixed_precision and self.scaler:
                with autocast():
                    loss = self._compute_loss(batch)
            else:
                loss = self._compute_loss(batch)
            
            # Backward pass
            if self.config.use_gradient_accumulation:
                loss = loss / self.config.gradient_accumulation_steps
            
            if self.config.use_mixed_precision and self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient accumulation
            if self.config.use_gradient_accumulation:
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    self._optimizer_step()
            else:
                self._optimizer_step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            self.current_step += 1
            
            # Update progress bar
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            # Log every n steps
            if self.current_step % self.config.log_every_n_steps == 0:
                self._log_step_metrics(loss.item())
        
        return {"train_loss": total_loss / num_batches}
    
    def _validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                # Move batch to device
                batch = self._move_batch_to_device(batch)
                
                # Forward pass
                if self.config.use_mixed_precision and self.scaler:
                    with autocast():
                        loss = self._compute_loss(batch)
                else:
                    loss = self._compute_loss(batch)
                
                total_loss += loss.item()
                num_batches += 1
        
        return {"val_loss": total_loss / num_batches}
    
    def _compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute loss for a batch."""
        # This is a placeholder - implement based on your specific model
        if "input_ids" in batch and "labels" in batch:
            # For language models
            outputs = self.model(batch["input_ids"])
            if hasattr(outputs, 'logits'):
                return F.cross_entropy(outputs.logits.view(-1, outputs.logits.size(-1)), 
                                     batch["labels"].view(-1))
            else:
                return F.cross_entropy(outputs, batch["labels"])
        elif "images" in batch and "targets" in batch:
            # For vision models
            outputs = self.model(batch["images"])
            return F.cross_entropy(outputs, batch["targets"])
        else:
            # Generic loss computation
            outputs = self.model(batch["input"])
            return F.mse_loss(outputs, batch["target"])
    
    def _move_batch_to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Move batch to device."""
        return {key: value.to(self.device) if isinstance(value, torch.Tensor) else value 
                for key, value in batch.items()}
    
    def _optimizer_step(self):
        """Perform optimizer step with gradient clipping."""
        if self.config.use_mixed_precision and self.scaler:
            # Gradient clipping
            if self.config.gradient_clip_norm > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # Gradient clipping
            if self.config.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
            
            self.optimizer.step()
        
        self.optimizer.zero_grad()
    
    def _update_training_history(self, train_metrics: Dict[str, float], val_metrics: Dict[str, float]):
        """Update training history."""
        self.training_history["train_loss"].append(train_metrics.get("train_loss", 0.0))
        
        if val_metrics:
            self.training_history["val_loss"].append(val_metrics.get("val_loss", 0.0))
            self.training_history["val_metrics"].append(val_metrics)
    
    def _log_metrics(self, epoch: int, train_metrics: Dict[str, float], val_metrics: Dict[str, float]):
        """Log metrics to various backends."""
        # Console logging
        self.logger.info(f"Epoch {epoch}: Train Loss: {train_metrics.get('train_loss', 0.0):.4f}")
        if val_metrics:
            self.logger.info(f"Epoch {epoch}: Val Loss: {val_metrics.get('val_loss', 0.0):.4f}")
        
        # TensorBoard logging
        if self.tb_writer:
            for key, value in train_metrics.items():
                self.tb_writer.add_scalar(f"Train/{key}", value, epoch)
            for key, value in val_metrics.items():
                self.tb_writer.add_scalar(f"Val/{key}", value, epoch)
        
        # Weights & Biases logging
        if self.config.use_wandb:
            wandb.log({
                "epoch": epoch,
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"val_{k}": v for k, v in val_metrics.items()}
            })
    
    def _log_step_metrics(self, loss: float):
        """Log step-level metrics."""
        if self.tb_writer:
            self.tb_writer.add_scalar("Train/StepLoss", loss, self.current_step)
        
        if self.config.use_wandb:
            wandb.log({"step": self.current_step, "train_loss": loss})
    
    def _call_callbacks(self, epoch: int, train_metrics: Dict[str, float], val_metrics: Dict[str, float]):
        """Call training callbacks."""
        for callback in self.callbacks:
            try:
                callback.on_epoch_end(epoch, train_metrics, val_metrics)
            except Exception as e:
                self.logger.error(f"Callback {callback.__class__.__name__} failed: {e}")
    
    def _should_stop_early(self) -> bool:
        """Check if training should stop early."""
        for callback in self.callbacks:
            if hasattr(callback, 'should_stop') and callback.should_stop():
                return True
        return False
    
    def _finalize_training(self, training_time: float) -> TrainingResult:
        """Finalize training and return results."""
        # Get best metrics
        best_epoch = np.argmin(self.training_history["val_loss"]) if self.training_history["val_loss"] else 0
        best_metrics = {
            "train_loss": self.training_history["train_loss"][best_epoch],
            "val_loss": self.training_history["val_loss"][best_epoch] if self.training_history["val_loss"] else 0.0
        }
        
        # Get final metrics
        final_metrics = {
            "train_loss": self.training_history["train_loss"][-1],
            "val_loss": self.training_history["val_loss"][-1] if self.training_history["val_loss"] else 0.0
        }
        
        # Close logging
        if self.tb_writer:
            self.tb_writer.close()
        
        if self.config.use_wandb:
            wandb.finish()
        
        return TrainingResult(
            best_epoch=best_epoch,
            best_metrics=best_metrics,
            final_metrics=final_metrics,
            training_history=self.training_history,
            training_time=training_time,
            total_steps=self.current_step,
            config=self.config
        )
    
    def save_model(self, path: str):
        """Save model to file."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'epoch': self.current_epoch,
            'config': self.config
        }, path)
        self.logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load model from file."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.logger.info(f"Model loaded from {path}")
