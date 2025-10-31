"""
Refactored Training Manager for HeyGen AI

This module provides clean, efficient training following deep learning best practices
with proper error handling, mixed precision, gradient accumulation, and experiment tracking.
Now enhanced with UltraPerformanceOptimizer for maximum speed improvements.

Following expert-level deep learning development principles:
- Proper PyTorch training loops with mixed precision
- Comprehensive error handling and validation
- Modern PyTorch features and optimizations
- Best practices for model training and evaluation
"""

import os
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from dataclasses import dataclass, field
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import Optimizer, AdamW
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, LinearLR
from torch.cuda.amp import autocast, GradScaler
from torch.nn.utils import clip_grad_norm_
import numpy as np
from tqdm import tqdm

from transformers import (
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup
)

# Import the ultra performance optimizer
from .ultra_performance_optimizer import (
    UltraPerformanceOptimizer, 
    UltraPerformanceConfig,
    PerformanceProfiler
)

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for training with comprehensive settings."""
    
    # General Settings
    seed: int = 42
    num_epochs: int = 10
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    warmup_steps: int = 100
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 100
    
    # Learning Rate
    initial_lr: float = 5e-5
    min_lr: float = 1e-6
    scheduler_type: str = "cosine"  # cosine, linear, constant, polynomial
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    
    # Mixed Precision
    mixed_precision_enabled: bool = True
    dtype: str = "fp16"  # fp16, bf16, fp32
    autocast: bool = True
    scaler: bool = True
    
    # Early Stopping
    early_stopping_enabled: bool = True
    patience: int = 3
    min_delta: float = 0.001
    monitor: str = "val_loss"
    mode: str = "min"  # min or max
    
    # Checkpointing
    save_best_only: bool = True
    save_last_checkpoint: bool = True
    checkpoint_dir: str = "checkpoints"
    
    # Validation
    validation_interval: int = 1
    eval_accumulation_steps: int = 1
    
    # Logging
    log_interval: int = 100
    tensorboard_logging: bool = True
    wandb_logging: bool = False
    project_name: str = "heygen-ai"
    run_name: str = "training-run"
    
    # Ultra Performance Settings
    enable_ultra_performance: bool = True
    performance_mode: str = "balanced"  # maximum, balanced, memory-efficient
    enable_torch_compile: bool = True
    enable_flash_attention: bool = True
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.num_epochs <= 0:
            raise ValueError("num_epochs must be positive")
        
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        
        if self.initial_lr <= 0:
            raise ValueError("initial_lr must be positive")
        
        if self.patience <= 0:
            raise ValueError("patience must be positive")
        
        if self.performance_mode not in ["maximum", "balanced", "memory-efficient"]:
            raise ValueError(f"Invalid performance_mode: {self.performance_mode}")


class TrainingMetrics:
    """Class for tracking training metrics and statistics."""
    
    def __init__(self):
        """Initialize training metrics."""
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        self.grad_norms = []
        self.training_times = []
        self.best_metric = float('inf')
        self.best_epoch = 0
    
    def update(self, train_loss: float, val_loss: Optional[float] = None, 
               lr: Optional[float] = None, grad_norm: Optional[float] = None,
               training_time: Optional[float] = None):
        """Update metrics with new values."""
        self.train_losses.append(train_loss)
        
        if val_loss is not None:
            self.val_losses.append(val_loss)
        
        if lr is not None:
            self.learning_rates.append(lr)
        
        if grad_norm is not None:
            self.grad_norms.append(grad_norm)
        
        if training_time is not None:
            self.training_times.append(training_time)
    
    def get_latest_metrics(self) -> Dict[str, float]:
        """Get the latest metrics."""
        metrics = {
            'train_loss': self.train_losses[-1] if self.train_losses else 0.0,
            'val_loss': self.val_losses[-1] if self.val_losses else 0.0,
            'learning_rate': self.learning_rates[-1] if self.learning_rates else 0.0,
            'grad_norm': self.grad_norms[-1] if self.grad_norms else 0.0,
            'training_time': self.training_times[-1] if self.training_times else 0.0
        }
        return metrics
    
    def is_best_metric(self, metric: float, mode: str = "min") -> bool:
        """Check if the current metric is the best so far."""
        if mode == "min":
            if metric < self.best_metric:
                self.best_metric = metric
                return True
        else:  # mode == "max"
            if metric > self.best_metric:
                self.best_metric = metric
                return True
        return False


class EarlyStopping:
    """Early stopping mechanism for training."""
    
    def __init__(self, patience: int = 3, min_delta: float = 0.001, mode: str = "min"):
        """Initialize early stopping."""
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, val_score: float) -> bool:
        """Check if training should stop early."""
        if self.best_score is None:
            self.best_score = val_score
        elif self.mode == "min":
            if val_score < self.best_score - self.min_delta:
                self.best_score = val_score
                self.counter = 0
            else:
                self.counter += 1
        else:  # mode == "max"
            if val_score > self.best_score + self.min_delta:
                self.best_score = val_score
                self.counter = 0
            else:
                self.counter += 1
        
        if self.counter >= self.patience:
            self.early_stop = True
        
        return self.early_stop


class ModelCheckpoint:
    """Model checkpointing with best model saving."""
    
    def __init__(self, checkpoint_dir: str, save_best_only: bool = True, 
                 save_last_checkpoint: bool = True):
        """Initialize model checkpointing."""
        self.checkpoint_dir = Path(checkpoint_dir)
        self.save_best_only = save_best_only
        self.save_last_checkpoint = save_last_checkpoint
        self.best_metric = float('inf')
        self.best_epoch = 0
        
        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def save_checkpoint(self, model: nn.Module, optimizer: Optimizer, 
                       scheduler: Any, epoch: int, metric: float, 
                       config: TrainingConfig, is_best: bool = False):
        """Save a model checkpoint."""
        try:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'metric': metric,
                'config': config,
                'timestamp': time.time()
            }
            
            # Save best model
            if is_best:
                best_path = self.checkpoint_dir / "best_model.pt"
                torch.save(checkpoint, best_path)
                self.best_metric = metric
                self.best_epoch = epoch
                logger.info(f"Best model saved at epoch {epoch} with metric {metric:.4f}")
            
            # Save last checkpoint
            if self.save_last_checkpoint:
                last_path = self.checkpoint_dir / "last_checkpoint.pt"
                torch.save(checkpoint, last_path)
            
            # Save epoch-specific checkpoint
            epoch_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
            torch.save(checkpoint, epoch_path)
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            raise
    
    def load_checkpoint(self, checkpoint_path: str, model: nn.Module, 
                       optimizer: Optimizer, scheduler: Any = None) -> Dict[str, Any]:
        """Load a model checkpoint."""
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            if scheduler and checkpoint['scheduler_state_dict']:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            logger.info(f"Checkpoint loaded from {checkpoint_path}")
            return checkpoint
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise


class EnhancedTrainer:
    """Enhanced trainer with comprehensive training functionality."""
    
    def __init__(self, model: nn.Module, train_dataloader: DataLoader, 
                 val_dataloader: Optional[DataLoader], config: TrainingConfig):
        """Initialize the enhanced trainer."""
        if not isinstance(config, TrainingConfig):
            raise TypeError("config must be a TrainingConfig instance")
        
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move model to device
        self.model.to(self.device)
        
        # Initialize components
        self._setup_optimizer()
        self._setup_scheduler()
        self._setup_mixed_precision()
        self._setup_components()
        
        # Apply performance optimizations
        if self.config.enable_ultra_performance:
            self._setup_ultra_performance()
    
    def _setup_optimizer(self):
        """Setup the optimizer."""
        try:
            # Get trainable parameters
            trainable_params = [p for p in self.model.parameters() if p.requires_grad]
            
            if not trainable_params:
                raise ValueError("No trainable parameters found in model")
            
            self.optimizer = AdamW(
                trainable_params,
                lr=self.config.initial_lr,
                weight_decay=self.config.weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8
            )
            
            logger.info(f"Optimizer initialized with {len(trainable_params)} trainable parameters")
            
        except Exception as e:
            logger.error(f"Failed to setup optimizer: {e}")
            raise
    
    def _setup_scheduler(self):
        """Setup the learning rate scheduler."""
        try:
            total_steps = len(self.train_dataloader) * self.config.num_epochs // self.config.gradient_accumulation_steps
            warmup_steps = int(total_steps * self.config.warmup_ratio)
            
            if self.config.scheduler_type == "cosine":
                self.scheduler = get_cosine_schedule_with_warmup(
                    self.optimizer,
                    num_warmup_steps=warmup_steps,
                    num_training_steps=total_steps
                )
            elif self.config.scheduler_type == "linear":
                self.scheduler = get_linear_schedule_with_warmup(
                    self.optimizer,
                    num_warmup_steps=warmup_steps,
                    num_training_steps=total_steps
                )
            elif self.config.scheduler_type == "polynomial":
                self.scheduler = get_polynomial_decay_schedule_with_warmup(
                    self.optimizer,
                    num_warmup_steps=warmup_steps,
                    num_training_steps=total_steps
                )
            else:
                # Constant learning rate
                self.scheduler = LambdaLR(self.optimizer, lambda _: 1.0)
            
            logger.info(f"Scheduler {self.config.scheduler_type} initialized with {warmup_steps} warmup steps")
            
        except Exception as e:
            logger.error(f"Failed to setup scheduler: {e}")
            raise
    
    def _setup_mixed_precision(self):
        """Setup mixed precision training."""
        try:
            if self.config.mixed_precision_enabled and self.device.type == 'cuda':
                self.scaler = GradScaler()
                self.autocast_enabled = self.config.autocast
                logger.info("Mixed precision training enabled")
            else:
                self.scaler = None
                self.autocast_enabled = False
                logger.info("Mixed precision training disabled")
                
        except Exception as e:
            logger.warning(f"Failed to setup mixed precision: {e}")
            self.scaler = None
            self.autocast_enabled = False
    
    def _setup_components(self):
        """Setup training components."""
        try:
            # Loss function (can be customized)
            self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
            
            # Metrics tracking
            self.metrics = TrainingMetrics()
            
            # Early stopping
            if self.config.early_stopping_enabled:
                self.early_stopping = EarlyStopping(
                    patience=self.config.patience,
                    min_delta=self.config.min_delta,
                    mode=self.config.mode
                )
            else:
                self.early_stopping = None
            
            # Model checkpointing
            self.checkpointer = ModelCheckpoint(
                checkpoint_dir=self.config.checkpoint_dir,
                save_best_only=self.config.save_best_only,
                save_last_checkpoint=self.config.save_last_checkpoint
            )
            
            # Experiment tracking
            if self.config.wandb_logging:
                self._setup_wandb()
            
            logger.info("Training components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to setup training components: {e}")
            raise
    
    def _setup_ultra_performance(self):
        """Setup ultra performance optimizations."""
        try:
            performance_config = UltraPerformanceConfig(
                enable_torch_compile=self.config.enable_torch_compile,
                enable_flash_attention=self.config.enable_flash_attention,
                performance_mode=self.config.performance_mode
            )
            
            self.ultra_performance_optimizer = UltraPerformanceOptimizer(
                config=performance_config,
                device=self.device
            )
            
            # Apply optimizations to model
            self.ultra_performance_optimizer.optimize_model(self.model)
            
            logger.info("Ultra performance optimizations applied successfully")
            
        except Exception as e:
            logger.warning(f"Failed to setup ultra performance optimizations: {e}")
            self.ultra_performance_optimizer = None
    
    def _setup_wandb(self):
        """Setup Weights & Biases logging."""
        try:
            import wandb
            wandb.init(
                project=self.config.project_name,
                name=self.config.run_name,
                config=vars(self.config)
            )
            logger.info("Weights & Biases logging initialized")
        except ImportError:
            logger.warning("Weights & Biases not available")
        except Exception as e:
            logger.warning(f"Failed to setup Weights & Biases: {e}")
    
    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        # Progress bar
        progress_bar = tqdm(
            self.train_dataloader,
            desc=f'Epoch {epoch}/{self.config.num_epochs}',
            leave=True
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            try:
                # Move batch to device
                batch = self._move_batch_to_device(batch)
                
                # Forward pass with mixed precision
                loss = self._forward_pass(batch)
                
                # Backward pass with gradient accumulation
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    self._backward_pass(loss)
                    
                    # Gradient clipping
                    if self.config.max_grad_norm > 0:
                        grad_norm = self._clip_gradients()
                    else:
                        grad_norm = 0.0
                    
                    # Optimizer step
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                
                # Update metrics
                total_loss += loss.item()
                num_batches += 1
                
                # Update progress bar
                current_lr = self.scheduler.get_last_lr()[0]
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'lr': f'{current_lr:.6f}',
                    'grad_norm': f'{grad_norm:.4f}' if 'grad_norm' in locals() else 'N/A'
                })
                
                # Log to wandb
                if self.config.wandb_logging:
                    self._log_to_wandb(epoch, batch_idx, loss.item(), current_lr, grad_norm)
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    logger.error(f"GPU OOM in batch {batch_idx}: {e}")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                else:
                    raise e
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def _move_batch_to_device(self, batch: Any) -> Any:
        """Move batch tensors to device."""
        if isinstance(batch, dict):
            return {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}
        elif isinstance(batch, (list, tuple)):
            return [v.to(self.device) if torch.is_tensor(v) else v for v in batch]
        else:
            return batch.to(self.device)
    
    def _forward_pass(self, batch: Any) -> torch.Tensor:
        """Perform forward pass with mixed precision."""
        if self.autocast_enabled and self.scaler:
            with autocast():
                outputs = self.model(**batch)
                loss = self.criterion(outputs.logits.view(-1, outputs.logits.size(-1)), 
                                   batch['labels'].view(-1))
        else:
            outputs = self.model(**batch)
            loss = self.criterion(outputs.logits.view(-1, outputs.logits.size(-1)), 
                               batch['labels'].view(-1))
        
        return loss
    
    def _backward_pass(self, loss: torch.Tensor):
        """Perform backward pass with mixed precision."""
        if self.scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
    
    def _clip_gradients(self) -> float:
        """Clip gradients and return gradient norm."""
        if self.scaler:
            self.scaler.unscale_(self.optimizer)
        
        grad_norm = clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
        return grad_norm.item()
    
    def validate(self, epoch: int) -> float:
        """Validate the model."""
        if not self.val_dataloader:
            return 0.0
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_dataloader:
                try:
                    # Move batch to device
                    batch = self._move_batch_to_device(batch)
                    
                    # Forward pass
                    outputs = self.model(**batch)
                    loss = self.criterion(outputs.logits.view(-1, outputs.logits.size(-1)), 
                                       batch['labels'].view(-1))
                    
                    total_loss += loss.item()
                    num_batches += 1
                    
                except Exception as e:
                    logger.error(f"Validation error in batch: {e}")
                    continue
        
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        return avg_loss
    
    def _log_to_wandb(self, epoch: int, batch_idx: int, loss: float, lr: float, grad_norm: float):
        """Log metrics to Weights & Biases."""
        try:
            import wandb
            wandb.log({
                'epoch': epoch,
                'batch': batch_idx,
                'train_loss': loss,
                'learning_rate': lr,
                'gradient_norm': grad_norm
            })
        except Exception as e:
            logger.warning(f"Failed to log to wandb: {e}")
    
    def train(self) -> Dict[str, Any]:
        """Main training loop."""
        logger.info("Starting training...")
        start_time = time.time()
        
        try:
            for epoch in range(self.config.num_epochs):
                epoch_start_time = time.time()
                
                # Training
                train_loss = self.train_epoch(epoch)
                
                # Validation
                val_loss = self.validate(epoch)
                
                # Update metrics
                epoch_time = time.time() - epoch_start_time
                current_lr = self.scheduler.get_last_lr()[0]
                
                self.metrics.update(
                    train_loss=train_loss,
                    val_loss=val_loss,
                    lr=current_lr,
                    training_time=epoch_time
                )
                
                # Log epoch results
                logger.info(
                    f'Epoch {epoch}: Train Loss: {train_loss:.4f}, '
                    f'Val Loss: {val_loss:.4f}, LR: {current_lr:.6f}, '
                    f'Time: {epoch_time:.2f}s'
                )
                
                # Checkpointing
                is_best = self.metrics.is_best_metric(val_loss, self.config.mode)
                self.checkpointer.save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch,
                    metric=val_loss,
                    config=self.config,
                    is_best=is_best
                )
                
                # Early stopping
                if self.early_stopping and self.early_stopping(val_loss):
                    logger.info(f'Early stopping triggered at epoch {epoch}')
                    break
                
                # Log to wandb
                if self.config.wandb_logging:
                    self._log_epoch_to_wandb(epoch, train_loss, val_loss, current_lr, epoch_time)
        
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
        finally:
            total_time = time.time() - start_time
            logger.info(f"Training completed in {total_time:.2f}s")
        
        return self.metrics.get_latest_metrics()
    
    def _log_epoch_to_wandb(self, epoch: int, train_loss: float, val_loss: float, lr: float, epoch_time: float):
        """Log epoch metrics to Weights & Biases."""
        try:
            import wandb
            wandb.log({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'learning_rate': lr,
                'epoch_time': epoch_time
            })
        except Exception as e:
            logger.warning(f"Failed to log epoch to wandb: {e}")
    
    def save_model(self, save_path: str):
        """Save the trained model."""
        try:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'config': self.config,
                'metrics': self.metrics
            }, save_path)
            
            logger.info(f"Model saved to {save_path}")
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise
    
    def load_model(self, load_path: str):
        """Load a trained model."""
        try:
            checkpoint = torch.load(load_path, map_location=self.device)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            if 'metrics' in checkpoint:
                self.metrics = checkpoint['metrics']
            
            logger.info(f"Model loaded from {load_path}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise


# Factory function for creating enhanced trainer
def create_enhanced_trainer(
    model: nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: Optional[DataLoader],
    config: TrainingConfig
) -> EnhancedTrainer:
    """Create an enhanced trainer with the given configuration."""
    return EnhancedTrainer(model, train_dataloader, val_dataloader, config)
