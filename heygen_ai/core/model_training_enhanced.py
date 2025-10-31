"""
Enhanced Model Training for HeyGen AI.

This module implements comprehensive training pipelines with proper
configuration management, experiment tracking, and model checkpointing.
Follows PyTorch best practices and includes advanced training techniques.
"""

import logging
import os
import time
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, asdict, field
from datetime import datetime
import hashlib
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
from torch.cuda.amp import autocast, GradScaler
import torch.distributed as dist
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import yaml
import json
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import wandb
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Comprehensive configuration for model training."""
    
    # Model settings
    model_name: str = "gpt2"
    model_type: str = "transformer"  # transformer, diffusion, custom
    model_path: Optional[str] = None
    
    # Training settings
    batch_size: int = 8
    learning_rate: float = 5e-5
    num_epochs: int = 10
    warmup_steps: int = 100
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 0.001
    
    # Optimization settings
    use_fp16: bool = True
    use_mixed_precision: bool = True
    use_distributed: bool = False
    use_gradient_clipping: bool = True
    use_amp: bool = True
    
    # Data settings
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    max_length: int = 512
    num_workers: int = 4
    pin_memory: bool = True
    
    # Checkpointing settings
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: int = 3
    checkpoint_dir: str = "./checkpoints"
    resume_from_checkpoint: Optional[str] = None
    
    # Logging settings
    logging_steps: int = 10
    log_dir: str = "./logs"
    use_wandb: bool = True
    use_tensorboard: bool = True
    project_name: str = "heygen_ai_training"
    
    # Cross-validation settings
    use_cross_validation: bool = False
    n_folds: int = 5
    
    # Device settings
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.use_fp16 and not torch.cuda.is_available():
            logger.warning("FP16 requested but CUDA not available, falling back to FP32")
            self.use_fp16 = False
        
        if self.train_split + self.val_split + self.test_split != 1.0:
            logger.warning("Train/val/test splits should sum to 1.0")
        
        # Create directories
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)


class TrainingMetrics:
    """Manages training metrics and logging."""
    
    def __init__(self, config: TrainingConfig):
        """Initialize training metrics.
        
        Args:
            config: Training configuration
        """
        self.config = config
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': [],
            'learning_rate': [],
            'epoch': [],
            'step': []
        }
        
        # Initialize logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Set up logging systems."""
        # TensorBoard
        if self.config.use_tensorboard:
            self.tensorboard_writer = SummaryWriter(self.config.log_dir)
        else:
            self.tensorboard_writer = None
        
        # Weights & Biases
        if self.config.use_wandb:
            try:
                wandb.init(
                    project=self.config.project_name,
                    config=asdict(self.config),
                    name=f"{self.config.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                )
            except Exception as e:
                logger.warning(f"Could not initialize wandb: {str(e)}")
                self.config.use_wandb = False
    
    def log_metrics(self, metrics: Dict[str, float], step: int, epoch: int):
        """Log metrics to all configured systems.
        
        Args:
            metrics: Dictionary of metrics to log
            step: Current training step
            epoch: Current epoch
        """
        # Store metrics locally
        for key, value in metrics.items():
            if key in self.metrics:
                self.metrics[key].append(value)
        
        # Log to TensorBoard
        if self.tensorboard_writer:
            for key, value in metrics.items():
                self.tensorboard_writer.add_scalar(key, value, step)
        
        # Log to Weights & Biases
        if self.config.use_wandb:
            wandb.log(metrics, step=step)
        
        # Log to console
        logger.info(f"Epoch {epoch}, Step {step}: {metrics}")
    
    def get_best_metric(self, metric_name: str, mode: str = "min") -> Tuple[float, int]:
        """Get the best value of a metric and its index.
        
        Args:
            metric_name: Name of the metric
            mode: "min" or "max" for optimization direction
            
        Returns:
            Tuple of (best_value, best_index)
        """
        if metric_name not in self.metrics:
            raise ValueError(f"Metric {metric_name} not found")
        
        values = self.metrics[metric_name]
        if not values:
            return float('inf') if mode == "min" else float('-inf'), -1
        
        if mode == "min":
            best_value = min(values)
            best_index = values.index(best_value)
        else:
            best_value = max(values)
            best_index = values.index(best_value)
        
        return best_value, best_index


class ModelCheckpointer:
    """Manages model checkpointing and saving."""
    
    def __init__(self, config: TrainingConfig, model: nn.Module):
        """Initialize model checkpointer.
        
        Args:
            config: Training configuration
            model: Model to checkpoint
        """
        self.config = config
        self.model = model
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Track checkpoints
        self.checkpoints = []
    
    def save_checkpoint(self, optimizer, scheduler, epoch: int, step: int, 
                       metrics: Dict[str, float], is_best: bool = False) -> str:
        """Save a model checkpoint.
        
        Args:
            optimizer: Training optimizer
            scheduler: Learning rate scheduler
            epoch: Current epoch
            step: Current step
            metrics: Current metrics
            is_best: Whether this is the best checkpoint so far
            
        Returns:
            Path to saved checkpoint
        """
        checkpoint = {
            'epoch': epoch,
            'step': step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'metrics': metrics,
            'config': asdict(self.config)
        }
        
        # Generate checkpoint filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        checkpoint_name = f"checkpoint_epoch_{epoch}_step_{step}_{timestamp}.pt"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)
        
        # Track checkpoint
        self.checkpoints.append({
            'path': checkpoint_path,
            'epoch': epoch,
            'step': step,
            'metrics': metrics,
            'is_best': is_best
        })
        
        # Save best checkpoint separately
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            shutil.copy(checkpoint_path, best_path)
        
        # Clean up old checkpoints
        self._cleanup_old_checkpoints()
        
        logger.info(f"Checkpoint saved: {checkpoint_path}")
        return str(checkpoint_path)
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """Load a model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Checkpoint data
        """
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.config.device)
            
            # Load model state
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            logger.info(f"Checkpoint loaded from {checkpoint_path}")
            return checkpoint
            
        except Exception as e:
            logger.error(f"Error loading checkpoint: {str(e)}")
            raise
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints to save disk space."""
        if len(self.checkpoints) <= self.config.save_total_limit:
            return
        
        # Sort by step (most recent first)
        self.checkpoints.sort(key=lambda x: x['step'], reverse=True)
        
        # Keep only the most recent checkpoints
        checkpoints_to_keep = self.checkpoints[:self.config.save_total_limit]
        checkpoints_to_remove = self.checkpoints[self.config.save_total_limit:]
        
        # Remove old checkpoints
        for checkpoint in checkpoints_to_remove:
            try:
                checkpoint['path'].unlink()
                logger.info(f"Removed old checkpoint: {checkpoint['path']}")
            except Exception as e:
                logger.warning(f"Could not remove checkpoint {checkpoint['path']}: {str(e)}")
        
        # Update tracking
        self.checkpoints = checkpoints_to_keep


class EarlyStopping:
    """Implements early stopping to prevent overfitting."""
    
    def __init__(self, patience: int = 5, min_delta: float = 0.001, mode: str = "min"):
        """Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: "min" or "max" for optimization direction
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.early_stop = False
    
    def __call__(self, current_score: float) -> bool:
        """Check if training should stop early.
        
        Args:
            current_score: Current metric value
            
        Returns:
            True if training should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = current_score
            return False
        
        if self.mode == "min":
            improved = current_score < self.best_score - self.min_delta
        else:
            improved = current_score > self.best_score + self.min_delta
        
        if improved:
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            self.early_stop = True
            return True
        
        return False


class ModelTrainer:
    """Main training class that orchestrates the training process."""
    
    def __init__(self, config: TrainingConfig, model: nn.Module, 
                 train_dataset: Dataset, val_dataset: Dataset = None):
        """Initialize model trainer.
        
        Args:
            config: Training configuration
            model: Model to train
            train_dataset: Training dataset
            val_dataset: Validation dataset (optional)
        """
        self.config = config
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        
        # Move model to device
        self.device = torch.device(config.device)
        self.model = self.model.to(self.device)
        
        # Initialize components
        self.metrics = TrainingMetrics(config)
        self.checkpointer = ModelCheckpointer(config, model)
        self.early_stopping = EarlyStopping(
            patience=config.early_stopping_patience,
            min_delta=config.early_stopping_min_delta
        )
        
        # Mixed precision training
        self.scaler = GradScaler() if config.use_fp16 else None
        
        # Initialize accelerator for distributed training
        if config.use_distributed:
            self.accelerator = Accelerator()
        else:
            self.accelerator = None
        
        # Training state
        self.current_epoch = 0
        self.current_step = 0
        self.best_val_loss = float('inf')
    
    def train(self) -> Dict[str, Any]:
        """Main training loop.
        
        Returns:
            Training results and metrics
        """
        try:
            logger.info("Starting training...")
            
            # Set up training components
            optimizer = self._setup_optimizer()
            scheduler = self._setup_scheduler(optimizer)
            train_dataloader = self._setup_dataloader(self.train_dataset, is_training=True)
            
            if self.val_dataset:
                val_dataloader = self._setup_dataloader(self.val_dataset, is_training=False)
            else:
                val_dataloader = None
            
            # Resume from checkpoint if specified
            if self.config.resume_from_checkpoint:
                self._resume_from_checkpoint(optimizer, scheduler)
            
            # Training loop
            for epoch in range(self.current_epoch, self.config.num_epochs):
                self.current_epoch = epoch
                
                # Training phase
                train_metrics = self._train_epoch(train_dataloader, optimizer, scheduler)
                
                # Validation phase
                if val_dataloader:
                    val_metrics = self._validate_epoch(val_dataloader)
                    
                    # Check early stopping
                    if self.early_stopping(val_metrics['val_loss']):
                        logger.info("Early stopping triggered")
                        break
                    
                    # Update best validation loss
                    if val_metrics['val_loss'] < self.best_val_loss:
                        self.best_val_loss = val_metrics['val_loss']
                        is_best = True
                    else:
                        is_best = False
                    
                    # Save checkpoint
                    self.checkpointer.save_checkpoint(
                        optimizer, scheduler, epoch, self.current_step,
                        {**train_metrics, **val_metrics}, is_best
                    )
                    
                    # Log metrics
                    self.metrics.log_metrics(
                        {**train_metrics, **val_metrics},
                        self.current_step, epoch
                    )
                else:
                    # Save checkpoint without validation
                    self.checkpointer.save_checkpoint(
                        optimizer, scheduler, epoch, self.current_step,
                        train_metrics, False
                    )
                    
                    # Log metrics
                    self.metrics.log_metrics(train_metrics, self.current_step, epoch)
            
            logger.info("Training completed successfully")
            return self.metrics.metrics
            
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise
    
    def _setup_optimizer(self):
        """Set up optimizer for training."""
        # Get trainable parameters
        trainable_params = []
        for param in self.model.parameters():
            if param.requires_grad:
                trainable_params.append(param)
        
        # Create optimizer
        optimizer = optim.AdamW(
            trainable_params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        return optimizer
    
    def _setup_scheduler(self, optimizer):
        """Set up learning rate scheduler."""
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=len(self.train_dataset) // self.config.batch_size * self.config.num_epochs,
            eta_min=self.config.learning_rate * 0.1
        )
        
        return scheduler
    
    def _setup_dataloader(self, dataset: Dataset, is_training: bool = True):
        """Set up data loader."""
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=is_training,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory
        )
        
        return dataloader
    
    def _train_epoch(self, dataloader: DataLoader, optimizer, scheduler) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        
        progress_bar = tqdm(dataloader, desc=f"Training Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = self._move_batch_to_device(batch)
            
            # Forward pass
            if self.config.use_fp16 and self.scaler:
                with autocast():
                    outputs = self.model(**batch)
                    loss = outputs['loss']
            else:
                outputs = self.model(**batch)
                loss = outputs['loss']
            
            # Backward pass
            if self.config.use_fp16 and self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.config.use_gradient_clipping:
                    if self.config.use_fp16 and self.scaler:
                        self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.max_grad_norm
                    )
                
                # Optimizer step
                if self.config.use_fp16 and self.scaler:
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    optimizer.step()
                
                optimizer.zero_grad()
                scheduler.step()
                
                self.current_step += 1
            
            # Update metrics
            total_loss += loss.item() * batch['input_ids'].size(0)
            total_samples += batch['input_ids'].size(0)
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{scheduler.get_last_lr()[0]:.2e}"
            })
        
        avg_loss = total_loss / total_samples
        return {'train_loss': avg_loss}
    
    def _validate_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Validation Epoch {self.current_epoch}"):
                # Move batch to device
                batch = self._move_batch_to_device(batch)
                
                # Forward pass
                outputs = self.model(**batch)
                loss = outputs['loss']
                
                # Update metrics
                total_loss += loss.item() * batch['input_ids'].size(0)
                total_samples += batch['input_ids'].size(0)
        
        avg_loss = total_loss / total_samples
        return {'val_loss': avg_loss}
    
    def _move_batch_to_device(self, batch):
        """Move batch tensors to the appropriate device."""
        if isinstance(batch, dict):
            return {key: self._move_batch_to_device(value) for key, value in batch.items()}
        elif isinstance(batch, (list, tuple)):
            return type(batch)(self._move_batch_to_device(item) for item in batch)
        elif isinstance(batch, torch.Tensor):
            return batch.to(self.device)
        else:
            return batch
    
    def _resume_from_checkpoint(self, optimizer, scheduler):
        """Resume training from a checkpoint."""
        try:
            checkpoint = self.checkpointer.load_checkpoint(self.config.resume_from_checkpoint)
            
            # Restore training state
            self.current_epoch = checkpoint['epoch']
            self.current_step = checkpoint['step']
            self.best_val_loss = checkpoint['metrics'].get('val_loss', float('inf'))
            
            # Restore optimizer and scheduler state
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if checkpoint['scheduler_state_dict'] and scheduler:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            logger.info(f"Resumed training from epoch {self.current_epoch}, step {self.current_step}")
            
        except Exception as e:
            logger.error(f"Error resuming from checkpoint: {str(e)}")
            raise
