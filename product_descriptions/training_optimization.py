from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import asyncio
import json
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from functools import partial
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import (
from torch.utils.data import DataLoader
import structlog
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
            from torch.utils.tensorboard import SummaryWriter
            import wandb
from typing import Any, List, Dict, Optional
"""
Training Optimization System for Cybersecurity Applications

This module provides comprehensive training optimization capabilities including:
- Early stopping with multiple strategies
- Learning rate scheduling with various algorithms
- Gradient clipping and normalization
- Training monitoring and checkpointing
- Model validation and selection
- Performance optimization techniques
"""


    StepLR, MultiStepLR, ExponentialLR, CosineAnnealingLR,
    CosineAnnealingWarmRestarts, ReduceLROnPlateau, OneCycleLR,
    CyclicLR, ChainedScheduler, SequentialLR
)

# Configure structured logging
logger = structlog.get_logger(__name__)


class EarlyStoppingMode(Enum):
    """Early stopping modes."""
    MIN = "min"  # Stop when metric stops decreasing
    MAX = "max"  # Stop when metric stops increasing


class LRSchedulerType(Enum):
    """Learning rate scheduler types."""
    STEP = "step"
    MULTI_STEP = "multi_step"
    EXPONENTIAL = "exponential"
    COSINE_ANNEALING = "cosine_annealing"
    COSINE_ANNEALING_WARM_RESTARTS = "cosine_annealing_warm_restarts"
    REDUCE_ON_PLATEAU = "reduce_on_plateau"
    ONE_CYCLE = "one_cycle"
    CYCLIC = "cyclic"
    CUSTOM = "custom"


@dataclass
class EarlyStoppingConfig:
    """Configuration for early stopping."""
    mode: EarlyStoppingMode = EarlyStoppingMode.MIN
    patience: int = 10
    min_delta: float = 0.0
    restore_best_weights: bool = True
    monitor: str = "val_loss"
    verbose: bool = True
    save_path: Optional[str] = None
    
    # Advanced options
    min_epochs: int = 0  # Minimum epochs before early stopping
    max_epochs: int = 1000  # Maximum epochs
    baseline: Optional[float] = None  # Baseline value to compare against
    cooldown: int = 0  # Cooldown period after reducing LR
    min_lr: float = 0.0  # Minimum learning rate


@dataclass
class LRSchedulerConfig:
    """Configuration for learning rate scheduling."""
    scheduler_type: LRSchedulerType = LRSchedulerType.REDUCE_ON_PLATEAU
    initial_lr: float = 1e-3
    min_lr: float = 1e-6
    max_lr: float = 1e-2
    
    # Step-based schedulers
    step_size: int = 30
    gamma: float = 0.1
    milestones: List[int] = field(default_factory=lambda: [30, 60, 90])
    
    # Cosine annealing
    T_max: int = 100
    eta_min: float = 0.0
    T_0: int = 10
    T_mult: int = 2
    
    # Reduce on plateau
    mode: str = "min"
    factor: float = 0.1
    patience: int = 10
    threshold: float = 1e-4
    threshold_mode: str = "rel"
    cooldown: int = 0
    
    # One cycle
    epochs: int = 100
    steps_per_epoch: int = 100
    pct_start: float = 0.3
    anneal_strategy: str = "cos"
    cycle_momentum: bool = True
    base_momentum: float = 0.85
    max_momentum: float = 0.95
    div_factor: float = 25.0
    final_div_factor: float = 1e4
    
    # Cyclic
    base_lr: float = 1e-6
    max_lr: float = 1e-3
    step_size_up: int = 2000
    step_size_down: int = 2000
    mode_cyclic: str = "triangular"
    scale_fn: Optional[Callable] = None
    scale_mode: str = "cycle"
    
    # Custom scheduler
    custom_scheduler_fn: Optional[Callable] = None


@dataclass
class TrainingOptimizationConfig:
    """Configuration for training optimization."""
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    lr_scheduler: LRSchedulerConfig = field(default_factory=LRSchedulerConfig)
    
    # Gradient optimization
    gradient_clip_norm: float = 1.0
    gradient_clip_value: Optional[float] = None
    gradient_accumulation_steps: int = 1
    
    # Training monitoring
    monitor_interval: int = 1
    save_checkpoints: bool = True
    checkpoint_dir: str = "./checkpoints"
    save_best_only: bool = True
    save_last: bool = True
    
    # Validation
    validation_frequency: int = 1
    validation_metrics: List[str] = field(default_factory=lambda: ["loss", "accuracy", "f1"])
    
    # Performance optimization
    mixed_precision: bool = True
    compile_model: bool = False
    use_amp: bool = True
    
    # Logging
    log_interval: int = 10
    tensorboard_logging: bool = True
    wandb_logging: bool = False


class EarlyStopping:
    """Early stopping implementation with multiple strategies."""
    
    def __init__(self, config: EarlyStoppingConfig):
        
    """__init__ function."""
self.config = config
        self.best_score = None
        self.best_epoch = 0
        self.counter = 0
        self.best_weights = None
        self.training_history = []
        
        # Initialize baseline
        if config.baseline is not None:
            self.best_score = config.baseline
        
        logger.info("Early stopping initialized", config=config.__dict__)
    
    def __call__(self, epoch: int, metric: float, model: nn.Module) -> bool:
        """Check if training should stop early."""
        self.training_history.append({
            'epoch': epoch,
            'metric': metric,
            'lr': self._get_current_lr(model)
        })
        
        # Check minimum epochs
        if epoch < self.config.min_epochs:
            return False
        
        # Check maximum epochs
        if epoch >= self.config.max_epochs:
            if self.config.verbose:
                logger.info(f"Stopping: reached maximum epochs ({self.config.max_epochs})")
            return True
        
        # Determine if metric improved
        improved = self._is_improved(metric)
        
        if improved:
            self.best_score = metric
            self.best_epoch = epoch
            self.counter = 0
            
            # Save best weights
            if self.config.restore_best_weights:
                self.best_weights = self._get_model_state(model)
            
            # Save checkpoint
            if self.config.save_path:
                self._save_checkpoint(model, epoch, metric)
                
        else:
            self.counter += 1
            
            # Check cooldown
            if self.counter <= self.config.cooldown:
                return False
        
        # Check patience
        if self.counter >= self.config.patience:
            if self.config.verbose:
                logger.info(f"Early stopping triggered after {self.config.patience} epochs without improvement")
                logger.info(f"Best {self.config.monitor}: {self.best_score:.6f} at epoch {self.best_epoch}")
            
            # Restore best weights
            if self.config.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
                logger.info("Restored best model weights")
            
            return True
        
        return False
    
    def _is_improved(self, metric: float) -> bool:
        """Check if metric has improved."""
        if self.best_score is None:
            return True
        
        if self.config.mode == EarlyStoppingMode.MIN:
            return metric < self.best_score - self.config.min_delta
        else:
            return metric > self.best_score + self.config.min_delta
    
    def _get_current_lr(self, model: nn.Module) -> float:
        """Get current learning rate."""
        for param_group in model.optimizer.param_groups:
            return param_group['lr']
        return 0.0
    
    def _get_model_state(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        """Get model state dict."""
        return model.state_dict().copy()
    
    def _save_checkpoint(self, model: nn.Module, epoch: int, metric: float):
        """Save model checkpoint."""
        checkpoint_path = Path(self.config.save_path)
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': model.optimizer.state_dict(),
            'metric': metric,
            'config': self.config.__dict__
        }
        
        torch.save(checkpoint, checkpoint_path / f"best_model_epoch_{epoch}.pt")
        logger.info(f"Saved checkpoint: {checkpoint_path / f'best_model_epoch_{epoch}.pt'}")
    
    def get_training_history(self) -> List[Dict[str, Any]]:
        """Get training history."""
        return self.training_history
    
    def plot_training_history(self, save_path: Optional[str] = None):
        """Plot training history."""
        if not self.training_history:
            logger.warning("No training history to plot")
            return
        
        epochs = [h['epoch'] for h in self.training_history]
        metrics = [h['metric'] for h in self.training_history]
        lrs = [h['lr'] for h in self.training_history]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot metric
        ax1.plot(epochs, metrics, 'b-', label=f'{self.config.monitor}')
        ax1.axhline(y=self.best_score, color='r', linestyle='--', label='Best')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel(self.config.monitor)
        ax1.set_title(f'{self.config.monitor} vs Epoch')
        ax1.legend()
        ax1.grid(True)
        
        # Plot learning rate
        ax2.plot(epochs, lrs, 'g-', label='Learning Rate')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Learning Rate')
        ax2.set_title('Learning Rate vs Epoch')
        ax2.legend()
        ax2.grid(True)
        ax2.set_yscale('log')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training history plot saved to: {save_path}")
        
        plt.show()


class LRSchedulerFactory:
    """Factory for creating learning rate schedulers."""
    
    @staticmethod
    def create_scheduler(
        optimizer: optim.Optimizer,
        config: LRSchedulerConfig,
        dataloader: Optional[DataLoader] = None
    ) -> Union[torch.optim.lr_scheduler._LRScheduler, Any]:
        """Create learning rate scheduler based on configuration."""
        
        if config.scheduler_type == LRSchedulerType.STEP:
            return StepLR(
                optimizer,
                step_size=config.step_size,
                gamma=config.gamma
            )
        
        elif config.scheduler_type == LRSchedulerType.MULTI_STEP:
            return MultiStepLR(
                optimizer,
                milestones=config.milestones,
                gamma=config.gamma
            )
        
        elif config.scheduler_type == LRSchedulerType.EXPONENTIAL:
            return ExponentialLR(
                optimizer,
                gamma=config.gamma
            )
        
        elif config.scheduler_type == LRSchedulerType.COSINE_ANNEALING:
            return CosineAnnealingLR(
                optimizer,
                T_max=config.T_max,
                eta_min=config.eta_min
            )
        
        elif config.scheduler_type == LRSchedulerType.COSINE_ANNEALING_WARM_RESTARTS:
            return CosineAnnealingWarmRestarts(
                optimizer,
                T_0=config.T_0,
                T_mult=config.T_mult,
                eta_min=config.eta_min
            )
        
        elif config.scheduler_type == LRSchedulerType.REDUCE_ON_PLATEAU:
            return ReduceLROnPlateau(
                optimizer,
                mode=config.mode,
                factor=config.factor,
                patience=config.patience,
                threshold=config.threshold,
                threshold_mode=config.threshold_mode,
                cooldown=config.cooldown,
                min_lr=config.min_lr,
                verbose=True
            )
        
        elif config.scheduler_type == LRSchedulerType.ONE_CYCLE:
            if dataloader is None:
                raise ValueError("DataLoader required for OneCycleLR")
            
            return OneCycleLR(
                optimizer,
                max_lr=config.max_lr,
                epochs=config.epochs,
                steps_per_epoch=len(dataloader),
                pct_start=config.pct_start,
                anneal_strategy=config.anneal_strategy,
                cycle_momentum=config.cycle_momentum,
                base_momentum=config.base_momentum,
                max_momentum=config.max_momentum,
                div_factor=config.div_factor,
                final_div_factor=config.final_div_factor
            )
        
        elif config.scheduler_type == LRSchedulerType.CYCLIC:
            return CyclicLR(
                optimizer,
                base_lr=config.base_lr,
                max_lr=config.max_lr,
                step_size_up=config.step_size_up,
                step_size_down=config.step_size_down,
                mode=config.mode_cyclic,
                scale_fn=config.scale_fn,
                scale_mode=config.scale_mode
            )
        
        elif config.scheduler_type == LRSchedulerType.CUSTOM:
            if config.custom_scheduler_fn is None:
                raise ValueError("Custom scheduler function required")
            return config.custom_scheduler_fn(optimizer)
        
        else:
            raise ValueError(f"Unsupported scheduler type: {config.scheduler_type}")


class GradientOptimizer:
    """Gradient optimization utilities."""
    
    def __init__(self, config: TrainingOptimizationConfig):
        
    """__init__ function."""
self.config = config
        self.gradient_norms = []
        self.clipped_gradients = 0
    
    def clip_gradients(self, model: nn.Module):
        """Clip gradients based on configuration."""
        if self.config.gradient_clip_norm is not None:
            # Clip by norm
            total_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                self.config.gradient_clip_norm
            )
            self.gradient_norms.append(total_norm.item())
            
            if total_norm > self.config.gradient_clip_norm:
                self.clipped_gradients += 1
        
        elif self.config.gradient_clip_value is not None:
            # Clip by value
            torch.nn.utils.clip_grad_value_(
                model.parameters(),
                self.config.gradient_clip_value
            )
    
    def get_gradient_stats(self) -> Dict[str, float]:
        """Get gradient statistics."""
        if not self.gradient_norms:
            return {}
        
        return {
            "avg_gradient_norm": np.mean(self.gradient_norms),
            "max_gradient_norm": np.max(self.gradient_norms),
            "min_gradient_norm": np.min(self.gradient_norms),
            "clipped_gradients_ratio": self.clipped_gradients / len(self.gradient_norms)
        }


class TrainingMonitor:
    """Training monitoring and logging."""
    
    def __init__(self, config: TrainingOptimizationConfig):
        
    """__init__ function."""
self.config = config
        self.metrics_history = {
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': [],
            'learning_rate': [],
            'epoch_time': []
        }
        self.start_time = time.time()
        
        # Setup logging
        if config.tensorboard_logging:
            self.writer = SummaryWriter(log_dir="./logs/tensorboard")
        else:
            self.writer = None
        
        if config.wandb_logging:
            wandb.init(project="cybersecurity-training")
            self.wandb = wandb
        else:
            self.wandb = None
    
    def log_metrics(self, epoch: int, metrics: Dict[str, float], learning_rate: float):
        """Log training metrics."""
        # Store metrics
        for key, value in metrics.items():
            if key in self.metrics_history:
                self.metrics_history[key].append(value)
        
        self.metrics_history['learning_rate'].append(learning_rate)
        
        # Log to tensorboard
        if self.writer:
            for key, value in metrics.items():
                self.writer.add_scalar(key, value, epoch)
            self.writer.add_scalar('learning_rate', learning_rate, epoch)
        
        # Log to wandb
        if self.wandb:
            log_dict = {**metrics, 'learning_rate': learning_rate, 'epoch': epoch}
            self.wandb.log(log_dict)
        
        # Console logging
        if epoch % self.config.log_interval == 0:
            metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
            logger.info(f"Epoch {epoch} | {metrics_str} | LR: {learning_rate:.6f}")
    
    def log_epoch_time(self, epoch: int, epoch_time: float):
        """Log epoch time."""
        self.metrics_history['epoch_time'].append(epoch_time)
        
        if self.writer:
            self.writer.add_scalar('epoch_time', epoch_time, epoch)
    
    def plot_training_curves(self, save_path: Optional[str] = None):
        """Plot training curves."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss curves
        if self.metrics_history['train_loss'] and self.metrics_history['val_loss']:
            epochs = range(len(self.metrics_history['train_loss']))
            axes[0, 0].plot(epochs, self.metrics_history['train_loss'], 'b-', label='Train Loss')
            axes[0, 0].plot(epochs, self.metrics_history['val_loss'], 'r-', label='Val Loss')
            axes[0, 0].set_title('Loss vs Epoch')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
        
        # Accuracy curves
        if self.metrics_history['train_accuracy'] and self.metrics_history['val_accuracy']:
            epochs = range(len(self.metrics_history['train_accuracy']))
            axes[0, 1].plot(epochs, self.metrics_history['train_accuracy'], 'b-', label='Train Accuracy')
            axes[0, 1].plot(epochs, self.metrics_history['val_accuracy'], 'r-', label='Val Accuracy')
            axes[0, 1].set_title('Accuracy vs Epoch')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Accuracy')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
        
        # Learning rate curve
        if self.metrics_history['learning_rate']:
            epochs = range(len(self.metrics_history['learning_rate']))
            axes[1, 0].plot(epochs, self.metrics_history['learning_rate'], 'g-')
            axes[1, 0].set_title('Learning Rate vs Epoch')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].set_yscale('log')
            axes[1, 0].grid(True)
        
        # Epoch time curve
        if self.metrics_history['epoch_time']:
            epochs = range(len(self.metrics_history['epoch_time']))
            axes[1, 1].plot(epochs, self.metrics_history['epoch_time'], 'm-')
            axes[1, 1].set_title('Epoch Time vs Epoch')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Time (seconds)')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training curves saved to: {save_path}")
        
        plt.show()
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get training summary."""
        total_time = time.time() - self.start_time
        
        summary = {
            "total_training_time": total_time,
            "total_epochs": len(self.metrics_history['train_loss']),
            "avg_epoch_time": np.mean(self.metrics_history['epoch_time']) if self.metrics_history['epoch_time'] else 0,
            "final_train_loss": self.metrics_history['train_loss'][-1] if self.metrics_history['train_loss'] else None,
            "final_val_loss": self.metrics_history['val_loss'][-1] if self.metrics_history['val_loss'] else None,
            "best_val_loss": min(self.metrics_history['val_loss']) if self.metrics_history['val_loss'] else None,
            "final_train_accuracy": self.metrics_history['train_accuracy'][-1] if self.metrics_history['train_accuracy'] else None,
            "final_val_accuracy": self.metrics_history['val_accuracy'][-1] if self.metrics_history['val_accuracy'] else None,
            "best_val_accuracy": max(self.metrics_history['val_accuracy']) if self.metrics_history['val_accuracy'] else None
        }
        
        return summary
    
    def close(self) -> Any:
        """Close logging connections."""
        if self.writer:
            self.writer.close()
        if self.wandb:
            self.wandb.finish()


class OptimizedTrainer:
    """Optimized trainer with early stopping and learning rate scheduling."""
    
    def __init__(self, config: TrainingOptimizationConfig):
        
    """__init__ function."""
self.config = config
        self.early_stopping = EarlyStopping(config.early_stopping)
        self.gradient_optimizer = GradientOptimizer(config)
        self.monitor = TrainingMonitor(config)
        
        # Create checkpoint directory
        if config.save_checkpoints:
            Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info("Optimized trainer initialized", config=config.__dict__)
    
    async def train(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        num_epochs: int,
        device: torch.device
    ) -> Dict[str, Any]:
        """Train model with optimization features."""
        
        # Create learning rate scheduler
        scheduler = LRSchedulerFactory.create_scheduler(
            optimizer, self.config.lr_scheduler, train_dataloader
        )
        
        # Setup model
        model.to(device)
        model.train()
        
        # Store optimizer reference for early stopping
        model.optimizer = optimizer
        
        # Training loop
        best_val_loss = float('inf')
        best_model_state = None
        
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            
            # Training phase
            train_metrics = await self._train_epoch(
                model, train_dataloader, optimizer, criterion, device
            )
            
            # Validation phase
            if epoch % self.config.validation_frequency == 0:
                val_metrics = await self._validate_epoch(
                    model, val_dataloader, criterion, device
                )
                
                # Update learning rate scheduler
                if isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(val_metrics['val_loss'])
                else:
                    scheduler.step()
                
                # Check early stopping
                if self.early_stopping(epoch, val_metrics['val_loss'], model):
                    logger.info(f"Training stopped early at epoch {epoch}")
                    break
                
                # Save best model
                if val_metrics['val_loss'] < best_val_loss:
                    best_val_loss = val_metrics['val_loss']
                    best_model_state = model.state_dict().copy()
                    
                    if self.config.save_checkpoints and self.config.save_best_only:
                        self._save_checkpoint(model, optimizer, epoch, val_metrics, "best")
            
            # Log metrics
            current_lr = optimizer.param_groups[0]['lr']
            self.monitor.log_metrics(epoch, {**train_metrics, **val_metrics}, current_lr)
            
            # Log epoch time
            epoch_time = time.time() - epoch_start_time
            self.monitor.log_epoch_time(epoch, epoch_time)
            
            # Save last checkpoint
            if self.config.save_checkpoints and self.config.save_last:
                self._save_checkpoint(model, optimizer, epoch, val_metrics, "last")
        
        # Restore best model if not already restored by early stopping
        if best_model_state is not None and not self.config.early_stopping.restore_best_weights:
            model.load_state_dict(best_model_state)
            logger.info("Restored best model weights")
        
        # Get training summary
        summary = self.monitor.get_training_summary()
        summary['gradient_stats'] = self.gradient_optimizer.get_gradient_stats()
        summary['early_stopping_history'] = self.early_stopping.get_training_history()
        
        # Close monitoring
        self.monitor.close()
        
        return summary
    
    async def _train_epoch(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        device: torch.device
    ) -> Dict[str, float]:
        """Train for one epoch."""
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            
            # Backward pass
            loss.backward()
            
            # Gradient optimization
            self.gradient_optimizer.clip_gradients(model)
            
            # Gradient accumulation
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            # Calculate metrics
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            total_correct += pred.eq(target.view_as(pred)).sum().item()
            total_samples += target.size(0)
        
        return {
            'train_loss': total_loss / len(dataloader),
            'train_accuracy': total_correct / total_samples
        }
    
    async def _validate_epoch(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        criterion: nn.Module,
        device: torch.device
    ) -> Dict[str, float]:
        """Validate for one epoch."""
        model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(device), target.to(device)
                
                output = model(data)
                loss = criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                total_correct += pred.eq(target.view_as(pred)).sum().item()
                total_samples += target.size(0)
                
                all_predictions.extend(pred.cpu().numpy().flatten())
                all_targets.extend(target.cpu().numpy())
        
        # Calculate additional metrics
        accuracy = total_correct / total_samples
        f1 = f1_score(all_targets, all_predictions, average='weighted')
        precision = precision_score(all_targets, all_predictions, average='weighted')
        recall = recall_score(all_targets, all_predictions, average='weighted')
        
        return {
            'val_loss': total_loss / len(dataloader),
            'val_accuracy': accuracy,
            'val_f1': f1,
            'val_precision': precision,
            'val_recall': recall
        }
    
    def _save_checkpoint(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        epoch: int,
        metrics: Dict[str, float],
        checkpoint_type: str
    ):
        """Save model checkpoint."""
        checkpoint_path = Path(self.config.checkpoint_dir)
        checkpoint_file = checkpoint_path / f"{checkpoint_type}_model_epoch_{epoch}.pt"
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config.__dict__
        }
        
        torch.save(checkpoint, checkpoint_file)
        logger.info(f"Saved {checkpoint_type} checkpoint: {checkpoint_file}")
    
    def plot_training_analysis(self, save_path: Optional[str] = None):
        """Plot comprehensive training analysis."""
        # Plot training curves
        self.monitor.plot_training_curves(save_path)
        
        # Plot early stopping analysis
        self.early_stopping.plot_training_history(
            save_path.replace('.png', '_early_stopping.png') if save_path else None
        )


# Utility functions
def create_optimized_trainer(
    early_stopping_patience: int = 10,
    lr_scheduler_type: LRSchedulerType = LRSchedulerType.REDUCE_ON_PLATEAU,
    initial_lr: float = 1e-3,
    **kwargs
) -> OptimizedTrainer:
    """Create an optimized trainer with default configurations."""
    
    early_stopping_config = EarlyStoppingConfig(
        patience=early_stopping_patience,
        monitor="val_loss",
        restore_best_weights=True
    )
    
    lr_scheduler_config = LRSchedulerConfig(
        scheduler_type=lr_scheduler_type,
        initial_lr=initial_lr,
        min_lr=1e-6
    )
    
    if lr_scheduler_type == LRSchedulerType.REDUCE_ON_PLATEAU:
        lr_scheduler_config.patience = early_stopping_patience // 2
        lr_scheduler_config.factor = 0.5
    
    config = TrainingOptimizationConfig(
        early_stopping=early_stopping_config,
        lr_scheduler=lr_scheduler_config,
        **kwargs
    )
    
    return OptimizedTrainer(config)


def load_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    checkpoint_path: str
) -> Dict[str, Any]:
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    return checkpoint


# Example usage
if __name__ == "__main__":
    # Example: Create and use optimized trainer
    trainer = create_optimized_trainer(
        early_stopping_patience=15,
        lr_scheduler_type=LRSchedulerType.REDUCE_ON_PLATEAU,
        initial_lr=1e-3,
        gradient_clip_norm=1.0,
        save_checkpoints=True
    )
    
    # This would be used in your training loop
    # summary = await trainer.train(model, train_loader, val_loader, optimizer, criterion, num_epochs, device)
    # trainer.plot_training_analysis("training_analysis.png") 