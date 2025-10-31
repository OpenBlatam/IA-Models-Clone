from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import (
from torch.cuda.amp import GradScaler
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from dataclasses import dataclass, field
import logging
import time
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import deque
import warnings
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
from typing import Any, List, Dict, Optional
import asyncio
#!/usr/bin/env python3
"""
Early Stopping and Learning Rate Scheduling Framework for SEO Deep Learning System
- Advanced early stopping strategies
- Multiple learning rate scheduling algorithms
- Comprehensive monitoring and logging
- Integration with training framework
"""

    StepLR, CosineAnnealingLR, CosineAnnealingWarmRestarts, ReduceLROnPlateau,
    ExponentialLR, MultiStepLR, OneCycleLR, LambdaLR, ChainedScheduler
)
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class EarlyStoppingConfig:
    """Configuration for early stopping"""
    # Basic early stopping
    enabled: bool = True
    patience: int = 10
    min_delta: float = 1e-4
    mode: str = "min"  # "min" for loss, "max" for accuracy
    monitor: str = "val_loss"  # "val_loss", "val_accuracy", "train_loss"
    
    # Advanced early stopping
    restore_best_weights: bool = True
    save_checkpoint: bool = True
    checkpoint_path: str = "./checkpoints/best_model.pth"
    
    # Multiple metric monitoring
    monitor_multiple: bool = False
    monitors: List[str] = field(default_factory=lambda: ["val_loss", "val_accuracy"])
    monitor_weights: List[float] = field(default_factory=lambda: [1.0, 1.0])
    
    # Adaptive early stopping
    adaptive_patience: bool = False
    patience_factor: float = 1.5
    min_patience: int = 5
    max_patience: int = 50
    
    # Plateau detection
    plateau_detection: bool = False
    plateau_window: int = 5
    plateau_threshold: float = 1e-3
    
    # Overfitting detection
    overfitting_detection: bool = False
    train_val_gap_threshold: float = 0.1
    overfitting_patience: int = 5
    
    # Logging
    verbose: bool = True
    log_interval: int = 1

@dataclass
class LRSchedulerConfig:
    """Configuration for learning rate scheduling"""
    # Scheduler type
    scheduler_type: str = "cosine"  # "step", "cosine", "plateau", "exponential", "multistep", "onecycle", "warmup_cosine"
    
    # Basic parameters
    initial_lr: float = 1e-3
    min_lr: float = 1e-6
    max_lr: float = 1e-2
    
    # Step scheduler
    step_size: int = 30
    gamma: float = 0.1
    
    # Cosine scheduler
    T_max: int = 100
    eta_min: float = 0.0
    
    # Cosine with warm restarts
    T_0: int = 10
    T_mult: int = 2
    
    # Plateau scheduler
    mode: str = "min"
    factor: float = 0.5
    patience: int = 5
    threshold: float = 1e-4
    threshold_mode: str = "rel"
    cooldown: int = 0
    
    # Exponential scheduler
    decay_rate: float = 0.95
    
    # Multi-step scheduler
    milestones: List[int] = field(default_factory=lambda: [30, 60, 90])
    
    # OneCycle scheduler
    epochs: int = 100
    steps_per_epoch: int = 100
    pct_start: float = 0.3
    anneal_strategy: str = "cos"
    cycle_momentum: bool = True
    base_momentum: float = 0.85
    max_momentum: float = 0.95
    div_factor: float = 25.0
    final_div_factor: float = 1e4
    
    # Warmup cosine
    warmup_steps: int = 1000
    warmup_start_lr: float = 1e-6
    
    # Custom scheduler
    custom_lr_fn: Optional[Callable] = None
    
    # Logging
    verbose: bool = True
    log_interval: int = 10

@dataclass
class TrainingMetrics:
    """Container for training metrics"""
    epoch: int = 0
    train_loss: float = 0.0
    val_loss: float = 0.0
    train_accuracy: float = 0.0
    val_accuracy: float = 0.0
    learning_rate: float = 0.0
    timestamp: float = field(default_factory=time.time)

class EarlyStopping:
    """Advanced early stopping with multiple strategies"""
    
    def __init__(self, config: EarlyStoppingConfig):
        
    """__init__ function."""
self.config = config
        self.best_score = None
        self.best_epoch = 0
        self.counter = 0
        self.best_weights = None
        self.history = []
        
        # Multiple metric monitoring
        if config.monitor_multiple:
            self.best_scores = {monitor: None for monitor in config.monitors}
            self.counters = {monitor: 0 for monitor in config.monitors}
        
        # Adaptive patience
        self.current_patience = config.patience
        
        # Plateau detection
        if config.plateau_detection:
            self.plateau_history = deque(maxlen=config.plateau_window)
        
        # Overfitting detection
        if config.overfitting_detection:
            self.overfitting_counter = 0
            self.train_val_gaps = deque(maxlen=config.overfitting_patience)
        
        logger.info(f"Early stopping initialized with patience={config.patience}, monitor={config.monitor}")
    
    def __call__(self, metrics: TrainingMetrics, model: nn.Module) -> bool:
        """Check if training should stop"""
        self.history.append(metrics)
        
        # Get current score
        current_score = self._get_monitored_value(metrics)
        
        # Check if this is the best score
        is_best = self._is_best_score(current_score)
        
        if is_best:
            self.best_score = current_score
            self.best_epoch = metrics.epoch
            self.counter = 0
            
            # Save best weights
            if self.config.restore_best_weights:
                self.best_weights = {name: param.clone().detach() for name, param in model.state_dict().items()}
            
            # Save checkpoint
            if self.config.save_checkpoint:
                self._save_checkpoint(model, metrics)
            
            if self.config.verbose:
                logger.info(f"New best score: {current_score:.6f} at epoch {metrics.epoch}")
        
        else:
            self.counter += 1
            
            # Adaptive patience
            if self.config.adaptive_patience and self.counter > self.current_patience // 2:
                self.current_patience = min(
                    int(self.current_patience * self.config.patience_factor),
                    self.config.max_patience
                )
                if self.config.verbose:
                    logger.info(f"Adaptive patience increased to {self.current_patience}")
        
        # Multiple metric monitoring
        if self.config.monitor_multiple:
            self._update_multiple_metrics(metrics)
        
        # Plateau detection
        if self.config.plateau_detection:
            plateau_detected = self._detect_plateau(current_score)
            if plateau_detected and self.config.verbose:
                logger.info(f"Plateau detected at epoch {metrics.epoch}")
        
        # Overfitting detection
        if self.config.overfitting_detection:
            overfitting_detected = self._detect_overfitting(metrics)
            if overfitting_detected and self.config.verbose:
                logger.info(f"Overfitting detected at epoch {metrics.epoch}")
        
        # Check if should stop
        should_stop = self.counter >= self.current_patience
        
        if should_stop and self.config.verbose:
            logger.info(f"Early stopping triggered at epoch {metrics.epoch}")
            logger.info(f"Best score: {self.best_score:.6f} at epoch {self.best_epoch}")
        
        return should_stop
    
    def _get_monitored_value(self, metrics: TrainingMetrics) -> float:
        """Get the monitored value from metrics"""
        if hasattr(metrics, self.config.monitor):
            return getattr(metrics, self.config.monitor)
        else:
            raise ValueError(f"Monitor '{self.config.monitor}' not found in metrics")
    
    def _is_best_score(self, current_score: float) -> bool:
        """Check if current score is the best"""
        if self.best_score is None:
            return True
        
        if self.config.mode == "min":
            return current_score < self.best_score - self.config.min_delta
        else:  # mode == "max"
            return current_score > self.best_score + self.config.min_delta
    
    def _update_multiple_metrics(self, metrics: TrainingMetrics):
        """Update multiple metric monitoring"""
        for i, monitor in enumerate(self.config.monitors):
            if hasattr(metrics, monitor):
                current_value = getattr(metrics, monitor)
                
                if self.best_scores[monitor] is None:
                    self.best_scores[monitor] = current_value
                    self.counters[monitor] = 0
                else:
                    if self.config.mode == "min":
                        if current_value < self.best_scores[monitor] - self.config.min_delta:
                            self.best_scores[monitor] = current_value
                            self.counters[monitor] = 0
                        else:
                            self.counters[monitor] += 1
                    else:
                        if current_value > self.best_scores[monitor] + self.config.min_delta:
                            self.best_scores[monitor] = current_value
                            self.counters[monitor] = 0
                        else:
                            self.counters[monitor] += 1
    
    def _detect_plateau(self, current_score: float) -> bool:
        """Detect if training has plateaued"""
        self.plateau_history.append(current_score)
        
        if len(self.plateau_history) < self.config.plateau_window:
            return False
        
        # Calculate variance of recent scores
        recent_scores = list(self.plateau_history)
        variance = np.var(recent_scores)
        
        return variance < self.config.plateau_threshold
    
    def _detect_overfitting(self, metrics: TrainingMetrics) -> bool:
        """Detect overfitting based on train-val gap"""
        if hasattr(metrics, 'train_accuracy') and hasattr(metrics, 'val_accuracy'):
            gap = metrics.train_accuracy - metrics.val_accuracy
            self.train_val_gaps.append(gap)
            
            if len(self.train_val_gaps) >= self.config.overfitting_patience:
                avg_gap = np.mean(list(self.train_val_gaps))
                if avg_gap > self.config.train_val_gap_threshold:
                    self.overfitting_counter += 1
                    return self.overfitting_counter >= self.config.overfitting_patience
        
        return False
    
    def _save_checkpoint(self, model: nn.Module, metrics: TrainingMetrics):
        """Save model checkpoint"""
        os.makedirs(os.path.dirname(self.config.checkpoint_path), exist_ok=True)
        
        checkpoint = {
            'epoch': metrics.epoch,
            'model_state_dict': model.state_dict(),
            'best_score': self.best_score,
            'best_epoch': self.best_epoch,
            'metrics': metrics
        }
        
        torch.save(checkpoint, self.config.checkpoint_path)
        
        if self.config.verbose:
            logger.info(f"Checkpoint saved: {self.config.checkpoint_path}")
    
    def restore_best_weights(self, model: nn.Module):
        """Restore best weights to model"""
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)
            logger.info(f"Best weights restored from epoch {self.best_epoch}")
        else:
            logger.warning("No best weights to restore")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get early stopping summary"""
        return {
            'best_score': self.best_score,
            'best_epoch': self.best_epoch,
            'total_epochs': len(self.history),
            'stopped_early': len(self.history) > self.best_epoch + self.config.patience,
            'history': self.history
        }

class AdvancedLRScheduler:
    """Advanced learning rate scheduler with multiple strategies"""
    
    def __init__(self, optimizer: optim.Optimizer, config: LRSchedulerConfig):
        
    """__init__ function."""
self.optimizer = optimizer
        self.config = config
        self.scheduler = None
        self.history = []
        self.current_epoch = 0
        
        self._create_scheduler()
        
        logger.info(f"LR scheduler initialized: {config.scheduler_type}")
    
    def _create_scheduler(self) -> Any:
        """Create the learning rate scheduler"""
        if self.config.scheduler_type == "step":
            self.scheduler = StepLR(
                self.optimizer,
                step_size=self.config.step_size,
                gamma=self.config.gamma
            )
        
        elif self.config.scheduler_type == "cosine":
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.T_max,
                eta_min=self.config.eta_min
            )
        
        elif self.config.scheduler_type == "cosine_warm_restarts":
            self.scheduler = CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=self.config.T_0,
                T_mult=self.config.T_mult,
                eta_min=self.config.eta_min
            )
        
        elif self.config.scheduler_type == "plateau":
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode=self.config.mode,
                factor=self.config.factor,
                patience=self.config.patience,
                threshold=self.config.threshold,
                threshold_mode=self.config.threshold_mode,
                cooldown=self.config.cooldown,
                min_lr=self.config.min_lr,
                verbose=self.config.verbose
            )
        
        elif self.config.scheduler_type == "exponential":
            self.scheduler = ExponentialLR(
                self.optimizer,
                gamma=self.config.decay_rate
            )
        
        elif self.config.scheduler_type == "multistep":
            self.scheduler = MultiStepLR(
                self.optimizer,
                milestones=self.config.milestones,
                gamma=self.config.gamma
            )
        
        elif self.config.scheduler_type == "onecycle":
            total_steps = self.config.epochs * self.config.steps_per_epoch
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=self.config.max_lr,
                total_steps=total_steps,
                pct_start=self.config.pct_start,
                anneal_strategy=self.config.anneal_strategy,
                cycle_momentum=self.config.cycle_momentum,
                base_momentum=self.config.base_momentum,
                max_momentum=self.config.max_momentum,
                div_factor=self.config.div_factor,
                final_div_factor=self.config.final_div_factor
            )
        
        elif self.config.scheduler_type == "warmup_cosine":
            # Custom warmup cosine scheduler
            self.scheduler = self._create_warmup_cosine_scheduler()
        
        elif self.config.scheduler_type == "custom":
            if self.config.custom_lr_fn is not None:
                self.scheduler = LambdaLR(self.optimizer, self.config.custom_lr_fn)
            else:
                raise ValueError("Custom scheduler requires custom_lr_fn")
        
        else:
            raise ValueError(f"Unknown scheduler type: {self.config.scheduler_type}")
    
    def _create_warmup_cosine_scheduler(self) -> Any:
        """Create warmup cosine scheduler"""
        def warmup_cosine_lr(epoch) -> Any:
            if epoch < self.config.warmup_steps:
                # Linear warmup
                return self.config.warmup_start_lr + (self.config.initial_lr - self.config.warmup_start_lr) * epoch / self.config.warmup_steps
            else:
                # Cosine annealing
                progress = (epoch - self.config.warmup_steps) / (self.config.T_max - self.config.warmup_steps)
                return self.config.eta_min + (self.config.initial_lr - self.config.eta_min) * 0.5 * (1 + np.cos(np.pi * progress))
        
        return LambdaLR(self.optimizer, warmup_cosine_lr)
    
    def step(self, metrics: Optional[Dict[str, float]] = None):
        """Step the scheduler"""
        if self.config.scheduler_type == "plateau" and metrics is not None:
            # Plateau scheduler needs metrics
            self.scheduler.step(metrics.get(self.config.monitor, 0))
        else:
            # Other schedulers step based on epoch
            self.scheduler.step()
        
        # Record current learning rate
        current_lr = self.optimizer.param_groups[0]['lr']
        self.history.append({
            'epoch': self.current_epoch,
            'learning_rate': current_lr,
            'timestamp': time.time()
        })
        
        self.current_epoch += 1
        
        # Log learning rate
        if self.config.verbose and self.current_epoch % self.config.log_interval == 0:
            logger.info(f"Epoch {self.current_epoch}: LR = {current_lr:.6f}")
    
    def get_lr(self) -> float:
        """Get current learning rate"""
        return self.optimizer.param_groups[0]['lr']
    
    def get_summary(self) -> Dict[str, Any]:
        """Get scheduler summary"""
        return {
            'scheduler_type': self.config.scheduler_type,
            'current_lr': self.get_lr(),
            'initial_lr': self.config.initial_lr,
            'min_lr': self.config.min_lr,
            'max_lr': self.config.max_lr,
            'current_epoch': self.current_epoch,
            'history': self.history
        }

class TrainingMonitor:
    """Comprehensive training monitoring with early stopping and LR scheduling"""
    
    def __init__(self, 
                 early_stopping_config: EarlyStoppingConfig,
                 lr_scheduler_config: LRSchedulerConfig,
                 optimizer: optim.Optimizer,
                 model: nn.Module):
        
        
    """__init__ function."""
self.early_stopping = EarlyStopping(early_stopping_config)
        self.lr_scheduler = AdvancedLRScheduler(optimizer, lr_scheduler_config)
        self.model = model
        self.optimizer = optimizer
        
        # Training state
        self.current_epoch = 0
        self.training_history = []
        self.best_metrics = None
        
        logger.info("Training monitor initialized")
    
    def update(self, metrics: TrainingMetrics) -> bool:
        """Update training monitor and check if should stop"""
        self.current_epoch = metrics.epoch
        self.training_history.append(metrics)
        
        # Update early stopping
        should_stop = self.early_stopping(metrics, self.model)
        
        # Update LR scheduler
        self.lr_scheduler.step()
        
        # Update best metrics
        if self.best_metrics is None or self._is_better_metrics(metrics):
            self.best_metrics = metrics
        
        return should_stop
    
    def _is_better_metrics(self, metrics: TrainingMetrics) -> bool:
        """Check if metrics are better than current best"""
        if self.best_metrics is None:
            return True
        
        # Compare based on validation loss (lower is better)
        return metrics.val_loss < self.best_metrics.val_loss
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get comprehensive training summary"""
        early_stopping_summary = self.early_stopping.get_summary()
        lr_scheduler_summary = self.lr_scheduler.get_summary()
        
        return {
            'early_stopping': early_stopping_summary,
            'lr_scheduler': lr_scheduler_summary,
            'best_metrics': self.best_metrics,
            'total_epochs': len(self.training_history),
            'training_history': self.training_history
        }
    
    def plot_training_curves(self, save_path: Optional[str] = None):
        """Plot training curves"""
        if not self.training_history:
            logger.warning("No training history to plot")
            return
        
        epochs = [m.epoch for m in self.training_history]
        train_losses = [m.train_loss for m in self.training_history]
        val_losses = [m.val_loss for m in self.training_history]
        train_accuracies = [m.train_accuracy for m in self.training_history]
        val_accuracies = [m.val_accuracy for m in self.training_history]
        learning_rates = [m.learning_rate for m in self.training_history]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss curves
        axes[0, 0].plot(epochs, train_losses, label='Train Loss', color='blue')
        axes[0, 0].plot(epochs, val_losses, label='Val Loss', color='red')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy curves
        axes[0, 1].plot(epochs, train_accuracies, label='Train Accuracy', color='blue')
        axes[0, 1].plot(epochs, val_accuracies, label='Val Accuracy', color='red')
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Learning rate curve
        axes[1, 0].plot(epochs, learning_rates, label='Learning Rate', color='green')
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        axes[1, 0].set_yscale('log')
        
        # Train-Val gap
        gaps = [train_acc - val_acc for train_acc, val_acc in zip(train_accuracies, val_accuracies)]
        axes[1, 1].plot(epochs, gaps, label='Train-Val Gap', color='orange')
        axes[1, 1].set_title('Training-Validation Accuracy Gap')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Gap')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training curves saved to {save_path}")
        
        plt.show()
    
    def save_training_log(self, save_path: str):
        """Save training log to file"""
        log_data = {
            'training_history': [
                {
                    'epoch': m.epoch,
                    'train_loss': m.train_loss,
                    'val_loss': m.val_loss,
                    'train_accuracy': m.train_accuracy,
                    'val_accuracy': m.val_accuracy,
                    'learning_rate': m.learning_rate,
                    'timestamp': m.timestamp
                }
                for m in self.training_history
            ],
            'summary': self.get_training_summary()
        }
        
        with open(save_path, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            json.dump(log_data, f, indent=2, default=str)
        
        logger.info(f"Training log saved to {save_path}")

class TrainingOptimizer:
    """High-level training optimizer with early stopping and LR scheduling"""
    
    def __init__(self, 
                 model: nn.Module,
                 optimizer: optim.Optimizer,
                 early_stopping_config: Optional[EarlyStoppingConfig] = None,
                 lr_scheduler_config: Optional[LRSchedulerConfig] = None):
        
        
    """__init__ function."""
self.model = model
        self.optimizer = optimizer
        
        # Default configurations
        if early_stopping_config is None:
            early_stopping_config = EarlyStoppingConfig()
        
        if lr_scheduler_config is None:
            lr_scheduler_config = LRSchedulerConfig()
        
        # Create monitor
        self.monitor = TrainingMonitor(
            early_stopping_config=early_stopping_config,
            lr_scheduler_config=lr_scheduler_config,
            optimizer=optimizer,
            model=model
        )
        
        logger.info("Training optimizer initialized")
    
    def train_epoch(self, train_loader, val_loader, criterion, device) -> TrainingMetrics:
        """Train for one epoch"""
        # Training phase
        self.model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            train_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            train_correct += pred.eq(target.view_as(pred)).sum().item()
            train_total += target.size(0)
        
        # Validation phase
        self.model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = self.model(data)
                loss = criterion(output, target)
                
                val_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                val_correct += pred.eq(target.view_as(pred)).sum().item()
                val_total += target.size(0)
        
        # Calculate metrics
        train_loss /= len(train_loader)
        train_accuracy = train_correct / train_total
        val_loss /= len(val_loader)
        val_accuracy = val_correct / val_total
        
        metrics = TrainingMetrics(
            epoch=self.monitor.current_epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            train_accuracy=train_accuracy,
            val_accuracy=val_accuracy,
            learning_rate=self.monitor.lr_scheduler.get_lr()
        )
        
        return metrics
    
    def train(self, train_loader, val_loader, criterion, device, max_epochs: int = 100) -> Dict[str, Any]:
        """Train the model with early stopping and LR scheduling"""
        logger.info(f"Starting training for up to {max_epochs} epochs")
        
        for epoch in range(max_epochs):
            # Train one epoch
            metrics = self.train_epoch(train_loader, val_loader, criterion, device)
            
            # Update monitor
            should_stop = self.monitor.update(metrics)
            
            # Log progress
            logger.info(
                f"Epoch {epoch+1}/{max_epochs} - "
                f"Train Loss: {metrics.train_loss:.4f}, "
                f"Train Acc: {metrics.train_accuracy:.4f}, "
                f"Val Loss: {metrics.val_loss:.4f}, "
                f"Val Acc: {metrics.val_accuracy:.4f}, "
                f"LR: {metrics.learning_rate:.6f}"
            )
            
            # Check early stopping
            if should_stop:
                logger.info("Early stopping triggered")
                break
        
        # Restore best weights
        self.monitor.early_stopping.restore_best_weights(self.model)
        
        # Get training summary
        summary = self.monitor.get_training_summary()
        
        logger.info("Training completed!")
        logger.info(f"Best validation loss: {summary['early_stopping']['best_score']:.4f}")
        logger.info(f"Best epoch: {summary['early_stopping']['best_epoch']}")
        
        return summary

# Example usage
if __name__ == "__main__":
    # Example: Create a simple model and dataset
    
    # Create sample data
    X_train = torch.randn(1000, 10)
    y_train = torch.randint(0, 2, (1000,))
    X_val = torch.randn(200, 10)
    y_val = torch.randint(0, 2, (200,))
    
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Create model
    model = nn.Sequential(
        nn.Linear(10, 64),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(32, 2)
    )
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    # Create configurations
    early_stopping_config = EarlyStoppingConfig(
        patience=10,
        monitor="val_loss",
        mode="min",
        restore_best_weights=True,
        verbose=True
    )
    
    lr_scheduler_config = LRSchedulerConfig(
        scheduler_type="cosine",
        initial_lr=1e-3,
        T_max=100,
        verbose=True
    )
    
    # Create training optimizer
    trainer = TrainingOptimizer(
        model=model,
        optimizer=optimizer,
        early_stopping_config=early_stopping_config,
        lr_scheduler_config=lr_scheduler_config
    )
    
    # Train
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    summary = trainer.train(train_loader, val_loader, criterion, device, max_epochs=50)
    
    # Plot training curves
    trainer.monitor.plot_training_curves()
    
    # Save training log
    trainer.monitor.save_training_log("training_log.json")
    
    print("Training completed successfully!") 