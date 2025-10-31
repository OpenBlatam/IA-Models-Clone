from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import asyncio
import time
import logging
import json
import math
import warnings
from typing import Dict, Any, Optional, List, Union, Callable, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
from collections import deque, defaultdict
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import (
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from .production_transformers import DeviceManager
from typing import Any, List, Dict, Optional
"""
🚀 Early Stopping & Learning Rate Scheduling System - Production Ready
=====================================================================

Enterprise-grade early stopping and learning rate scheduling system with
multiple strategies, monitoring, and production-ready features for AI training.
"""


    StepLR, MultiStepLR, ExponentialLR, CosineAnnealingLR,
    CosineAnnealingWarmRestarts, ReduceLROnPlateau, OneCycleLR,
    CyclicLR, LambdaLR, ChainedScheduler
)

# Import our production engines

logger = logging.getLogger(__name__)

class EarlyStoppingMode(Enum):
    """Early stopping modes."""
    MIN = "min"  # Stop when metric stops decreasing
    MAX = "max"  # Stop when metric stops increasing

class EarlyStoppingStrategy(Enum):
    """Early stopping strategies."""
    PATIENCE = "patience"  # Stop after N epochs without improvement
    DELTA = "delta"  # Stop when improvement is less than delta
    PERCENTAGE = "percentage"  # Stop when improvement is less than percentage
    MOVING_AVERAGE = "moving_average"  # Use moving average for stability
    CUSTOM = "custom"  # Custom early stopping logic

class LRSchedulerType(Enum):
    """Learning rate scheduler types."""
    STEP = "step"
    MULTI_STEP = "multi_step"
    EXPONENTIAL = "exponential"
    COSINE_ANNEALING = "cosine_annealing"
    COSINE_ANNEALING_WARM_RESTARTS = "cosine_annealing_warm_restarts"
    REDUCE_LR_ON_PLATEAU = "reduce_lr_on_plateau"
    ONE_CYCLE = "one_cycle"
    CYCLIC = "cyclic"
    LAMBDA = "lambda"
    CUSTOM = "custom"

@dataclass
class EarlyStoppingConfig:
    """Configuration for early stopping."""
    enabled: bool = True
    strategy: EarlyStoppingStrategy = EarlyStoppingStrategy.PATIENCE
    mode: EarlyStoppingMode = EarlyStoppingMode.MIN
    patience: int = 10
    min_delta: float = 0.0
    min_percentage: float = 0.01  # 1% improvement
    moving_average_window: int = 5
    restore_best_weights: bool = True
    verbose: bool = True
    
    # Custom early stopping
    custom_stopping_function: Optional[Callable] = None
    
    # Monitoring
    monitor: str = "val_loss"  # Metric to monitor
    min_epochs: int = 0  # Minimum epochs before stopping
    
    def __post_init__(self) -> Any:
        """Validate configuration."""
        if self.patience < 0:
            raise ValueError("Patience must be non-negative")
        if self.min_delta < 0:
            raise ValueError("Min delta must be non-negative")
        if self.min_percentage < 0 or self.min_percentage > 1:
            raise ValueError("Min percentage must be between 0 and 1")
        if self.moving_average_window < 1:
            raise ValueError("Moving average window must be at least 1")

@dataclass
class LRSchedulerConfig:
    """Configuration for learning rate scheduling."""
    scheduler_type: LRSchedulerType = LRSchedulerType.COSINE_ANNEALING
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
    
    # Cosine annealing with warm restarts
    T_0: int = 10
    T_mult: int = 2
    
    # Reduce LR on plateau
    factor: float = 0.1
    patience: int = 10
    threshold: float = 1e-4
    threshold_mode: str = "rel"
    cooldown: int = 0
    min_lr_plateau: float = 0.0
    
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
    base_lr: float = 1e-4
    max_lr: float = 1e-2
    step_size_up: int = 2000
    step_size_down: int = 2000
    mode: str = "triangular"
    scale_fn: Optional[Callable] = None
    scale_mode: str = "cycle"
    
    # Custom
    custom_scheduler_function: Optional[Callable] = None
    custom_lr_function: Optional[Callable] = None
    
    def __post_init__(self) -> Any:
        """Validate configuration."""
        if self.initial_lr <= 0:
            raise ValueError("Initial learning rate must be positive")
        if self.min_lr < 0:
            raise ValueError("Minimum learning rate must be non-negative")
        if self.max_lr <= 0:
            raise ValueError("Maximum learning rate must be positive")
        if self.min_lr >= self.max_lr:
            raise ValueError("Min LR must be less than Max LR")

@dataclass
class EarlyStoppingState:
    """State for early stopping."""
    best_score: float = float('inf')
    best_epoch: int = 0
    counter: int = 0
    stopped: bool = False
    history: List[float] = field(default_factory=list)
    moving_averages: List[float] = field(default_factory=list)
    
    def reset(self) -> Any:
        """Reset state."""
        self.best_score = float('inf')
        self.best_epoch = 0
        self.counter = 0
        self.stopped = False
        self.history.clear()
        self.moving_averages.clear()

@dataclass
class LRSchedulerState:
    """State for learning rate scheduling."""
    current_lr: float = 1e-3
    history: List[float] = field(default_factory=list)
    best_lr: float = 1e-3
    scheduler: Optional[optim.lr_scheduler._LRScheduler] = None

class EarlyStopping:
    """Production-ready early stopping implementation."""
    
    def __init__(self, config: EarlyStoppingConfig, device_manager: DeviceManager):
        
    """__init__ function."""
self.config = config
        self.device_manager = device_manager
        self.state = EarlyStoppingState()
        self.logger = logging.getLogger(f"{__name__}.EarlyStopping")
        
        # Initialize best score based on mode
        if config.mode == EarlyStoppingMode.MIN:
            self.state.best_score = float('inf')
        else:
            self.state.best_score = float('-inf')
    
    def __call__(self, epoch: int, metric: float, model: nn.Module) -> bool:
        """Check if training should stop early."""
        if not self.config.enabled:
            return False
        
        # Add to history
        self.state.history.append(metric)
        
        # Calculate moving average if enabled
        if self.config.strategy == EarlyStoppingStrategy.MOVING_AVERAGE:
            window = min(self.config.moving_average_window, len(self.state.history))
            moving_avg = np.mean(self.state.history[-window:])
            self.state.moving_averages.append(moving_avg)
            metric = moving_avg
        
        # Check if this is the best score so far
        is_better = self._is_better(metric)
        
        if is_better:
            self.state.best_score = metric
            self.state.best_epoch = epoch
            self.state.counter = 0
            
            # Save best model if requested
            if self.config.restore_best_weights:
                self._save_best_model(model)
                
            if self.config.verbose:
                self.logger.info(f"Early stopping: New best {self.config.monitor} = {metric:.6f} at epoch {epoch}")
        else:
            self.state.counter += 1
            
            if self.config.verbose:
                self.logger.info(f"Early stopping: {self.config.monitor} did not improve for {self.state.counter} epochs")
        
        # Check if we should stop
        should_stop = self._should_stop(epoch, metric)
        
        if should_stop and self.config.verbose:
            self.logger.info(f"Early stopping triggered at epoch {epoch}")
            if self.config.restore_best_weights:
                self.logger.info(f"Restoring best model from epoch {self.state.best_epoch}")
        
        return should_stop
    
    def _is_better(self, metric: float) -> bool:
        """Check if the current metric is better than the best."""
        if self.config.mode == EarlyStoppingMode.MIN:
            return metric < self.state.best_score - self.config.min_delta
        else:
            return metric > self.state.best_score + self.config.min_delta
    
    def _should_stop(self, epoch: int, metric: float) -> bool:
        """Determine if training should stop."""
        # Check minimum epochs
        if epoch < self.config.min_epochs:
            return False
        
        # Check custom stopping function
        if self.config.strategy == EarlyStoppingStrategy.CUSTOM and self.config.custom_stopping_function:
            return self.config.custom_stopping_function(epoch, metric, self.state)
        
        # Check patience
        if self.config.strategy == EarlyStoppingStrategy.PATIENCE:
            return self.state.counter >= self.config.patience
        
        # Check delta
        if self.config.strategy == EarlyStoppingStrategy.DELTA:
            if self.config.mode == EarlyStoppingMode.MIN:
                return metric >= self.state.best_score - self.config.min_delta
            else:
                return metric <= self.state.best_score + self.config.min_delta
        
        # Check percentage
        if self.config.strategy == EarlyStoppingStrategy.PERCENTAGE:
            if self.state.best_score == 0:
                return False
            improvement = abs(metric - self.state.best_score) / abs(self.state.best_score)
            return improvement < self.config.min_percentage
        
        # Check moving average
        if self.config.strategy == EarlyStoppingStrategy.MOVING_AVERAGE:
            if len(self.state.moving_averages) < 2:
                return False
            recent_avg = self.state.moving_averages[-1]
            previous_avg = self.state.moving_averages[-2]
            if self.config.mode == EarlyStoppingMode.MIN:
                return recent_avg >= previous_avg
            else:
                return recent_avg <= previous_avg
        
        return False
    
    def _save_best_model(self, model: nn.Module):
        """Save the best model state."""
        if not hasattr(self, '_best_model_state'):
            self._best_model_state = copy.deepcopy(model.state_dict())
        else:
            self._best_model_state = copy.deepcopy(model.state_dict())
    
    def restore_best_model(self, model: nn.Module):
        """Restore the best model state."""
        if hasattr(self, '_best_model_state') and self.config.restore_best_weights:
            model.load_state_dict(self._best_model_state)
            self.logger.info("Best model weights restored")
    
    def get_state(self) -> Dict[str, Any]:
        """Get current state."""
        return {
            'best_score': self.state.best_score,
            'best_epoch': self.state.best_epoch,
            'counter': self.state.counter,
            'stopped': self.state.stopped,
            'history': self.state.history.copy(),
            'moving_averages': self.state.moving_averages.copy()
        }
    
    def reset(self) -> Any:
        """Reset early stopping state."""
        self.state.reset()
        if hasattr(self, '_best_model_state'):
            delattr(self, '_best_model_state')

class LRScheduler:
    """Production-ready learning rate scheduler."""
    
    def __init__(self, config: LRSchedulerConfig, device_manager: DeviceManager):
        
    """__init__ function."""
self.config = config
        self.device_manager = device_manager
        self.state = LRSchedulerState()
        self.logger = logging.getLogger(f"{__name__}.LRScheduler")
        
        # Initialize scheduler
        self.scheduler = None
    
    def create_scheduler(self, optimizer: optim.Optimizer, num_training_steps: int = None) -> optim.lr_scheduler._LRScheduler:
        """Create learning rate scheduler."""
        if self.config.scheduler_type == LRSchedulerType.STEP:
            self.scheduler = StepLR(
                optimizer,
                step_size=self.config.step_size,
                gamma=self.config.gamma
            )
        
        elif self.config.scheduler_type == LRSchedulerType.MULTI_STEP:
            self.scheduler = MultiStepLR(
                optimizer,
                milestones=self.config.milestones,
                gamma=self.config.gamma
            )
        
        elif self.config.scheduler_type == LRSchedulerType.EXPONENTIAL:
            self.scheduler = ExponentialLR(
                optimizer,
                gamma=self.config.gamma
            )
        
        elif self.config.scheduler_type == LRSchedulerType.COSINE_ANNEALING:
            self.scheduler = CosineAnnealingLR(
                optimizer,
                T_max=self.config.T_max,
                eta_min=self.config.eta_min
            )
        
        elif self.config.scheduler_type == LRSchedulerType.COSINE_ANNEALING_WARM_RESTARTS:
            self.scheduler = CosineAnnealingWarmRestarts(
                optimizer,
                T_0=self.config.T_0,
                T_mult=self.config.T_mult,
                eta_min=self.config.eta_min
            )
        
        elif self.config.scheduler_type == LRSchedulerType.REDUCE_LR_ON_PLATEAU:
            self.scheduler = ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=self.config.factor,
                patience=self.config.patience,
                threshold=self.config.threshold,
                threshold_mode=self.config.threshold_mode,
                cooldown=self.config.cooldown,
                min_lr=self.config.min_lr_plateau,
                verbose=True
            )
        
        elif self.config.scheduler_type == LRSchedulerType.ONE_CYCLE:
            if num_training_steps is None:
                raise ValueError("num_training_steps is required for OneCycleLR")
            
            self.scheduler = OneCycleLR(
                optimizer,
                max_lr=self.config.max_lr,
                epochs=self.config.epochs,
                steps_per_epoch=self.config.steps_per_epoch,
                pct_start=self.config.pct_start,
                anneal_strategy=self.config.anneal_strategy,
                cycle_momentum=self.config.cycle_momentum,
                base_momentum=self.config.base_momentum,
                max_momentum=self.config.max_momentum,
                div_factor=self.config.div_factor,
                final_div_factor=self.config.final_div_factor
            )
        
        elif self.config.scheduler_type == LRSchedulerType.CYCLIC:
            self.scheduler = CyclicLR(
                optimizer,
                base_lr=self.config.base_lr,
                max_lr=self.config.max_lr,
                step_size_up=self.config.step_size_up,
                step_size_down=self.config.step_size_down,
                mode=self.config.mode,
                scale_fn=self.config.scale_fn,
                scale_mode=self.config.scale_mode
            )
        
        elif self.config.scheduler_type == LRSchedulerType.LAMBDA:
            if self.config.custom_lr_function is None:
                raise ValueError("custom_lr_function is required for LambdaLR")
            
            self.scheduler = LambdaLR(
                optimizer,
                lr_lambda=self.config.custom_lr_function
            )
        
        elif self.config.scheduler_type == LRSchedulerType.CUSTOM:
            if self.config.custom_scheduler_function is None:
                raise ValueError("custom_scheduler_function is required for Custom scheduler")
            
            self.scheduler = self.config.custom_scheduler_function(optimizer)
        
        else:
            raise ValueError(f"Unknown scheduler type: {self.config.scheduler_type}")
        
        self.state.scheduler = self.scheduler
        self.state.current_lr = self.config.initial_lr
        
        self.logger.info(f"Created {self.config.scheduler_type.value} scheduler")
        return self.scheduler
    
    def step(self, epoch: int = None, metrics: Dict[str, float] = None):
        """Step the scheduler."""
        if self.scheduler is None:
            raise RuntimeError("Scheduler not created. Call create_scheduler() first.")
        
        # Handle different scheduler types
        if isinstance(self.scheduler, ReduceLROnPlateau):
            if metrics is None:
                raise ValueError("Metrics required for ReduceLROnPlateau")
            metric_value = metrics.get('val_loss', 0.0)
            self.scheduler.step(metric_value)
        else:
            self.scheduler.step()
        
        # Update state
        current_lr = self.scheduler.optimizer.param_groups[0]['lr']
        self.state.current_lr = current_lr
        self.state.history.append(current_lr)
        
        # Track best learning rate
        if current_lr > self.state.best_lr:
            self.state.best_lr = current_lr
        
        if epoch is not None:
            self.logger.debug(f"Epoch {epoch}: LR = {current_lr:.2e}")
    
    def get_lr(self) -> float:
        """Get current learning rate."""
        if self.scheduler is None:
            return self.config.initial_lr
        return self.scheduler.optimizer.param_groups[0]['lr']
    
    def get_state(self) -> Dict[str, Any]:
        """Get current state."""
        return {
            'current_lr': self.state.current_lr,
            'best_lr': self.state.best_lr,
            'history': self.state.history.copy(),
            'scheduler_type': self.config.scheduler_type.value
        }
    
    def reset(self) -> Any:
        """Reset scheduler state."""
        self.state = LRSchedulerState()
        self.scheduler = None

class TrainingMonitor:
    """Monitor training progress and manage early stopping and LR scheduling."""
    
    def __init__(self, device_manager: DeviceManager):
        
    """__init__ function."""
self.device_manager = device_manager
        self.early_stopping = None
        self.lr_scheduler = None
        self.logger = logging.getLogger(f"{__name__}.TrainingMonitor")
        
        # Training history
        self.training_history = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': [],
            'learning_rate': [],
            'early_stopping_counter': [],
            'best_metric': []
        }
    
    def setup_early_stopping(self, config: EarlyStoppingConfig):
        """Setup early stopping."""
        self.early_stopping = EarlyStopping(config, self.device_manager)
        self.logger.info(f"Early stopping configured: {config.strategy.value}")
    
    def setup_lr_scheduler(self, config: LRSchedulerConfig, optimizer: optim.Optimizer, num_training_steps: int = None):
        """Setup learning rate scheduler."""
        self.lr_scheduler = LRScheduler(config, self.device_manager)
        scheduler = self.lr_scheduler.create_scheduler(optimizer, num_training_steps)
        self.logger.info(f"LR scheduler configured: {config.scheduler_type.value}")
        return scheduler
    
    def update(self, epoch: int, metrics: Dict[str, float], model: nn.Module) -> bool:
        """Update monitor and check if training should stop."""
        # Update training history
        self._update_history(epoch, metrics)
        
        # Update early stopping
        should_stop = False
        if self.early_stopping:
            monitored_metric = metrics.get(self.early_stopping.config.monitor, 0.0)
            should_stop = self.early_stopping(epoch, monitored_metric, model)
        
        # Update LR scheduler
        if self.lr_scheduler:
            self.lr_scheduler.step(epoch, metrics)
        
        return should_stop
    
    def _update_history(self, epoch: int, metrics: Dict[str, float]):
        """Update training history."""
        self.training_history['epoch'].append(epoch)
        self.training_history['train_loss'].append(metrics.get('train_loss', 0.0))
        self.training_history['val_loss'].append(metrics.get('val_loss', 0.0))
        self.training_history['train_accuracy'].append(metrics.get('train_accuracy', 0.0))
        self.training_history['val_accuracy'].append(metrics.get('val_accuracy', 0.0))
        self.training_history['learning_rate'].append(metrics.get('learning_rate', 0.0))
        
        if self.early_stopping:
            self.training_history['early_stopping_counter'].append(self.early_stopping.state.counter)
            self.training_history['best_metric'].append(self.early_stopping.state.best_score)
    
    def restore_best_model(self, model: nn.Module):
        """Restore best model if early stopping is enabled."""
        if self.early_stopping:
            self.early_stopping.restore_best_model(model)
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get training summary."""
        summary = {
            'total_epochs': len(self.training_history['epoch']),
            'final_train_loss': self.training_history['train_loss'][-1] if self.training_history['train_loss'] else 0.0,
            'final_val_loss': self.training_history['val_loss'][-1] if self.training_history['val_loss'] else 0.0,
            'final_train_accuracy': self.training_history['train_accuracy'][-1] if self.training_history['train_accuracy'] else 0.0,
            'final_val_accuracy': self.training_history['val_accuracy'][-1] if self.training_history['val_accuracy'] else 0.0,
            'training_history': self.training_history.copy()
        }
        
        if self.early_stopping:
            summary.update({
                'early_stopping_triggered': self.early_stopping.state.stopped,
                'best_epoch': self.early_stopping.state.best_epoch,
                'best_metric': self.early_stopping.state.best_score,
                'early_stopping_state': self.early_stopping.get_state()
            })
        
        if self.lr_scheduler:
            summary.update({
                'lr_scheduler_state': self.lr_scheduler.get_state()
            })
        
        return summary
    
    def plot_training_curves(self, save_path: Optional[str] = None):
        """Plot training curves."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss curves
        axes[0, 0].plot(self.training_history['epoch'], self.training_history['train_loss'], label='Train Loss')
        axes[0, 0].plot(self.training_history['epoch'], self.training_history['val_loss'], label='Val Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy curves
        axes[0, 1].plot(self.training_history['epoch'], self.training_history['train_accuracy'], label='Train Accuracy')
        axes[0, 1].plot(self.training_history['epoch'], self.training_history['val_accuracy'], label='Val Accuracy')
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Learning rate curve
        axes[1, 0].plot(self.training_history['epoch'], self.training_history['learning_rate'])
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True)
        
        # Early stopping counter
        if self.training_history['early_stopping_counter']:
            axes[1, 1].plot(self.training_history['epoch'], self.training_history['early_stopping_counter'])
            axes[1, 1].set_title('Early Stopping Counter')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Counter')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Training curves saved to {save_path}")
        
        plt.show()
    
    def reset(self) -> Any:
        """Reset monitor state."""
        if self.early_stopping:
            self.early_stopping.reset()
        if self.lr_scheduler:
            self.lr_scheduler.reset()
        
        self.training_history = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': [],
            'learning_rate': [],
            'early_stopping_counter': [],
            'best_metric': []
        }

# Factory functions
async def create_training_monitor(device_manager: DeviceManager) -> TrainingMonitor:
    """Create a training monitor instance."""
    return TrainingMonitor(device_manager)

def create_early_stopping_config(
    enabled: bool = True,
    strategy: EarlyStoppingStrategy = EarlyStoppingStrategy.PATIENCE,
    mode: EarlyStoppingMode = EarlyStoppingMode.MIN,
    patience: int = 10,
    monitor: str = "val_loss",
    min_epochs: int = 0
) -> EarlyStoppingConfig:
    """Create early stopping configuration."""
    return EarlyStoppingConfig(
        enabled=enabled,
        strategy=strategy,
        mode=mode,
        patience=patience,
        monitor=monitor,
        min_epochs=min_epochs
    )

def create_lr_scheduler_config(
    scheduler_type: LRSchedulerType = LRSchedulerType.COSINE_ANNEALING,
    initial_lr: float = 1e-3,
    min_lr: float = 1e-6,
    max_lr: float = 1e-2
) -> LRSchedulerConfig:
    """Create learning rate scheduler configuration."""
    return LRSchedulerConfig(
        scheduler_type=scheduler_type,
        initial_lr=initial_lr,
        min_lr=min_lr,
        max_lr=max_lr
    )

# Quick usage functions
async def quick_early_stopping_example():
    """Quick example of early stopping usage."""
    device_manager = DeviceManager()
    monitor = await create_training_monitor(device_manager)
    
    # Setup early stopping
    es_config = create_early_stopping_config(
        enabled=True,
        strategy=EarlyStoppingStrategy.PATIENCE,
        mode=EarlyStoppingMode.MIN,
        patience=5,
        monitor="val_loss"
    )
    monitor.setup_early_stopping(es_config)
    
    # Setup LR scheduler
    lr_config = create_lr_scheduler_config(
        scheduler_type=LRSchedulerType.COSINE_ANNEALING,
        initial_lr=1e-3
    )
    
    # Mock optimizer and model
    model = nn.Linear(10, 1)
    optimizer = optim.Adam(model.parameters(), lr=lr_config.initial_lr)
    
    scheduler = monitor.setup_lr_scheduler(lr_config, optimizer)
    
    print("✅ Early stopping and LR scheduling configured successfully")
    return monitor

# Example usage
if __name__ == "__main__":
    async def demo():
        
    """demo function."""
# Quick example
        monitor = await quick_early_stopping_example()
        
        # Simulate training
        model = nn.Linear(10, 1)
        
        for epoch in range(20):
            # Simulate metrics
            metrics = {
                'train_loss': 1.0 / (epoch + 1),
                'val_loss': 1.2 / (epoch + 1),
                'train_accuracy': 0.5 + epoch * 0.02,
                'val_accuracy': 0.48 + epoch * 0.018,
                'learning_rate': 1e-3 * (0.9 ** epoch)
            }
            
            # Update monitor
            should_stop = monitor.update(epoch, metrics, model)
            
            if should_stop:
                print(f"Early stopping triggered at epoch {epoch}")
                break
        
        # Get summary
        summary = monitor.get_training_summary()
        print(f"Training completed: {summary['total_epochs']} epochs")
        print(f"Best metric: {summary['best_metric']:.6f}")
        
        # Plot curves
        monitor.plot_training_curves()
    
    asyncio.run(demo()) 