from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass
from collections import defaultdict
import math
import time
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from ..models.sequence import EmailSequence, SequenceStep
from ..models.subscriber import Subscriber
from ..models.template import EmailTemplate
from .gradient_management import GradientManager, GradientConfig, create_gradient_manager
from typing import Any, List, Dict, Optional
"""
Training Optimization for Email Sequence System

Advanced early stopping and learning rate scheduling implementations
with multiple strategies, monitoring, and optimization techniques.
"""


    StepLR,
    MultiStepLR,
    ExponentialLR,
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    ReduceLROnPlateau,
    OneCycleLR,
    CyclicLR,
    LambdaLR,
    MultiplicativeLR,
    ChainedScheduler,
    SequentialLR
)


logger = logging.getLogger(__name__)


@dataclass
class EarlyStoppingConfig:
    """Configuration for early stopping"""
    # Basic parameters
    patience: int = 10
    min_delta: float = 0.001
    monitor: str = "val_loss"  # val_loss, train_loss, val_accuracy, etc.
    mode: str = "min"  # min, max
    
    # Advanced parameters
    restore_best_weights: bool = True
    verbose: bool = True
    save_checkpoint: bool = True
    checkpoint_dir: str = "./checkpoints"
    
    # Custom stopping conditions
    min_epochs: int = 0
    max_epochs: int = 1000
    min_lr: float = 1e-8
    
    # Monitoring parameters
    monitor_window: int = 5
    smoothing_factor: float = 0.1
    
    # Performance tracking
    track_metrics: List[str] = None  # List of additional metrics to track


@dataclass
class LRSchedulerConfig:
    """Configuration for learning rate scheduling"""
    # Scheduler type
    scheduler_type: str = "cosine"  # cosine, step, exponential, plateau, onecycle, cyclic
    
    # Basic parameters
    initial_lr: float = 1e-3
    min_lr: float = 1e-6
    max_lr: float = 1e-2
    
    # Step scheduler parameters
    step_size: int = 30
    gamma: float = 0.1
    milestones: List[int] = None
    
    # Cosine scheduler parameters
    T_max: int = 100
    eta_min: float = 1e-6
    
    # Plateau scheduler parameters
    factor: float = 0.5
    patience: int = 5
    threshold: float = 1e-4
    cooldown: int = 0
    
    # OneCycle scheduler parameters
    epochs: int = 100
    steps_per_epoch: int = 100
    pct_start: float = 0.3
    anneal_strategy: str = "cos"
    
    # Cyclic scheduler parameters
    base_lr: float = 1e-6
    max_lr: float = 1e-3
    step_size_up: int = 2000
    step_size_down: int = 2000
    mode: str = "triangular"
    
    # Warmup parameters
    warmup_steps: int = 0
    warmup_factor: float = 0.1
    
    # Custom scheduling
    custom_schedule: Callable = None


@dataclass
class GradientManagementConfig:
    """Configuration for gradient management"""
    # Enable gradient management
    enable_gradient_management: bool = True
    
    # Gradient clipping
    max_grad_norm: float = 1.0
    clip_type: str = "norm"  # "norm", "value", "adaptive"
    
    # NaN/Inf handling
    enable_nan_inf_check: bool = True
    replace_nan_with: float = 0.0
    replace_inf_with: float = 1e6
    
    # Monitoring
    enable_gradient_monitoring: bool = True
    verbose_logging: bool = False
    
    # Adaptive clipping
    adaptive_clipping: bool = False
    adaptive_window_size: int = 100
    adaptive_percentile: float = 95.0
    
    # Performance
    check_frequency: int = 1  # Check every N steps


class EarlyStopping:
    """Advanced early stopping implementation"""
    
    def __init__(self, config: EarlyStoppingConfig):
        
    """__init__ function."""
self.config = config
        self.best_score = None
        self.best_epoch = 0
        self.counter = 0
        self.early_stop = False
        self.best_weights = None
        
        # Initialize tracking
        self.monitor_values = []
        self.epochs = []
        self.best_monitor_value = float('inf') if config.mode == "min" else float('-inf')
        
        # Setup checkpoint directory
        if config.save_checkpoint:
            self.checkpoint_dir = Path(config.checkpoint_dir)
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize metric tracking
        if config.track_metrics is None:
            config.track_metrics = []
        
        self.metric_history = {metric: [] for metric in config.track_metrics}
        
        logger.info(f"Early Stopping initialized with patience={config.patience}, monitor={config.monitor}")
    
    def __call__(
        self,
        monitor_value: float,
        model: nn.Module,
        epoch: int,
        additional_metrics: Dict[str, float] = None
    ) -> bool:
        """Check if training should stop early"""
        
        # Update tracking
        self.monitor_values.append(monitor_value)
        self.epochs.append(epoch)
        
        # Update additional metrics
        if additional_metrics:
            for metric_name, value in additional_metrics.items():
                if metric_name in self.metric_history:
                    self.metric_history[metric_name].append(value)
        
        # Check if this is the best score
        if self._is_better_score(monitor_value):
            self.best_score = monitor_value
            self.best_epoch = epoch
            self.counter = 0
            
            # Save best weights
            if self.config.restore_best_weights:
                self.best_weights = self._get_model_state(model)
            
            # Save checkpoint
            if self.config.save_checkpoint:
                self._save_checkpoint(model, epoch, monitor_value)
            
            if self.config.verbose:
                logger.info(f"Best {self.config.monitor}: {monitor_value:.6f} at epoch {epoch}")
        else:
            self.counter += 1
            
            if self.config.verbose and self.counter >= self.config.patience:
                logger.info(f"Early stopping triggered after {self.counter} epochs without improvement")
        
        # Check early stopping conditions
        self.early_stop = self._should_stop(epoch)
        
        return self.early_stop
    
    def _is_better_score(self, monitor_value: float) -> bool:
        """Check if current score is better than best score"""
        
        if self.best_score is None:
            return True
        
        if self.config.mode == "min":
            return monitor_value < self.best_score - self.config.min_delta
        else:
            return monitor_value > self.best_score + self.config.min_delta
    
    def _should_stop(self, epoch: int) -> bool:
        """Determine if training should stop"""
        
        # Check patience
        if self.counter >= self.config.patience:
            return True
        
        # Check minimum epochs
        if epoch < self.config.min_epochs:
            return False
        
        # Check maximum epochs
        if epoch >= self.config.max_epochs:
            return True
        
        # Check learning rate
        if hasattr(self, 'current_lr') and self.current_lr <= self.config.min_lr:
            return True
        
        return False
    
    def _get_model_state(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        """Get model state dict"""
        return {name: param.clone().detach() for name, param in model.state_dict().items()}
    
    def _save_checkpoint(
        self,
        model: nn.Module,
        epoch: int,
        monitor_value: float
    ):
        """Save model checkpoint"""
        
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "best_score": self.best_score,
            "monitor_value": monitor_value,
            "config": self.config.__dict__,
            "metric_history": self.metric_history
        }
        
        checkpoint_path = self.checkpoint_dir / f"best_model_epoch_{epoch}.pth"
        torch.save(checkpoint, checkpoint_path)
        
        if self.config.verbose:
            logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def restore_best_weights(self, model: nn.Module):
        """Restore best weights to model"""
        
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)
            logger.info(f"Restored best weights from epoch {self.best_epoch}")
        else:
            logger.warning("No best weights to restore")
    
    def get_best_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Get best checkpoint information"""
        
        if self.best_score is None:
            return None
        
        return {
            "best_score": self.best_score,
            "best_epoch": self.best_epoch,
            "monitor_values": self.monitor_values,
            "epochs": self.epochs,
            "metric_history": self.metric_history
        }
    
    def plot_training_curves(self, save_path: str = None):
        """Plot training curves"""
        
        if not self.monitor_values:
            logger.warning("No monitor values to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot monitor values
        axes[0, 0].plot(self.epochs, self.monitor_values, 'b-', label=self.config.monitor)
        axes[0, 0].axhline(y=self.best_score, color='r', linestyle='--', label=f'Best: {self.best_score:.6f}')
        axes[0, 0].set_title(f"{self.config.monitor} Over Time")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel(self.config.monitor)
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Plot improvement
        improvements = []
        for i in range(1, len(self.monitor_values)):
            if self.config.mode == "min":
                improvement = self.monitor_values[i-1] - self.monitor_values[i]
            else:
                improvement = self.monitor_values[i] - self.monitor_values[i-1]
            improvements.append(improvement)
        
        axes[0, 1].plot(self.epochs[1:], improvements, 'g-')
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        axes[0, 1].set_title("Improvement per Epoch")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("Improvement")
        axes[0, 1].grid(True)
        
        # Plot additional metrics
        if self.metric_history:
            for i, (metric_name, values) in enumerate(self.metric_history.items()):
                if values:
                    axes[1, 0].plot(range(len(values)), values, label=metric_name)
            
            axes[1, 0].set_title("Additional Metrics")
            axes[1, 0].set_xlabel("Epoch")
            axes[1, 0].set_ylabel("Value")
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # Plot patience counter
        patience_counter = []
        current_counter = 0
        for i in range(len(self.monitor_values)):
            if i > 0:
                if self._is_better_score(self.monitor_values[i]):
                    current_counter = 0
                else:
                    current_counter += 1
            patience_counter.append(current_counter)
        
        axes[1, 1].plot(self.epochs, patience_counter, 'r-')
        axes[1, 1].axhline(y=self.config.patience, color='r', linestyle='--', label=f'Patience: {self.config.patience}')
        axes[1, 1].set_title("Patience Counter")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("Counter")
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()


class AdvancedLRScheduler:
    """Advanced learning rate scheduler with multiple strategies"""
    
    def __init__(self, config: LRSchedulerConfig, optimizer: Optimizer):
        
    """__init__ function."""
self.config = config
        self.optimizer = optimizer
        self.scheduler = None
        self.current_lr = config.initial_lr
        
        # Initialize scheduler
        self._setup_scheduler()
        
        # Tracking
        self.lr_history = []
        self.epoch_history = []
        
        logger.info(f"LR Scheduler initialized: {config.scheduler_type}")
    
    def _setup_scheduler(self) -> Any:
        """Setup learning rate scheduler"""
        
        if self.config.scheduler_type == "step":
            self.scheduler = StepLR(
                self.optimizer,
                step_size=self.config.step_size,
                gamma=self.config.gamma
            )
        
        elif self.config.scheduler_type == "multistep":
            if self.config.milestones is None:
                self.config.milestones = [30, 60, 90]
            self.scheduler = MultiStepLR(
                self.optimizer,
                milestones=self.config.milestones,
                gamma=self.config.gamma
            )
        
        elif self.config.scheduler_type == "exponential":
            self.scheduler = ExponentialLR(
                self.optimizer,
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
                T_0=self.config.T_max,
                T_mult=2,
                eta_min=self.config.eta_min
            )
        
        elif self.config.scheduler_type == "plateau":
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=self.config.factor,
                patience=self.config.patience,
                threshold=self.config.threshold,
                cooldown=self.config.cooldown,
                min_lr=self.config.min_lr
            )
        
        elif self.config.scheduler_type == "onecycle":
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=self.config.max_lr,
                epochs=self.config.epochs,
                steps_per_epoch=self.config.steps_per_epoch,
                pct_start=self.config.pct_start,
                anneal_strategy=self.config.anneal_strategy
            )
        
        elif self.config.scheduler_type == "cyclic":
            self.scheduler = CyclicLR(
                self.optimizer,
                base_lr=self.config.base_lr,
                max_lr=self.config.max_lr,
                step_size_up=self.config.step_size_up,
                step_size_down=self.config.step_size_down,
                mode=self.config.mode
            )
        
        elif self.config.scheduler_type == "custom":
            if self.config.custom_schedule is None:
                raise ValueError("Custom schedule function must be provided")
            self.scheduler = LambdaLR(
                self.optimizer,
                lr_lambda=self.config.custom_schedule
            )
        
        else:
            raise ValueError(f"Unknown scheduler type: {self.config.scheduler_type}")
    
    def step(self, metrics: Optional[float] = None):
        """Step the scheduler"""
        
        if self.config.scheduler_type == "plateau":
            if metrics is None:
                raise ValueError("Metrics required for plateau scheduler")
            self.scheduler.step(metrics)
        else:
            self.scheduler.step()
        
        # Update current learning rate
        self.current_lr = self.optimizer.param_groups[0]['lr']
        self.lr_history.append(self.current_lr)
    
    def step_epoch(self, epoch: int, metrics: Optional[float] = None):
        """Step scheduler for an epoch"""
        
        self.step(metrics)
        self.epoch_history.append(epoch)
    
    def get_lr(self) -> float:
        """Get current learning rate"""
        return self.optimizer.param_groups[0]['lr']
    
    def set_lr(self, lr: float):
        """Set learning rate manually"""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self.current_lr = lr
    
    def plot_lr_schedule(self, save_path: str = None):
        """Plot learning rate schedule"""
        
        if not self.lr_history:
            logger.warning("No LR history to plot")
            return
        
        plt.figure(figsize=(12, 6))
        
        if self.epoch_history:
            plt.plot(self.epoch_history, self.lr_history, 'b-', linewidth=2)
            plt.xlabel("Epoch")
        else:
            plt.plot(range(len(self.lr_history)), self.lr_history, 'b-', linewidth=2)
            plt.xlabel("Step")
        
        plt.ylabel("Learning Rate")
        plt.title(f"Learning Rate Schedule: {self.config.scheduler_type}")
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        
        # Add annotations
        plt.annotate(f'Initial LR: {self.config.initial_lr:.2e}', 
                    xy=(0, self.config.initial_lr), 
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        
        plt.annotate(f'Min LR: {self.config.min_lr:.2e}', 
                    xy=(len(self.lr_history)-1, self.config.min_lr), 
                    xytext=(-10, -10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()


class TrainingOptimizer:
    """Comprehensive training optimization manager with gradient management"""
    
    def __init__(
        self,
        early_stopping_config: EarlyStoppingConfig,
        lr_scheduler_config: LRSchedulerConfig,
        gradient_config: GradientManagementConfig,
        optimizer: Optimizer
    ):
        
    """__init__ function."""
self.early_stopping = EarlyStopping(early_stopping_config)
        self.lr_scheduler = AdvancedLRScheduler(lr_scheduler_config, optimizer)
        self.optimizer = optimizer
        
        # Initialize gradient management
        if gradient_config.enable_gradient_management:
            self.gradient_manager = create_gradient_manager(
                max_grad_norm=gradient_config.max_grad_norm,
                enable_monitoring=gradient_config.enable_gradient_monitoring,
                enable_nan_inf_check=gradient_config.enable_nan_inf_check,
                verbose=gradient_config.verbose_logging
            )
        else:
            self.gradient_manager = None
        
        # Performance tracking
        self.optimization_stats = defaultdict(list)
        self.gradient_history = []
        
        logger.info("Training Optimizer initialized with gradient management")
    
    async def optimize_training(
        self,
        model: nn.Module,
        train_func: Callable,
        val_func: Callable,
        max_epochs: int = 100
    ) -> Dict[str, Any]:
        """Optimize training with early stopping, LR scheduling, and gradient management"""
        
        training_history = {
            "train_loss": [],
            "val_loss": [],
            "learning_rate": [],
            "epoch": [],
            "gradient_stats": []
        }
        
        for epoch in range(max_epochs):
            # Training with gradient management
            if self.gradient_manager:
                train_loss = await self._train_with_gradient_management(model, train_func, epoch)
            else:
                train_loss = await train_func(model, epoch)
            
            # Validation
            val_loss = await val_func(model, epoch)
            
            # Update learning rate scheduler
            self.lr_scheduler.step_epoch(epoch, val_loss)
            
            # Check early stopping
            should_stop = self.early_stopping(
                monitor_value=val_loss,
                model=model,
                epoch=epoch,
                additional_metrics={"train_loss": train_loss, "lr": self.lr_scheduler.get_lr()}
            )
            
            # Record history
            training_history["train_loss"].append(train_loss)
            training_history["val_loss"].append(val_loss)
            training_history["learning_rate"].append(self.lr_scheduler.get_lr())
            training_history["epoch"].append(epoch)
            
            # Record gradient statistics
            if self.gradient_manager:
                gradient_summary = self.gradient_manager.get_training_summary()
                training_history["gradient_stats"].append(gradient_summary)
            
            logger.info(f"Epoch {epoch+1}/{max_epochs}, "
                       f"Train Loss: {train_loss:.4f}, "
                       f"Val Loss: {val_loss:.4f}, "
                       f"LR: {self.lr_scheduler.get_lr():.2e}")
            
            if should_stop:
                logger.info("Early stopping triggered")
                break
        
        # Restore best weights
        self.early_stopping.restore_best_weights(model)
        
        return training_history
    
    async def _train_with_gradient_management(
        self,
        model: nn.Module,
        train_func: Callable,
        epoch: int
    ) -> float:
        """Train with gradient management"""
        
        # This is a wrapper that integrates gradient management into the training function
        # The actual implementation depends on how train_func is structured
        
        # For now, we'll assume train_func returns the loss
        # In practice, you might need to modify this based on your training loop structure
        
        loss = await train_func(model, epoch)
        
        # Apply gradient management after the training step
        if self.gradient_manager:
            step_info = self.gradient_manager.step(
                model=model,
                optimizer=self.optimizer,
                loss=loss,
                backward=False  # Assume backward was already called in train_func
            )
            
            if step_info and not step_info.get("skipped", False):
                self.gradient_history.append(step_info)
        
        return loss
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Generate comprehensive optimization report"""
        
        report = {
            "early_stopping": self.early_stopping.get_best_checkpoint(),
            "lr_scheduler": {
                "current_lr": self.lr_scheduler.get_lr(),
                "lr_history": self.lr_scheduler.lr_history,
                "scheduler_type": self.lr_scheduler.config.scheduler_type
            },
            "optimization_stats": dict(self.optimization_stats)
        }
        
        # Add gradient management information
        if self.gradient_manager:
            report["gradient_management"] = self.gradient_manager.get_training_summary()
            report["gradient_history"] = self.gradient_history
        
        return report
    
    def plot_optimization_curves(self, save_path: str = None):
        """Plot optimization curves including gradient management"""
        
        # Create subplots for comprehensive visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot early stopping curves
        self.early_stopping.plot_training_curves()
        axes[0, 0].set_title("Training and Validation Loss")
        
        # Plot LR schedule
        self.lr_scheduler.plot_lr_schedule()
        axes[0, 1].set_title("Learning Rate Schedule")
        
        # Plot gradient statistics if available
        if self.gradient_manager and self.gradient_history:
            self.gradient_manager.plot_training_curves()
            axes[1, 0].set_title("Gradient Statistics")
        
        # Plot gradient health over time
        if self.gradient_history:
            health_status = [step.get("health", {}).get("healthy", True) for step in self.gradient_history]
            steps = range(len(health_status))
            axes[1, 1].plot(steps, health_status, 'g-', linewidth=1)
            axes[1, 1].set_title("Gradient Health Over Time")
            axes[1, 1].set_xlabel("Step")
            axes[1, 1].set_ylabel("Healthy")
            axes[1, 1].set_ylim(-0.1, 1.1)
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show() 