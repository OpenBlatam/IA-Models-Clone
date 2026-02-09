"""
Optimized Training System for Video-OpusClip

High-performance training with early stopping, learning rate scheduling,
mixed precision training, gradient accumulation, comprehensive monitoring,
and PyTorch debugging tools integration.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import (
    CosineAnnealingLR, CosineAnnealingWarmRestarts, 
    ReduceLROnPlateau, ExponentialLR, StepLR, MultiStepLR,
    OneCycleLR, LinearLR, ChainedScheduler
)
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import structlog
from dataclasses import dataclass, field
import time
import json
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict, deque
import warnings

# Import PyTorch debugging tools
try:
    from pytorch_debug_tools import (
        PyTorchDebugManager, PyTorchDebugConfig, AutogradAnomalyDetector,
        GradientDebugger, PyTorchMemoryDebugger
    )
    PYTORCH_DEBUG_AVAILABLE = True
except ImportError:
    PYTORCH_DEBUG_AVAILABLE = False
    logger.warning("PyTorch debugging tools not available")

logger = structlog.get_logger()

# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class TrainingConfig:
    """Configuration for training."""
    # Basic training parameters
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    gradient_clip_norm: float = 1.0
    
    # Mixed precision
    use_amp: bool = True
    amp_dtype: torch.dtype = torch.float16
    
    # Gradient accumulation
    gradient_accumulation_steps: int = 1
    
    # Early stopping
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1e-4
    early_stopping_mode: str = 'min'  # 'min' for loss, 'max' for accuracy
    
    # Learning rate scheduling
    lr_scheduler: str = 'cosine'  # 'cosine', 'plateau', 'step', 'exponential', 'onecycle'
    lr_warmup_epochs: int = 5
    lr_warmup_factor: float = 0.1
    lr_decay_factor: float = 0.1
    lr_patience: int = 5
    lr_min: float = 1e-7
    
    # Checkpointing
    save_best_model: bool = True
    save_last_model: bool = True
    checkpoint_dir: str = "checkpoints"
    checkpoint_frequency: int = 5
    
    # Monitoring
    log_frequency: int = 100
    eval_frequency: int = 1
    tensorboard_logging: bool = True
    
    # Validation
    validation_split: float = 0.2
    validation_frequency: int = 1
    
    # Device
    device: str = "cuda"
    num_workers: int = 4
    
    # PyTorch debugging
    enable_pytorch_debugging: bool = False
    debug_config: Optional[PyTorchDebugConfig] = None

@dataclass
class TrainingMetrics:
    """Container for training metrics."""
    train_loss: float = 0.0
    train_accuracy: float = 0.0
    val_loss: float = 0.0
    val_accuracy: float = 0.0
    learning_rate: float = 0.0
    epoch: int = 0
    step: int = 0
    time_elapsed: float = 0.0
    memory_used: float = 0.0
    gpu_utilization: float = 0.0

@dataclass
class TrainingState:
    """Training state for resuming training."""
    epoch: int = 0
    step: int = 0
    best_metric: float = float('inf')
    best_epoch: int = 0
    patience_counter: int = 0
    learning_rate: float = 1e-4
    optimizer_state: Optional[Dict] = None
    scheduler_state: Optional[Dict] = None
    scaler_state: Optional[Dict] = None
    model_state: Optional[Dict] = None
    training_history: List[TrainingMetrics] = field(default_factory=list)

# =============================================================================
# EARLY STOPPING
# =============================================================================

class EarlyStopping:
    """Early stopping with configurable patience and monitoring."""
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-4,
        mode: str = 'min',
        restore_best_weights: bool = True,
        verbose: bool = True
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose
        
        self.best_metric = float('inf') if mode == 'min' else float('-inf')
        self.best_epoch = 0
        self.counter = 0
        self.best_weights = None
        self.stopped_epoch = 0
        
        logger.info(f"Early stopping initialized: patience={patience}, mode={mode}")
    
    def __call__(self, epoch: int, metric: float, model: nn.Module) -> bool:
        """Check if training should stop."""
        if self.mode == 'min':
            improved = metric < self.best_metric - self.min_delta
        else:
            improved = metric > self.best_metric + self.min_delta
        
        if improved:
            self.best_metric = metric
            self.best_epoch = epoch
            self.counter = 0
            
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
                
            if self.verbose:
                logger.info(f"Early stopping: New best metric {self.best_metric:.6f} at epoch {epoch}")
        else:
            self.counter += 1
            if self.verbose:
                logger.info(f"Early stopping: No improvement for {self.counter} epochs")
        
        if self.counter >= self.patience:
            self.stopped_epoch = epoch
            if self.verbose:
                logger.info(f"Early stopping triggered at epoch {epoch}")
            return True
        
        return False
    
    def restore_best_weights(self, model: nn.Module):
        """Restore best weights to model."""
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)
            logger.info(f"Restored best weights from epoch {self.best_epoch}")
    
    def get_best_metric(self) -> Tuple[float, int]:
        """Get best metric and corresponding epoch."""
        return self.best_metric, self.best_epoch

# =============================================================================
# LEARNING RATE SCHEDULERS
# =============================================================================

class OptimizedLRScheduler:
    """Optimized learning rate scheduler with warmup and multiple strategies."""
    
    def __init__(
        self,
        optimizer: optim.Optimizer,
        scheduler_type: str = 'cosine',
        total_steps: int = 1000,
        warmup_steps: int = 100,
        warmup_factor: float = 0.1,
        **kwargs
    ):
        self.optimizer = optimizer
        self.scheduler_type = scheduler_type
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.warmup_factor = warmup_factor
        
        # Create warmup scheduler
        if warmup_steps > 0:
            self.warmup_scheduler = LinearLR(
                optimizer,
                start_factor=warmup_factor,
                end_factor=1.0,
                total_iters=warmup_steps
            )
        else:
            self.warmup_scheduler = None
        
        # Create main scheduler
        self.main_scheduler = self._create_main_scheduler(**kwargs)
        
        # Create chained scheduler if warmup is used
        if self.warmup_scheduler is not None:
            self.scheduler = ChainedScheduler([
                self.warmup_scheduler,
                self.main_scheduler
            ])
        else:
            self.scheduler = self.main_scheduler
        
        logger.info(f"LR scheduler initialized: {scheduler_type}, warmup_steps={warmup_steps}")
    
    def _create_main_scheduler(self, **kwargs) -> Any:
        """Create main learning rate scheduler."""
        if self.scheduler_type == 'cosine':
            return CosineAnnealingLR(
                self.optimizer,
                T_max=self.total_steps - self.warmup_steps,
                eta_min=kwargs.get('lr_min', 1e-7)
            )
        
        elif self.scheduler_type == 'cosine_warm_restarts':
            return CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=kwargs.get('T_0', 10),
                T_mult=kwargs.get('T_mult', 2),
                eta_min=kwargs.get('lr_min', 1e-7)
            )
        
        elif self.scheduler_type == 'plateau':
            return ReduceLROnPlateau(
                self.optimizer,
                mode=kwargs.get('mode', 'min'),
                factor=kwargs.get('factor', 0.1),
                patience=kwargs.get('patience', 5),
                min_lr=kwargs.get('min_lr', 1e-7),
                verbose=kwargs.get('verbose', True)
            )
        
        elif self.scheduler_type == 'step':
            return StepLR(
                self.optimizer,
                step_size=kwargs.get('step_size', 30),
                gamma=kwargs.get('gamma', 0.1)
            )
        
        elif self.scheduler_type == 'multistep':
            return MultiStepLR(
                self.optimizer,
                milestones=kwargs.get('milestones', [30, 60, 90]),
                gamma=kwargs.get('gamma', 0.1)
            )
        
        elif self.scheduler_type == 'exponential':
            return ExponentialLR(
                self.optimizer,
                gamma=kwargs.get('gamma', 0.95)
            )
        
        elif self.scheduler_type == 'onecycle':
            return OneCycleLR(
                self.optimizer,
                max_lr=kwargs.get('max_lr', 1e-3),
                total_steps=self.total_steps - self.warmup_steps,
                pct_start=kwargs.get('pct_start', 0.3),
                anneal_strategy=kwargs.get('anneal_strategy', 'cos')
            )
        
        else:
            raise ValueError(f"Unknown scheduler type: {self.scheduler_type}")
    
    def step(self, metrics: Optional[float] = None):
        """Step the scheduler."""
        if self.scheduler_type == 'plateau' and metrics is not None:
            self.scheduler.step(metrics)
        else:
            self.scheduler.step()
    
    def get_last_lr(self) -> List[float]:
        """Get current learning rate."""
        return self.scheduler.get_last_lr()
    
    def state_dict(self) -> Dict:
        """Get scheduler state."""
        return self.scheduler.state_dict()
    
    def load_state_dict(self, state_dict: Dict):
        """Load scheduler state."""
        self.scheduler.load_state_dict(state_dict)

# =============================================================================
# OPTIMIZED TRAINER
# =============================================================================

class OptimizedTrainer:
    """High-performance trainer with early stopping, LR scheduling, and monitoring."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: Optional[torch.utils.data.DataLoader] = None,
        config: Optional[TrainingConfig] = None,
        loss_fn: Optional[nn.Module] = None,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[OptimizedLRScheduler] = None,
        device: str = "cuda"
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config or TrainingConfig()
        self.device = device
        
        # Loss function
        self.loss_fn = loss_fn or nn.CrossEntropyLoss()
        
        # Optimizer
        if optimizer is None:
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        else:
            self.optimizer = optimizer
        
        # Learning rate scheduler
        if scheduler is None:
            total_steps = len(train_loader) * self.config.epochs
            warmup_steps = len(train_loader) * self.config.lr_warmup_epochs
            
            self.scheduler = OptimizedLRScheduler(
                optimizer=self.optimizer,
                scheduler_type=self.config.lr_scheduler,
                total_steps=total_steps,
                warmup_steps=warmup_steps,
                warmup_factor=self.config.lr_warmup_factor,
                lr_min=self.config.lr_min,
                patience=self.config.lr_patience
            )
        else:
            self.scheduler = scheduler
        
        # Mixed precision training
        self.scaler = GradScaler() if self.config.use_amp else None
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=self.config.early_stopping_patience,
            min_delta=self.config.early_stopping_min_delta,
            mode=self.config.early_stopping_mode
        )
        
        # Training state
        self.state = TrainingState()
        
        # Metrics tracking
        self.train_history = []
        self.val_history = []
        
        # Setup checkpointing
        self.checkpoint_dir = Path(self.config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # PyTorch debugging setup
        self.debug_manager = None
        if self.config.enable_pytorch_debugging and PYTORCH_DEBUG_AVAILABLE:
            debug_config = self.config.debug_config or PyTorchDebugConfig(
                enable_autograd_anomaly=True,
                enable_gradient_debugging=True,
                enable_memory_debugging=True,
                enable_training_debugging=True
            )
            self.debug_manager = PyTorchDebugManager(debug_config)
            logger.info("PyTorch debugging enabled")
        
        logger.info(f"Trainer initialized: {self.config.epochs} epochs, "
                   f"batch_size={self.config.batch_size}, lr={self.config.learning_rate}")
    
    def train_epoch(self, epoch: int) -> TrainingMetrics:
        """Train for one epoch."""
        self.model.train()
        
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        num_batches = 0
        
        start_time = time.time()
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move batch to device
            if isinstance(batch, (list, tuple)):
                inputs = batch[0].to(self.device)
                targets = batch[1].to(self.device)
            else:
                inputs = batch.video_frames.to(self.device)
                targets = batch.labels.to(self.device) if hasattr(batch, 'labels') else None
            
            # Forward pass with mixed precision
            if self.config.use_amp:
                with autocast(dtype=self.config.amp_dtype):
                    outputs = self.model(inputs)
                    if targets is not None:
                        loss = self.loss_fn(outputs, targets)
                    else:
                        loss = outputs.loss if hasattr(outputs, 'loss') else outputs
            else:
                outputs = self.model(inputs)
                if targets is not None:
                    loss = self.loss_fn(outputs, targets)
                else:
                    loss = outputs.loss if hasattr(outputs, 'loss') else outputs
            
            # Backward pass with PyTorch debugging
            if self.debug_manager:
                with self.debug_manager.anomaly_detector.detect_anomaly():
                    if self.config.use_amp:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()
                
                # Check gradients for anomalies
                gradient_info = self.debug_manager.gradient_debugger.check_gradients(
                    self.model, batch_idx
                )
                
                # Apply gradient clipping if anomalies detected
                if gradient_info.get('anomalies'):
                    logger.warning(f"Gradient anomalies detected: {gradient_info['anomalies']}")
                    if self.config.gradient_clip_norm > 0:
                        self.debug_manager.gradient_debugger.clip_gradients(
                            self.model, self.config.gradient_clip_norm
                        )
            else:
                if self.config.use_amp:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.config.gradient_clip_norm > 0:
                    if self.config.use_amp:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip_norm
                    )
                
                # Optimizer step
                if self.config.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                
                # Learning rate scheduling
                self.scheduler.step()
            
            # Update metrics
            epoch_loss += loss.item()
            if targets is not None:
                if outputs.dim() > 1:
                    accuracy = (outputs.argmax(dim=1) == targets).float().mean().item()
                else:
                    accuracy = ((outputs > 0.5) == targets).float().mean().item()
                epoch_accuracy += accuracy
            
            num_batches += 1
            
            # Memory debugging
            if self.debug_manager and batch_idx % 100 == 0:
                self.debug_manager.memory_debugger.take_memory_snapshot(
                    f"epoch_{epoch}_batch_{batch_idx}"
                )
            
            # Logging
            if batch_idx % self.config.log_frequency == 0:
                current_lr = self.scheduler.get_last_lr()[0]
                logger.info(f"Epoch {epoch}, Batch {batch_idx}: "
                           f"Loss={loss.item():.4f}, LR={current_lr:.2e}")
        
        # Calculate epoch metrics
        avg_loss = epoch_loss / num_batches
        avg_accuracy = epoch_accuracy / num_batches if num_batches > 0 else 0.0
        
        # Get memory usage
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated() / 1024**3
            gpu_utilization = torch.cuda.utilization()
        else:
            memory_used = 0.0
            gpu_utilization = 0.0
        
        metrics = TrainingMetrics(
            train_loss=avg_loss,
            train_accuracy=avg_accuracy,
            learning_rate=self.scheduler.get_last_lr()[0],
            epoch=epoch,
            step=self.state.step,
            time_elapsed=time.time() - start_time,
            memory_used=memory_used,
            gpu_utilization=gpu_utilization
        )
        
        return metrics
    
    def validate(self, epoch: int) -> TrainingMetrics:
        """Validate the model."""
        if self.val_loader is None:
            return TrainingMetrics(epoch=epoch)
        
        self.model.eval()
        
        val_loss = 0.0
        val_accuracy = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Move batch to device
                if isinstance(batch, (list, tuple)):
                    inputs = batch[0].to(self.device)
                    targets = batch[1].to(self.device)
                else:
                    inputs = batch.video_frames.to(self.device)
                    targets = batch.labels.to(self.device) if hasattr(batch, 'labels') else None
                
                # Forward pass
                if self.config.use_amp:
                    with autocast(dtype=self.config.amp_dtype):
                        outputs = self.model(inputs)
                        if targets is not None:
                            loss = self.loss_fn(outputs, targets)
                        else:
                            loss = outputs.loss if hasattr(outputs, 'loss') else outputs
                else:
                    outputs = self.model(inputs)
                    if targets is not None:
                        loss = self.loss_fn(outputs, targets)
                    else:
                        loss = outputs.loss if hasattr(outputs, 'loss') else outputs
                
                # Update metrics
                val_loss += loss.item()
                if targets is not None:
                    if outputs.dim() > 1:
                        accuracy = (outputs.argmax(dim=1) == targets).float().mean().item()
                    else:
                        accuracy = ((outputs > 0.5) == targets).float().mean().item()
                    val_accuracy += accuracy
                
                num_batches += 1
        
        # Calculate average metrics
        avg_loss = val_loss / num_batches
        avg_accuracy = val_accuracy / num_batches if num_batches > 0 else 0.0
        
        metrics = TrainingMetrics(
            val_loss=avg_loss,
            val_accuracy=avg_accuracy,
            epoch=epoch
        )
        
        return metrics
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'config': self.config,
            'train_history': self.train_history,
            'val_history': self.val_history,
            'best_metric': self.early_stopping.best_metric,
            'best_epoch': self.early_stopping.best_epoch
        }
        
        # Save last checkpoint
        if self.config.save_last_model:
            checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Saved checkpoint: {checkpoint_path}")
        
        # Save best checkpoint
        if is_best and self.config.save_best_model:
            best_path = self.checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model: {best_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.scaler and checkpoint['scaler_state_dict']:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.train_history = checkpoint.get('train_history', [])
        self.val_history = checkpoint.get('val_history', [])
        
        logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    def train(self) -> Dict[str, Any]:
        """Main training loop."""
        logger.info("Starting training...")
        
        start_time = time.time()
        
        for epoch in range(self.config.epochs):
            # Train epoch
            train_metrics = self.train_epoch(epoch)
            self.train_history.append(train_metrics)
            
            # Validation
            if self.val_loader and epoch % self.config.eval_frequency == 0:
                val_metrics = self.validate(epoch)
                self.val_history.append(val_metrics)
                
                # Early stopping
                metric_to_monitor = val_metrics.val_loss if self.config.early_stopping_mode == 'min' else val_metrics.val_accuracy
                should_stop = self.early_stopping(epoch, metric_to_monitor, self.model)
                
                # Logging
                logger.info(f"Epoch {epoch}: "
                           f"Train Loss={train_metrics.train_loss:.4f}, "
                           f"Train Acc={train_metrics.train_accuracy:.4f}, "
                           f"Val Loss={val_metrics.val_loss:.4f}, "
                           f"Val Acc={val_metrics.val_accuracy:.4f}, "
                           f"LR={train_metrics.learning_rate:.2e}")
                
                if should_stop:
                    logger.info("Early stopping triggered")
                    break
            else:
                logger.info(f"Epoch {epoch}: "
                           f"Train Loss={train_metrics.train_loss:.4f}, "
                           f"Train Acc={train_metrics.train_accuracy:.4f}, "
                           f"LR={train_metrics.learning_rate:.2e}")
            
            # Checkpointing
            if epoch % self.config.checkpoint_frequency == 0:
                is_best = epoch == self.early_stopping.best_epoch
                self.save_checkpoint(epoch, is_best)
            
            self.state.step += len(self.train_loader)
        
        # Restore best weights
        self.early_stopping.restore_best_weights(self.model)
        
        # Final checkpoint
        self.save_checkpoint(self.config.epochs - 1, True)
        
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f} seconds")
        
        # Generate debugging report if enabled
        debug_report = None
        if self.debug_manager:
            debug_report = self.debug_manager.generate_comprehensive_report()
            logger.info("PyTorch debugging report generated")
        
        return {
            'train_history': self.train_history,
            'val_history': self.val_history,
            'best_metric': self.early_stopping.best_metric,
            'best_epoch': self.early_stopping.best_epoch,
            'training_time': training_time,
            'debug_report': debug_report
        }
    
    def plot_training_history(self, save_path: Optional[str] = None):
        """Plot training history."""
        if not self.train_history:
            logger.warning("No training history to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss plot
        epochs = [m.epoch for m in self.train_history]
        train_losses = [m.train_loss for m in self.train_history]
        val_losses = [m.val_loss for m in self.val_history] if self.val_history else []
        
        axes[0, 0].plot(epochs, train_losses, label='Train Loss')
        if val_losses:
            val_epochs = [m.epoch for m in self.val_history]
            axes[0, 0].plot(val_epochs, val_losses, label='Val Loss')
        axes[0, 0].set_title('Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy plot
        train_accuracies = [m.train_accuracy for m in self.train_history]
        val_accuracies = [m.val_accuracy for m in self.val_history] if self.val_history else []
        
        axes[0, 1].plot(epochs, train_accuracies, label='Train Accuracy')
        if val_accuracies:
            axes[0, 1].plot(val_epochs, val_accuracies, label='Val Accuracy')
        axes[0, 1].set_title('Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Learning rate plot
        lrs = [m.learning_rate for m in self.train_history]
        axes[1, 0].plot(epochs, lrs)
        axes[1, 0].set_title('Learning Rate')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True)
        
        # Memory usage plot
        memory_usage = [m.memory_used for m in self.train_history]
        axes[1, 1].plot(epochs, memory_usage)
        axes[1, 1].set_title('GPU Memory Usage')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Memory (GB)')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training history plot saved to {save_path}")
        
        plt.show()

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_trainer(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: Optional[torch.utils.data.DataLoader] = None,
    config: Optional[TrainingConfig] = None,
    **kwargs
) -> OptimizedTrainer:
    """Create optimized trainer with default settings."""
    
    if config is None:
        config = TrainingConfig(**kwargs)
    
    return OptimizedTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config
    )

def resume_training(
    trainer: OptimizedTrainer,
    checkpoint_path: str
) -> Dict[str, Any]:
    """Resume training from checkpoint."""
    trainer.load_checkpoint(checkpoint_path)
    return trainer.train()

# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def get_trainer_factory(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: Optional[torch.utils.data.DataLoader] = None,
    **kwargs
):
    """Get trainer factory with default settings."""
    
    def create_trainer(**override_kwargs):
        """Create trainer with overridden parameters."""
        params = {
            'model': model,
            'train_loader': train_loader,
            'val_loader': val_loader,
            **kwargs,
            **override_kwargs
        }
        
        return create_trainer(**params)
    
    return create_trainer

# Global factory instance
trainer_factory = None

def get_global_trainer_factory(model: nn.Module, train_loader, val_loader=None, **kwargs):
    """Get global trainer factory."""
    global trainer_factory
    if trainer_factory is None:
        trainer_factory = get_trainer_factory(model, train_loader, val_loader, **kwargs)
    return trainer_factory 