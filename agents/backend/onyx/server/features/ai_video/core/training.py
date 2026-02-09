from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

import os
import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime
import time
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Optimizer, Adam, AdamW, SGD
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ReduceLROnPlateau
import numpy as np
from tqdm import tqdm
import wandb
from torch.utils.tensorboard import SummaryWriter
from .models import BaseVideoModel, ModelConfig
from .data_loader import DataConfig
    from .models import ModelConfig, create_model
    from .data_loader import DataConfig, create_train_val_test_loaders
from typing import Any, List, Dict, Optional
import asyncio
"""
AI Video Training Module
========================

This module provides a modular structure for training AI video generation models,
including training loops, loss functions, optimizers, and monitoring.

Features:
- Flexible training loops with callbacks
- Multiple loss functions and optimizers
- Progress tracking and logging
- Model checkpointing and early stopping
- Distributed training support
"""



# Import local modules

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for training."""
    
    # Training parameters
    num_epochs: int = 100
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    gradient_clip: float = 1.0
    
    # Optimizer parameters
    optimizer: str = "adam"  # "adam", "adamw", "sgd"
    scheduler: str = "cosine"  # "step", "cosine", "plateau"
    scheduler_params: Dict[str, Any] = field(default_factory=dict)
    
    # Loss parameters
    loss_type: str = "mse"  # "mse", "l1", "perceptual", "adversarial"
    loss_weights: Dict[str, float] = field(default_factory=dict)
    
    # Monitoring parameters
    save_frequency: int = 10
    eval_frequency: int = 5
    log_frequency: int = 100
    early_stopping_patience: int = 20
    
    # Checkpoint parameters
    checkpoint_dir: str = "checkpoints"
    save_best_only: bool = True
    max_checkpoints: int = 5
    
    # Experiment tracking
    use_wandb: bool = False
    use_tensorboard: bool = True
    experiment_name: str = "ai_video_training"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'num_epochs': self.num_epochs,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'gradient_clip': self.gradient_clip,
            'optimizer': self.optimizer,
            'scheduler': self.scheduler,
            'scheduler_params': self.scheduler_params,
            'loss_type': self.loss_type,
            'loss_weights': self.loss_weights,
            'save_frequency': self.save_frequency,
            'eval_frequency': self.eval_frequency,
            'log_frequency': self.log_frequency,
            'early_stopping_patience': self.early_stopping_patience,
            'checkpoint_dir': self.checkpoint_dir,
            'save_best_only': self.save_best_only,
            'max_checkpoints': self.max_checkpoints,
            'use_wandb': self.use_wandb,
            'use_tensorboard': self.use_tensorboard,
            'experiment_name': self.experiment_name
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TrainingConfig':
        """Create config from dictionary."""
        return cls(**config_dict)
    
    def save(self, filepath: str) -> None:
        """Save config to JSON file."""
        with open(filepath, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'TrainingConfig':
        """Load config from JSON file."""
        with open(filepath, 'r') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


class BaseLoss(ABC, nn.Module):
    """Base class for loss functions."""
    
    def __init__(self, config: TrainingConfig):
        
    """__init__ function."""
super().__init__()
        self.config = config
    
    @abstractmethod
    def forward(self, predictions: Tensor, targets: Tensor, **kwargs) -> Tensor:
        """Compute loss. Must be implemented by subclasses."""
        pass


class MSELoss(BaseLoss):
    """Mean Squared Error loss."""
    
    def forward(self, predictions: Tensor, targets: Tensor, **kwargs) -> Tensor:
        """Compute MSE loss."""
        return F.mse_loss(predictions, targets)


class L1Loss(BaseLoss):
    """L1 loss."""
    
    def forward(self, predictions: Tensor, targets: Tensor, **kwargs) -> Tensor:
        """Compute L1 loss."""
        return F.l1_loss(predictions, targets)


class PerceptualLoss(BaseLoss):
    """Perceptual loss using pre-trained VGG features."""
    
    def __init__(self, config: TrainingConfig):
        
    """__init__ function."""
super().__init__(config)
        # Load pre-trained VGG for feature extraction
        self.vgg = self._load_vgg()
        self.vgg.eval()
        
        # Freeze VGG parameters
        for param in self.vgg.parameters():
            param.requires_grad = False
    
    def _load_vgg(self) -> nn.Module:
        """Load pre-trained VGG model."""
        # Simplified VGG-like feature extractor
        return nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
    
    def forward(self, predictions: Tensor, targets: Tensor, **kwargs) -> Tensor:
        """Compute perceptual loss."""
        # Extract features for each frame
        pred_features = []
        target_features = []
        
        for i in range(predictions.shape[2]):  # Iterate over frames
            pred_frame = predictions[:, :, i, :, :]
            target_frame = targets[:, :, i, :, :]
            
            pred_feat = self.vgg(pred_frame)
            target_feat = self.vgg(target_frame)
            
            pred_features.append(pred_feat)
            target_features.append(target_feat)
        
        # Stack features
        pred_features = torch.stack(pred_features, dim=2)
        target_features = torch.stack(target_features, dim=2)
        
        return F.mse_loss(pred_features, target_features)


class AdversarialLoss(BaseLoss):
    """Adversarial loss for GAN training."""
    
    def __init__(self, config: TrainingConfig):
        
    """__init__ function."""
super().__init__(config)
        self.bce_loss = nn.BCELoss()
    
    def forward(self, predictions: Tensor, targets: Tensor, discriminator_outputs: Tensor, **kwargs) -> Tensor:
        """Compute adversarial loss."""
        # For generator: want discriminator to classify generated as real (1)
        real_labels = torch.ones_like(discriminator_outputs)
        return self.bce_loss(discriminator_outputs, real_labels)


class LossFactory:
    """Factory class for creating loss functions."""
    
    _losses = {
        'mse': MSELoss,
        'l1': L1Loss,
        'perceptual': PerceptualLoss,
        'adversarial': AdversarialLoss
    }
    
    @classmethod
    def create_loss(cls, loss_type: str, config: TrainingConfig) -> BaseLoss:
        """Create a loss function instance."""
        if loss_type not in cls._losses:
            raise ValueError(f"Unknown loss type: {loss_type}. Available: {list(cls._losses.keys())}")
        
        loss_class = cls._losses[loss_type]
        return loss_class(config)
    
    @classmethod
    def get_available_losses(cls) -> List[str]:
        """Get list of available loss types."""
        return list(cls._losses.keys())


class TrainingCallback(ABC):
    """Base class for training callbacks."""
    
    def on_epoch_begin(self, epoch: int, trainer: 'VideoTrainer') -> None:
        """Called at the beginning of each epoch."""
        pass
    
    def on_epoch_end(self, epoch: int, trainer: 'VideoTrainer') -> None:
        """Called at the end of each epoch."""
        pass
    
    def on_batch_begin(self, batch_idx: int, trainer: 'VideoTrainer') -> None:
        """Called at the beginning of each batch."""
        pass
    
    def on_batch_end(self, batch_idx: int, trainer: 'VideoTrainer') -> None:
        """Called at the end of each batch."""
        pass


class ProgressCallback(TrainingCallback):
    """Progress tracking callback."""
    
    def __init__(self) -> Any:
        self.epoch_pbar = None
        self.batch_pbar = None
    
    def on_epoch_begin(self, epoch: int, trainer: 'VideoTrainer') -> None:
        """Initialize epoch progress bar."""
        self.epoch_pbar = tqdm(
            total=len(trainer.train_loader),
            desc=f"Epoch {epoch}/{trainer.config.num_epochs}",
            leave=True
        )
    
    def on_epoch_end(self, epoch: int, trainer: 'VideoTrainer') -> None:
        """Close epoch progress bar."""
        if self.epoch_pbar:
            self.epoch_pbar.close()
    
    def on_batch_end(self, batch_idx: int, trainer: 'VideoTrainer') -> None:
        """Update progress bar."""
        if self.epoch_pbar:
            self.epoch_pbar.update(1)
            self.epoch_pbar.set_postfix({
                'loss': f"{trainer.current_loss:.4f}",
                'lr': f"{trainer.current_lr:.6f}"
            })


class LoggingCallback(TrainingCallback):
    """Logging callback for TensorBoard and wandb."""
    
    def __init__(self, config: TrainingConfig, log_dir: str):
        
    """__init__ function."""
self.config = config
        self.log_dir = log_dir
        self.writer = None
        self.wandb_run = None
        
        # Initialize TensorBoard
        if config.use_tensorboard:
            self.writer = SummaryWriter(log_dir)
        
        # Initialize wandb
        if config.use_wandb:
            try:
                self.wandb_run = wandb.init(
                    project=config.experiment_name,
                    config=config.to_dict()
                )
            except Exception as e:
                logger.warning(f"Failed to initialize wandb: {e}")
    
    def on_batch_end(self, batch_idx: int, trainer: 'VideoTrainer') -> None:
        """Log batch metrics."""
        if batch_idx % self.config.log_frequency == 0:
            step = trainer.current_epoch * len(trainer.train_loader) + batch_idx
            
            # Log to TensorBoard
            if self.writer:
                self.writer.add_scalar('Loss/train', trainer.current_loss, step)
                self.writer.add_scalar('Learning_rate', trainer.current_lr, step)
            
            # Log to wandb
            if self.wandb_run:
                wandb.log({
                    'train_loss': trainer.current_loss,
                    'learning_rate': trainer.current_lr,
                    'epoch': trainer.current_epoch,
                    'step': step
                })
    
    def on_epoch_end(self, epoch: int, trainer: 'VideoTrainer') -> None:
        """Log epoch metrics."""
        if self.writer:
            self.writer.add_scalar('Loss/epoch', trainer.epoch_loss, epoch)
            if hasattr(trainer, 'val_loss'):
                self.writer.add_scalar('Loss/val', trainer.val_loss, epoch)
        
        if self.wandb_run:
            log_dict = {
                'epoch_loss': trainer.epoch_loss,
                'epoch': epoch
            }
            if hasattr(trainer, 'val_loss'):
                log_dict['val_loss'] = trainer.val_loss
            wandb.log(log_dict)
    
    def close(self) -> Any:
        """Close logging resources."""
        if self.writer:
            self.writer.close()
        if self.wandb_run:
            self.wandb_run.finish()


class CheckpointCallback(TrainingCallback):
    """Model checkpointing callback."""
    
    def __init__(self, config: TrainingConfig, model: BaseVideoModel):
        
    """__init__ function."""
self.config = config
        self.model = model
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_loss = float('inf')
        self.checkpoints = []
    
    def on_epoch_end(self, epoch: int, trainer: 'VideoTrainer') -> None:
        """Save checkpoint at end of epoch."""
        # Save checkpoint
        if epoch % self.config.save_frequency == 0:
            self._save_checkpoint(trainer, epoch, is_best=False)
        
        # Save best model
        if hasattr(trainer, 'val_loss') and trainer.val_loss < self.best_loss:
            self.best_loss = trainer.val_loss
            self._save_checkpoint(trainer, epoch, is_best=True)
        
        # Clean up old checkpoints
        self._cleanup_checkpoints()
    
    def _save_checkpoint(self, trainer: 'VideoTrainer', epoch: int, is_best: bool) -> None:
        """Save model checkpoint."""
        checkpoint_name = f"checkpoint_epoch_{epoch}.pt"
        if is_best:
            checkpoint_name = "best_model.pt"
        
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        # Save model
        self.model.save_model(
            str(checkpoint_path),
            save_optimizer=True,
            optimizer=trainer.optimizer
        )
        
        # Save training state
        training_state = {
            'epoch': epoch,
            'trainer_state': trainer.get_state(),
            'config': self.config.to_dict()
        }
        
        state_path = self.checkpoint_dir / f"training_state_{epoch}.json"
        with open(state_path, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            json.dump(training_state, f, indent=2)
        
        self.checkpoints.append(checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def _cleanup_checkpoints(self) -> None:
        """Remove old checkpoints if exceeding max count."""
        if len(self.checkpoints) > self.config.max_checkpoints:
            # Remove oldest checkpoint (excluding best model)
            for checkpoint in self.checkpoints[:-self.config.max_checkpoints]:
                if checkpoint.name != "best_model.pt":
                    checkpoint.unlink()
                    logger.info(f"Removed old checkpoint: {checkpoint}")


class VideoTrainer:
    """Main training class for AI video models."""
    
    def __init__(self, 
                 model: BaseVideoModel,
                 train_loader: DataLoader,
                 val_loader: Optional[DataLoader] = None,
                 config: TrainingConfig = None):
        
        
    """__init__ function."""
self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config or TrainingConfig()
        
        # Initialize components
        self.device = model.device
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        self.loss_fn = LossFactory.create_loss(self.config.loss_type, self.config)
        
        # Training state
        self.current_epoch = 0
        self.current_loss = 0.0
        self.current_lr = self.config.learning_rate
        self.epoch_loss = 0.0
        self.val_loss = float('inf')
        
        # Initialize callbacks
        self.callbacks = self._create_callbacks()
        
        logger.info(f"Initialized trainer with {len(self.callbacks)} callbacks")
    
    def _create_optimizer(self) -> Optimizer:
        """Create optimizer based on config."""
        optimizers = {
            'adam': Adam,
            'adamw': AdamW,
            'sgd': SGD
        }
        
        optimizer_class = optimizers.get(self.config.optimizer, Adam)
        
        return optimizer_class(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
    
    def _create_scheduler(self) -> Optional[Any]:
        """Create learning rate scheduler."""
        schedulers = {
            'step': StepLR,
            'cosine': CosineAnnealingLR,
            'plateau': ReduceLROnPlateau
        }
        
        scheduler_class = schedulers.get(self.config.scheduler)
        if scheduler_class is None:
            return None
        
        scheduler_params = self.config.scheduler_params.copy()
        if self.config.scheduler == 'plateau':
            scheduler_params.setdefault('patience', 10)
            scheduler_params.setdefault('factor', 0.5)
        
        return scheduler_class(self.optimizer, **scheduler_params)
    
    def _create_callbacks(self) -> List[TrainingCallback]:
        """Create training callbacks."""
        callbacks = [
            ProgressCallback(),
            LoggingCallback(self.config, "logs/training"),
            CheckpointCallback(self.config, self.model)
        ]
        return callbacks
    
    def train(self) -> Dict[str, List[float]]:
        """Main training loop."""
        logger.info(f"Starting training for {self.config.num_epochs} epochs")
        
        history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }
        
        try:
            for epoch in range(self.current_epoch, self.config.num_epochs):
                self.current_epoch = epoch
                
                # Notify callbacks
                for callback in self.callbacks:
                    callback.on_epoch_begin(epoch, self)
                
                # Training epoch
                train_loss = self._train_epoch()
                history['train_loss'].append(train_loss)
                history['learning_rate'].append(self.current_lr)
                
                # Validation epoch
                if self.val_loader is not None and epoch % self.config.eval_frequency == 0:
                    val_loss = self._validate_epoch()
                    history['val_loss'].append(val_loss)
                    self.val_loss = val_loss
                else:
                    history['val_loss'].append(self.val_loss)
                
                # Update learning rate
                if self.scheduler is not None:
                    if isinstance(self.scheduler, ReduceLROnPlateau):
                        self.scheduler.step(self.val_loss)
                    else:
                        self.scheduler.step()
                    self.current_lr = self.optimizer.param_groups[0]['lr']
                
                # Notify callbacks
                for callback in self.callbacks:
                    callback.on_epoch_end(epoch, self)
                
                logger.info(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {self.val_loss:.4f}")
        
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        
        finally:
            # Clean up callbacks
            for callback in self.callbacks:
                if hasattr(callback, 'close'):
                    callback.close()
        
        return history
    
    def _train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train_mode()
        total_loss = 0.0
        num_batches = len(self.train_loader)
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Notify callbacks
            for callback in self.callbacks:
                callback.on_batch_begin(batch_idx, self)
            
            # Forward pass
            loss = self._training_step(batch)
            total_loss += loss.item()
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if self.config.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
            
            self.optimizer.step()
            
            # Update current loss
            self.current_loss = loss.item()
            self.epoch_loss = total_loss / (batch_idx + 1)
            
            # Notify callbacks
            for callback in self.callbacks:
                callback.on_batch_end(batch_idx, self)
        
        return total_loss / num_batches
    
    def _training_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Single training step."""
        videos = batch['video'].to(self.device)
        
        # Forward pass
        predictions = self.model(videos)
        
        # Compute loss
        loss = self.loss_fn(predictions, videos)
        
        return loss
    
    def _validate_epoch(self) -> float:
        """Validate for one epoch."""
        self.model.eval_mode()
        total_loss = 0.0
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            for batch in self.val_loader:
                videos = batch['video'].to(self.device)
                predictions = self.model(videos)
                loss = self.loss_fn(predictions, videos)
                total_loss += loss.item()
        
        return total_loss / num_batches
    
    def get_state(self) -> Dict[str, Any]:
        """Get current training state."""
        return {
            'current_epoch': self.current_epoch,
            'current_loss': self.current_loss,
            'current_lr': self.current_lr,
            'epoch_loss': self.epoch_loss,
            'val_loss': self.val_loss
        }
    
    def load_state(self, state: Dict[str, Any]) -> None:
        """Load training state."""
        self.current_epoch = state.get('current_epoch', 0)
        self.current_loss = state.get('current_loss', 0.0)
        self.current_lr = state.get('current_lr', self.config.learning_rate)
        self.epoch_loss = state.get('epoch_loss', 0.0)
        self.val_loss = state.get('val_loss', float('inf'))


# Convenience functions
def create_trainer(model: BaseVideoModel,
                  train_loader: DataLoader,
                  val_loader: Optional[DataLoader] = None,
                  config: TrainingConfig = None) -> VideoTrainer:
    """Create a trainer instance."""
    return VideoTrainer(model, train_loader, val_loader, config)


def train_model(model: BaseVideoModel,
                train_loader: DataLoader,
                val_loader: Optional[DataLoader] = None,
                config: TrainingConfig = None) -> Dict[str, List[float]]:
    """Train a model and return training history."""
    trainer = create_trainer(model, train_loader, val_loader, config)
    return trainer.train()


if __name__ == "__main__":
    # Example usage
    
    # Create model
    model_config = ModelConfig(
        model_type="diffusion",
        model_name="test_model",
        frame_size=(64, 64),
        num_frames=8
    )
    model = create_model("diffusion", model_config)
    
    # Create data loaders
    data_config = DataConfig(
        data_dir="data/videos",
        frame_size=(64, 64),
        num_frames=8,
        batch_size=4
    )
    loaders = create_train_val_test_loaders("video_file", data_config)
    
    # Create training config
    train_config = TrainingConfig(
        num_epochs=10,
        batch_size=4,
        learning_rate=1e-4
    )
    
    # Train model
    history = train_model(model, loaders['train'], loaders['val'], train_config)
    print(f"Training completed. Final loss: {history['train_loss'][-1]:.4f}") 