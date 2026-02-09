from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import logging
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import wandb
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
    import torch.nn as nn
from typing import Any, List, Dict, Optional
import asyncio
#!/usr/bin/env python3
"""
Base Trainer Class
Foundation class for all training implementations in the modular deep learning system.
"""


logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for training."""
    # Training parameters
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    gradient_clipping: bool = True
    max_grad_norm: float = 1.0
    
    # Optimization parameters
    optimizer: str = "adam"  # adam, sgd, adamw, rmsprop
    scheduler: str = "cosine"  # cosine, step, exponential, plateau
    scheduler_params: Dict[str, Any] = field(default_factory=dict)
    
    # Loss parameters
    loss_function: str = "cross_entropy"  # cross_entropy, mse, bce, focal
    loss_params: Dict[str, Any] = field(default_factory=dict)
    
    # Device parameters
    device: str = "auto"  # auto, cpu, cuda
    mixed_precision: bool = True
    num_workers: int = 4
    
    # Logging parameters
    log_interval: int = 10
    save_interval: int = 10
    eval_interval: int = 1
    tensorboard: bool = True
    wandb: bool = False
    wandb_project: str = "deep_learning_project"
    
    # Checkpoint parameters
    save_dir: str = "checkpoints"
    resume_from: Optional[str] = None
    save_best_only: bool = True
    save_last: bool = True
    
    # Early stopping parameters
    early_stopping: bool = True
    patience: int = 10
    min_delta: float = 1e-4
    
    # Validation parameters
    validation_split: float = 0.2
    validation_metrics: List[str] = field(default_factory=lambda: ["accuracy", "loss"])
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'gradient_clipping': self.gradient_clipping,
            'max_grad_norm': self.max_grad_norm,
            'optimizer': self.optimizer,
            'scheduler': self.scheduler,
            'scheduler_params': self.scheduler_params,
            'loss_function': self.loss_function,
            'loss_params': self.loss_params,
            'device': self.device,
            'mixed_precision': self.mixed_precision,
            'num_workers': self.num_workers,
            'log_interval': self.log_interval,
            'save_interval': self.save_interval,
            'eval_interval': self.eval_interval,
            'tensorboard': self.tensorboard,
            'wandb': self.wandb,
            'wandb_project': self.wandb_project,
            'save_dir': self.save_dir,
            'resume_from': self.resume_from,
            'save_best_only': self.save_best_only,
            'save_last': self.save_last,
            'early_stopping': self.early_stopping,
            'patience': self.patience,
            'min_delta': self.min_delta,
            'validation_split': self.validation_split,
            'validation_metrics': self.validation_metrics
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TrainingConfig':
        """Create config from dictionary."""
        return cls(**config_dict)
    
    def save(self, filepath: str):
        """Save config to file."""
        with open(filepath, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'TrainingConfig':
        """Load config from file."""
        with open(filepath, 'r') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


class BaseTrainer(ABC):
    """Base trainer class for all training implementations."""
    
    def __init__(self, model: nn.Module, train_dataset: data.Dataset, 
                 val_dataset: Optional[data.Dataset] = None, config: TrainingConfig = None):
        
    """__init__ function."""
self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config or TrainingConfig()
        
        # Setup device
        self.device = self._setup_device()
        self.model.to(self.device)
        
        # Setup data loaders
        self.train_loader = self._create_dataloader(train_dataset, shuffle=True)
        self.val_loader = self._create_dataloader(val_dataset, shuffle=False) if val_dataset else None
        
        # Setup training components
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        self.criterion = self._setup_criterion()
        self.scaler = torch.cuda.amp.GradScaler() if self.config.mixed_precision else None
        
        # Setup logging
        self.writer = self._setup_tensorboard()
        self.wandb_run = self._setup_wandb()
        
        # Training state
        self.current_epoch = 0
        self.best_metric = float('inf')
        self.best_epoch = 0
        self.early_stopping_counter = 0
        self.training_history = {
            'train_loss': [],
            'train_metrics': [],
            'val_loss': [],
            'val_metrics': []
        }
        
        # Create save directory
        Path(self.config.save_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized trainer with {len(train_dataset)} training samples")
        if val_dataset:
            logger.info(f"Validation dataset with {len(val_dataset)} samples")
    
    def _setup_device(self) -> torch.device:
        """Setup device for training."""
        if self.config.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(self.config.device)
        
        logger.info(f"Using device: {device}")
        return device
    
    def _create_dataloader(self, dataset: data.Dataset, shuffle: bool) -> data.DataLoader:
        """Create data loader for dataset."""
        return data.DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=self.config.num_workers,
            pin_memory=True if self.device.type == 'cuda' else False,
            drop_last=shuffle
        )
    
    def _setup_optimizer(self) -> optim.Optimizer:
        """Setup optimizer."""
        optimizer_map = {
            'adam': optim.Adam,
            'adamw': optim.AdamW,
            'sgd': optim.SGD,
            'rmsprop': optim.RMSprop
        }
        
        optimizer_class = optimizer_map.get(self.config.optimizer.lower(), optim.Adam)
        
        if self.config.optimizer.lower() == 'sgd':
            optimizer = optimizer_class(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                momentum=0.9
            )
        else:
            optimizer = optimizer_class(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        
        logger.info(f"Using optimizer: {self.config.optimizer}")
        return optimizer
    
    def _setup_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Setup learning rate scheduler."""
        if not self.config.scheduler:
            return None
        
        scheduler_map = {
            'cosine': optim.lr_scheduler.CosineAnnealingLR,
            'step': optim.lr_scheduler.StepLR,
            'exponential': optim.lr_scheduler.ExponentialLR,
            'plateau': optim.lr_scheduler.ReduceLROnPlateau
        }
        
        scheduler_class = scheduler_map.get(self.config.scheduler.lower())
        if not scheduler_class:
            logger.warning(f"Unknown scheduler: {self.config.scheduler}")
            return None
        
        if self.config.scheduler.lower() == 'plateau':
            scheduler = scheduler_class(
                self.optimizer,
                mode='min',
                patience=5,
                factor=0.5,
                verbose=True
            )
        elif self.config.scheduler.lower() == 'step':
            scheduler = scheduler_class(
                self.optimizer,
                step_size=30,
                gamma=0.1
            )
        elif self.config.scheduler.lower() == 'exponential':
            scheduler = scheduler_class(
                self.optimizer,
                gamma=0.95
            )
        else:  # cosine
            scheduler = scheduler_class(
                self.optimizer,
                T_max=self.config.epochs
            )
        
        logger.info(f"Using scheduler: {self.config.scheduler}")
        return scheduler
    
    def _setup_criterion(self) -> nn.Module:
        """Setup loss function."""
        criterion_map = {
            'cross_entropy': nn.CrossEntropyLoss,
            'mse': nn.MSELoss,
            'bce': nn.BCELoss,
            'bce_with_logits': nn.BCEWithLogitsLoss,
            'focal': self._focal_loss
        }
        
        criterion_class = criterion_map.get(self.config.loss_function.lower(), nn.CrossEntropyLoss)
        
        if self.config.loss_function.lower() == 'focal':
            criterion = criterion_class(**self.config.loss_params)
        else:
            criterion = criterion_class()
        
        logger.info(f"Using loss function: {self.config.loss_function}")
        return criterion
    
    def _focal_loss(self, alpha: float = 1.0, gamma: float = 2.0):
        """Focal loss implementation."""
        def focal_loss(inputs, targets) -> Any:
            ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
            pt = torch.exp(-ce_loss)
            focal_loss = alpha * (1 - pt) ** gamma * ce_loss
            return focal_loss.mean()
        return focal_loss
    
    def _setup_tensorboard(self) -> Optional[SummaryWriter]:
        """Setup TensorBoard logging."""
        if not self.config.tensorboard:
            return None
        
        log_dir = Path("logs") / "tensorboard" / time.strftime("%Y%m%d-%H%M%S")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        writer = SummaryWriter(log_dir=str(log_dir))
        logger.info(f"TensorBoard logging to {log_dir}")
        return writer
    
    def _setup_wandb(self) -> Optional[Any]:
        """Setup Weights & Biases logging."""
        if not self.config.wandb:
            return None
        
        try:
            wandb.init(
                project=self.config.wandb_project,
                config=self.config.to_dict(),
                name=f"run_{time.strftime('%Y%m%d_%H%M%S')}"
            )
            logger.info("Weights & Biases logging enabled")
            return wandb
        except Exception as e:
            logger.warning(f"Failed to setup Weights & Biases: {e}")
            return None
    
    def train(self, resume_from: Optional[str] = None):
        """Main training loop."""
        if resume_from:
            self.load_checkpoint(resume_from)
        
        logger.info("Starting training...")
        
        for epoch in range(self.current_epoch, self.config.epochs):
            self.current_epoch = epoch
            
            # Training phase
            train_loss, train_metrics = self._train_epoch()
            
            # Validation phase
            val_loss, val_metrics = self._validate_epoch() if self.val_loader else (0.0, {})
            
            # Update learning rate
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Log metrics
            self._log_metrics(epoch, train_loss, train_metrics, val_loss, val_metrics)
            
            # Save checkpoint
            self._save_checkpoint(epoch, val_loss, val_metrics)
            
            # Early stopping
            if self._should_stop_early(val_loss):
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break
        
        logger.info("Training completed!")
        self._finalize_training()
    
    def _train_epoch(self) -> Tuple[float, Dict[str, float]]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        metrics = {}
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")
        
        for batch_idx, (data, targets) in enumerate(progress_bar):
            data, targets = data.to(self.device), targets.to(self.device)
            
            # Forward pass
            if self.config.mixed_precision:
                with torch.cuda.amp.autocast():
                    outputs = self.model(data)
                    loss = self.criterion(outputs, targets)
            else:
                outputs = self.model(data)
                loss = self.criterion(outputs, targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            
            if self.config.mixed_precision:
                self.scaler.scale(loss).backward()
                if self.config.gradient_clipping:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.config.gradient_clipping:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            batch_metrics = self._calculate_metrics(outputs, targets)
            
            for key, value in batch_metrics.items():
                if key not in metrics:
                    metrics[key] = 0.0
                metrics[key] += value
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })
        
        # Average metrics
        num_batches = len(self.train_loader)
        avg_loss = total_loss / num_batches
        avg_metrics = {key: value / num_batches for key, value in metrics.items()}
        
        return avg_loss, avg_metrics
    
    def _validate_epoch(self) -> Tuple[float, Dict[str, float]]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        metrics = {}
        
        with torch.no_grad():
            for data, targets in tqdm(self.val_loader, desc="Validation"):
                data, targets = data.to(self.device), targets.to(self.device)
                
                if self.config.mixed_precision:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(data)
                        loss = self.criterion(outputs, targets)
                else:
                    outputs = self.model(data)
                    loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                batch_metrics = self._calculate_metrics(outputs, targets)
                
                for key, value in batch_metrics.items():
                    if key not in metrics:
                        metrics[key] = 0.0
                    metrics[key] += value
        
        # Average metrics
        num_batches = len(self.val_loader)
        avg_loss = total_loss / num_batches
        avg_metrics = {key: value / num_batches for key, value in metrics.items()}
        
        return avg_loss, avg_metrics
    
    def _calculate_metrics(self, outputs: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """Calculate metrics for a batch."""
        metrics = {}
        
        # Accuracy
        if outputs.dim() > 1:
            _, predicted = torch.max(outputs, 1)
            accuracy = (predicted == targets).float().mean().item()
            metrics['accuracy'] = accuracy
        
        # Additional metrics can be added here
        return metrics
    
    def _log_metrics(self, epoch: int, train_loss: float, train_metrics: Dict[str, float],
                    val_loss: float, val_metrics: Dict[str, float]):
        """Log metrics to various logging systems."""
        # Update training history
        self.training_history['train_loss'].append(train_loss)
        self.training_history['train_metrics'].append(train_metrics)
        self.training_history['val_loss'].append(val_loss)
        self.training_history['val_metrics'].append(val_metrics)
        
        # Log to console
        logger.info(f"Epoch {epoch + 1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Log to TensorBoard
        if self.writer:
            self.writer.add_scalar('Loss/Train', train_loss, epoch)
            self.writer.add_scalar('Loss/Validation', val_loss, epoch)
            
            for metric_name, metric_value in train_metrics.items():
                self.writer.add_scalar(f'Metrics/Train_{metric_name}', metric_value, epoch)
            
            for metric_name, metric_value in val_metrics.items():
                self.writer.add_scalar(f'Metrics/Val_{metric_name}', metric_value, epoch)
            
            self.writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)
        
        # Log to Weights & Biases
        if self.wandb_run:
            log_dict = {
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'learning_rate': self.optimizer.param_groups[0]['lr']
            }
            
            for metric_name, metric_value in train_metrics.items():
                log_dict[f'train_{metric_name}'] = metric_value
            
            for metric_name, metric_value in val_metrics.items():
                log_dict[f'val_{metric_name}'] = metric_value
            
            wandb.log(log_dict)
    
    def _save_checkpoint(self, epoch: int, val_loss: float, val_metrics: Dict[str, float]):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'config': self.config.to_dict(),
            'training_history': self.training_history,
            'val_loss': val_loss,
            'val_metrics': val_metrics
        }
        
        # Save best model
        if self.config.save_best_only and val_loss < self.best_metric:
            self.best_metric = val_loss
            self.best_epoch = epoch
            best_path = Path(self.config.save_dir) / "best_model.pth"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model at epoch {epoch} with val_loss: {val_loss:.4f}")
        
        # Save last model
        if self.config.save_last:
            last_path = Path(self.config.save_dir) / "last_model.pth"
            torch.save(checkpoint, last_path)
        
        # Save periodic checkpoints
        if (epoch + 1) % self.config.save_interval == 0:
            periodic_path = Path(self.config.save_dir) / f"checkpoint_epoch_{epoch + 1}.pth"
            torch.save(checkpoint, periodic_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if checkpoint['scheduler_state_dict'] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if checkpoint['scaler_state_dict'] and self.scaler:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.training_history = checkpoint.get('training_history', self.training_history)
        self.best_metric = checkpoint.get('val_loss', float('inf'))
        
        logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")
    
    def _should_stop_early(self, val_loss: float) -> bool:
        """Check if training should stop early."""
        if not self.config.early_stopping:
            return False
        
        if val_loss < self.best_metric - self.config.min_delta:
            self.best_metric = val_loss
            self.early_stopping_counter = 0
        else:
            self.early_stopping_counter += 1
        
        return self.early_stopping_counter >= self.config.patience
    
    def _finalize_training(self) -> Any:
        """Finalize training and cleanup."""
        if self.writer:
            self.writer.close()
        
        if self.wandb_run:
            wandb.finish()
        
        # Save final training history
        history_path = Path(self.config.save_dir) / "training_history.json"
        with open(history_path, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            json.dump(self.training_history, f, indent=2)
        
        # Plot training curves
        self._plot_training_curves()
        
        logger.info("Training finalized successfully!")
    
    def _plot_training_curves(self) -> Any:
        """Plot training curves."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss curves
        axes[0, 0].plot(self.training_history['train_loss'], label='Train Loss')
        if self.training_history['val_loss']:
            axes[0, 0].plot(self.training_history['val_loss'], label='Validation Loss')
        axes[0, 0].set_title('Loss Curves')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy curves
        if self.training_history['train_metrics']:
            train_acc = [metrics.get('accuracy', 0) for metrics in self.training_history['train_metrics']]
            axes[0, 1].plot(train_acc, label='Train Accuracy')
            
            if self.training_history['val_metrics']:
                val_acc = [metrics.get('accuracy', 0) for metrics in self.training_history['val_metrics']]
                axes[0, 1].plot(val_acc, label='Validation Accuracy')
            
            axes[0, 1].set_title('Accuracy Curves')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Accuracy')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = Path(self.config.save_dir) / "training_curves.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Training curves saved to {plot_path}")


# Example usage
if __name__ == "__main__":
    # Create a simple model and dataset for testing
    
    class SimpleModel(nn.Module):
        def __init__(self) -> Any:
            super().__init__()
            self.fc = nn.Linear(784, 10)
        
        def forward(self, x) -> Any:
            return self.fc(x.view(x.size(0), -1))
    
    # Create synthetic data
    train_data = torch.randn(1000, 784)
    train_labels = torch.randint(0, 10, (1000,))
    val_data = torch.randn(200, 784)
    val_labels = torch.randint(0, 10, (200,))
    
    class SimpleDataset(data.Dataset):
        def __init__(self, data, labels) -> Any:
            self.data = data
            self.labels = labels
        
        def __len__(self) -> Any:
            return len(self.data)
        
        def __getitem__(self, idx) -> Optional[Dict[str, Any]]:
            return self.data[idx], self.labels[idx]
    
    # Create datasets
    train_dataset = SimpleDataset(train_data, train_labels)
    val_dataset = SimpleDataset(val_data, val_labels)
    
    # Create model
    model = SimpleModel()
    
    # Create training config
    config = TrainingConfig(
        epochs=5,
        batch_size=32,
        learning_rate=0.001,
        save_dir="test_checkpoints"
    )
    
    # Create trainer
    trainer = BaseTrainer(model, train_dataset, val_dataset, config)
    
    # Start training
    trainer.train()
    
    print("Training completed successfully!") 