from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import (
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import logging
import time
import json
from dataclasses import dataclass
import matplotlib.pyplot as plt
import os
from typing import Any, List, Dict, Optional
import asyncio
    StepLR, CosineAnnealingLR, ExponentialLR, ReduceLROnPlateau,
    OneCycleLR, CosineAnnealingWarmRestarts, LambdaLR
)

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for training optimization."""
    learning_rate: float = 1e-4
    weight_decay: float = 0.1
    batch_size: int = 32
    num_epochs: int = 100
    warmup_steps: int = 10
    max_grad_norm: float = 1.0
    scheduler_type: str = "cosine"  # step, cosine, exponential, reduce_on_plateau, onecycle
    patience: int = 5
    min_delta: float = 0.01
    save_best_model: bool = True
    checkpoint_dir: str = "./checkpoints"


class EarlyStopping:
    """Early stopping implementation with patience and minimum delta."""
    def __init__(self, patience: int = 5, min_delta: float = 0.01, restore_best_weights: bool = True):
        
    """__init__ function."""
self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_score = float('-inf')
        self.counter = 0
        self.best_weights = None
        self.should_stop = False
    
    def __call__(self, current_score: float, model: nn.Module) -> bool:
        """Check if training should stop."""
        if current_score > self.best_score + self.min_delta:
            self.best_score = current_score
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                if self.restore_best_weights and self.best_weights is not None:
                    model.load_state_dict(self.best_weights)
        
        return self.should_stop
    
    def reset(self) -> Any:
        """Set early stopping state."""
        self.best_score = float('-inf')
        self.counter = 0
        self.best_weights = None
        self.should_stop = False


class LearningRateScheduler:
    """Comprehensive learning rate scheduler with multiple strategies."""
    def __init__(self, optimizer: optim.Optimizer, config: TrainingConfig, num_training_steps: int):
        
    """__init__ function."""
self.optimizer = optimizer
        self.config = config
        self.num_training_steps = num_training_steps
        self.scheduler = self._create_scheduler()
    
    def _create_scheduler(self) -> Any:
        """Create scheduler based on configuration."""
        if self.config.scheduler_type == "step":
            return StepLR(self.optimizer, step_size=30, gamma=0.1)
        
        elif self.config.scheduler_type == "cosine":
            return CosineAnnealingLR(
                self.optimizer,
                T_max=self.num_training_steps,
                eta_min=1e-6
            )
        
        elif self.config.scheduler_type == "exponential":
            return ExponentialLR(self.optimizer, gamma=0.95)
        
        elif self.config.scheduler_type == "reduce_on_plateau":
            return ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=0.5,
                patience=3,
                min_lr=1e-6
            )
        
        elif self.config.scheduler_type == "onecycle":
            return OneCycleLR(
                self.optimizer,
                max_lr=self.config.learning_rate,
                total_steps=self.num_training_steps,
                pct_start=0.3,
                anneal_strategy='cos'
            )
        
        elif self.config.scheduler_type == "cosine_warmup":
            return CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=10,
                T_mult=2,
                eta_min=1e-6
            )
        
        elif self.config.scheduler_type == "custom":
            return self._create_custom_scheduler()
        
        else:
            logger.warning(f"Unknown scheduler type: {self.config.scheduler_type}")
            return None
    
    def _create_custom_scheduler(self) -> Any:
        """Create custom learning rate scheduler with warmup."""
        def lr_lambda(step) -> Any:
            if step < self.config.warmup_steps:
                return float(step) / float(max(1, self.config.warmup_steps))
            else:
                progress = float(step - self.config.warmup_steps) / float(
                    max(1, self.num_training_steps - self.config.warmup_steps)
                )
                return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))
        
        return LambdaLR(self.optimizer, lr_lambda)
    
    def step(self, metrics: Optional[float] = None):
        """Step the scheduler."""
        if self.scheduler is not None:
            if isinstance(self.scheduler, ReduceLROnPlateau):
                if metrics is not None:
                    self.scheduler.step(metrics)
            else:
                self.scheduler.step()
    
    def get_last_lr(self) -> List[float]:
        """Get current learning rate."""
        if self.scheduler is not None:
            return self.scheduler.get_last_lr()
        return [group['lr'] for group in self.optimizer.param_groups]


class GradientClipping:
    """Gradient clipping implementation."""
    def __init__(self, max_norm: float = 1.0):
        
    """__init__ function."""
self.max_norm = max_norm
    
    def __call__(self, model: nn.Module):
        """Perform gradient clipping."""
        torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_norm)


class TrainingMonitor:
    """Training progress and metrics."""
    def __init__(self, config: TrainingConfig):
        
    """__init__ function."""
self.config = config
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.learning_rates = []
        self.epoch_times = []
        self.start_time = time.time()
    
    def update(self, epoch: int, train_loss: float, val_loss: float, 
               train_acc: float, val_acc: float, lr: float, epoch_time: float):
        """Update training metrics."""
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_accuracies.append(train_acc)
        self.val_accuracies.append(val_acc)
        self.learning_rates.append(lr)
        self.epoch_times.append(epoch_time)
        
        logger.info(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                   f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, LR: {lr:.2f}")
    
    def plot_training_curves(self, save_path: str = "./training_curves.png"):
        """Plot training curves."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss curves
        axes[0, 0].plot(self.train_losses, label='Train Loss')
        axes[0, 0].plot(self.val_losses, label='Val Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy curves
        axes[0, 1].plot(self.train_accuracies, label='Train Accuracy')
        axes[0, 1].plot(self.val_accuracies, label='Val Accuracy')
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Learning rate curve
        axes[1, 0].plot(self.learning_rates)
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True)
        
        # Epoch time curve
        axes[1, 1].plot(self.epoch_times)
        axes[1, 1].set_title('Epoch Training Time')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Time (seconds)')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_metrics(self, save_path: str = "./training_metrics.json"):
        """Save training metrics to JSON."""
        metrics = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'learning_rates': self.learning_rates,
            'epoch_times': self.epoch_times,
            'total_training_time': time.time() - self.start_time
        }
        
        with open(save_path, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            json.dump(metrics, f, indent=2)


class ModelCheckpoint:
    """Model checkpointing with best model saving."""
    def __init__(self, config: TrainingConfig):
        
    """__init__ function."""
self.config = config
        self.best_score = float('-inf')
        self.checkpoint_dir = config.checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
    
    def save_checkpoint(self, model: nn.Module, optimizer: optim.Optimizer, 
                       scheduler: LearningRateScheduler, epoch: int, score: float,
                       monitor: TrainingMonitor, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.scheduler.state_dict() if scheduler.scheduler else None,
            'score': score,
            'config': self.config.__dict__,
            'monitor_data': {
                'train_losses': monitor.train_losses,
                'val_losses': monitor.val_losses,
                'train_accuracies': monitor.train_accuracies,
                'val_accuracies': monitor.val_accuracies,
                'learning_rates': monitor.learning_rates,
                'epoch_times': monitor.epoch_times
            }
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model if score improved
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, "best_model.pt")
            torch.save(checkpoint, best_path)
            self.best_score = score
            logger.info(f"New best model saved with score: {score:.4f}")
    
    def load_checkpoint(self, model: nn.Module, optimizer: optim.Optimizer,
                       scheduler: LearningRateScheduler, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if scheduler.scheduler and checkpoint['scheduler_state_dict']:
            scheduler.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        return checkpoint['epoch'], checkpoint['score'], checkpoint['monitor_data']


class OptimizedTrainer:
    """Optimized trainer with early stopping, LR scheduling, and monitoring."""
    def __init__(self, model: nn.Module, config: TrainingConfig):
        
    """__init__ function."""
self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Initialize components
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        self.criterion = nn.CrossEntropyLoss()
        self.gradient_clipping = GradientClipping(config.max_grad_norm)
        self.early_stopping = EarlyStopping(
            patience=config.patience,
            min_delta=config.min_delta
        )
        self.monitor = TrainingMonitor(config)
        self.checkpoint = ModelCheckpoint(config)
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0      
        for batch in train_loader:
            inputs = {k: v.to(self.device) for k, v in batch.items() if k != 'labels'}
            targets = batch['labels'].to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(**inputs)
            loss = self.criterion(outputs.logits, targets)
            
            loss.backward()
            self.gradient_clipping(self.model)
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.logits, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        return avg_loss, accuracy
    
    def validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0     
        with torch.no_grad():
            for batch in val_loader:
                inputs = {k: v.to(self.device) for k, v in batch.items() if k != 'labels'}
                targets = batch['labels'].to(self.device)
                
                outputs = self.model(**inputs)
                loss = self.criterion(outputs.logits, targets)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.logits, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total
        return avg_loss, accuracy
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict[str, Any]:
        """Complete training loop with optimization."""
        num_training_steps = len(train_loader) * self.config.num_epochs
        scheduler = LearningRateScheduler(self.optimizer, self.config, num_training_steps)
        
        for epoch in range(self.config.num_epochs):
            epoch_start_time = time.time()
            
            # Training
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validation
            val_loss, val_acc = self.validate_epoch(val_loader)
            
            # Update scheduler
            scheduler.step(val_acc)
            current_lr = scheduler.get_last_lr()[0]
            
            # Update monitor
            epoch_time = time.time() - epoch_start_time
            self.monitor.update(epoch, train_loss, val_loss, train_acc, val_acc, current_lr, epoch_time)
            
            # Check early stopping
            is_best = val_acc > self.checkpoint.best_score
            if self.early_stopping(val_acc, self.model):
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break
            
            # Save checkpoint
            if self.config.save_best_model:
                self.checkpoint.save_checkpoint(
                    self.model, self.optimizer, scheduler, epoch, val_acc, self.monitor, is_best
                )
        
        # Save final metrics and plots
        self.monitor.plot_training_curves()
        self.monitor.save_metrics()
        
        return {
            'best_val_accuracy': self.checkpoint.best_score,
            'final_epoch': epoch,
            'training_metrics': self.monitor.get_metrics()
        }


# Example usage functions
def create_optimized_trainer(
    model: nn.Module,
    learning_rate: float = 1e-4,
    batch_size: int = 32,
    num_epochs: int = 100,
    scheduler_type: str = "cosine",
    patience: int = 5
) -> OptimizedTrainer:
    """Create an optimized trainer with best practices."""
    
    config = TrainingConfig(
        learning_rate=learning_rate,
        batch_size=batch_size,
        num_epochs=num_epochs,
        scheduler_type=scheduler_type,
        patience=patience,
        save_best_model=True
    )
    
    return OptimizedTrainer(model, config)


def train_instagram_caption_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    output_dir: str = "./training_output"
) -> Dict[str, Any]:
    """Instagram caption model with optimization."""
    
    config = TrainingConfig(
        learning_rate=2e-5,
        batch_size=16,
        num_epochs=50,
        scheduler_type="cosine_warmup",
        patience=7,
        save_best_model=True,
        checkpoint_dir=f"{output_dir}/checkpoints"
    )
    
    trainer = OptimizedTrainer(model, config)
    results = trainer.train(train_loader, val_loader)
    
    return results 