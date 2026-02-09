from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from dataclasses import dataclass, field
import logging
import time
import json
import os
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
import warnings
            from transformers import get_cosine_schedule_with_warmup
            from loss_functions import FocalLoss
from typing import Any, List, Dict, Optional
import asyncio
#!/usr/bin/env python3
"""
Model Training and Evaluation Framework for SEO Deep Learning System
- Efficient data loading using PyTorch's DataLoader
- Comprehensive training loops with monitoring
- Advanced evaluation metrics and validation
- Model checkpointing and early stopping
"""


    accuracy_score, precision_recall_fscore_support, 
    confusion_matrix, classification_report, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score
)
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Configuration for model training"""
    # Model configuration
    model: nn.Module = None
    model_name: str = "seo_model"
    
    # Data configuration
    train_dataset: Dataset = None
    val_dataset: Dataset = None
    test_dataset: Dataset = None
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    drop_last: bool = True
    shuffle: bool = True
    
    # Training configuration
    epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    optimizer: str = "adamw"  # "adam", "adamw", "sgd", "rmsprop"
    scheduler: str = "cosine"  # "cosine", "step", "plateau", "warmup_cosine"
    warmup_steps: int = 1000
    gradient_clip_val: float = 1.0
    gradient_accumulation_steps: int = 1
    
    # Loss configuration
    loss_function: str = "cross_entropy"  # "cross_entropy", "focal", "mse", "mae"
    class_weights: Optional[torch.Tensor] = None
    
    # Evaluation configuration
    eval_interval: int = 1
    save_interval: int = 5
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1e-4
    
    # Device configuration
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_mixed_precision: bool = True
    use_distributed: bool = False
    local_rank: int = -1
    
    # Logging configuration
    log_dir: str = "./logs"
    tensorboard: bool = True
    wandb: bool = False
    wandb_project: str = "seo-deep-learning"
    
    # Checkpoint configuration
    checkpoint_dir: str = "./checkpoints"
    save_best_only: bool = True
    max_checkpoints: int = 5
    
    def __post_init__(self) -> Any:
        """Validate and set default values"""
        if self.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            self.device = "cpu"
        
        # Create directories
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

@dataclass
class TrainingMetrics:
    """Container for training metrics"""
    train_loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    train_accuracy: List[float] = field(default_factory=list)
    val_accuracy: List[float] = field(default_factory=list)
    learning_rate: List[float] = field(default_factory=list)
    epoch_times: List[float] = field(default_factory=list)
    best_val_loss: float = float('inf')
    best_val_accuracy: float = 0.0
    current_epoch: int = 0

class EfficientDataLoader:
    """Efficient data loading with PyTorch DataLoader and optimizations"""
    
    def __init__(self, config: TrainingConfig):
        
    """__init__ function."""
self.config = config
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self._setup_data_loaders()
    
    def _setup_data_loaders(self) -> Any:
        """Setup efficient data loaders with optimizations"""
        logger.info("Setting up efficient data loaders...")
        
        # Common DataLoader arguments
        loader_kwargs = {
            'batch_size': self.config.batch_size,
            'num_workers': self.config.num_workers,
            'pin_memory': self.config.pin_memory,
            'drop_last': self.config.drop_last
        }
        
        # Setup train loader
        if self.config.train_dataset is not None:
            if self.config.use_distributed:
                train_sampler = DistributedSampler(
                    self.config.train_dataset,
                    shuffle=self.config.shuffle
                )
                self.train_loader = DataLoader(
                    self.config.train_dataset,
                    sampler=train_sampler,
                    **loader_kwargs
                )
            else:
                self.train_loader = DataLoader(
                    self.config.train_dataset,
                    shuffle=self.config.shuffle,
                    **loader_kwargs
                )
            logger.info(f"Train loader: {len(self.train_loader)} batches")
        
        # Setup validation loader
        if self.config.val_dataset is not None:
            self.val_loader = DataLoader(
                self.config.val_dataset,
                shuffle=False,
                **loader_kwargs
            )
            logger.info(f"Validation loader: {len(self.val_loader)} batches")
        
        # Setup test loader
        if self.config.test_dataset is not None:
            self.test_loader = DataLoader(
                self.config.test_dataset,
                shuffle=False,
                **loader_kwargs
            )
            logger.info(f"Test loader: {len(self.test_loader)} batches")
    
    def get_train_loader(self) -> DataLoader:
        return self.train_loader
    
    def get_val_loader(self) -> DataLoader:
        return self.val_loader
    
    def get_test_loader(self) -> DataLoader:
        return self.test_loader
    
    def set_epoch(self, epoch: int):
        """Set epoch for distributed training"""
        if self.config.use_distributed and self.train_loader is not None:
            self.train_loader.sampler.set_epoch(epoch)

class ModelTrainer:
    """Comprehensive model trainer with monitoring and optimization"""
    
    def __init__(self, config: TrainingConfig):
        
    """__init__ function."""
self.config = config
        self.device = torch.device(config.device)
        self.metrics = TrainingMetrics()
        
        # Setup model
        self.model = config.model.to(self.device)
        if config.use_distributed:
            self.model = DDP(self.model, device_ids=[config.local_rank])
        
        # Setup data loaders
        self.data_loader = EfficientDataLoader(config)
        
        # Setup optimizer
        self.optimizer = self._create_optimizer()
        
        # Setup scheduler
        self.scheduler = self._create_scheduler()
        
        # Setup loss function
        self.criterion = self._create_loss_function()
        
        # Setup mixed precision
        self.scaler = GradScaler() if config.use_mixed_precision else None
        
        # Setup logging
        self.writer = None
        if config.tensorboard:
            self.writer = SummaryWriter(config.log_dir)
        
        # Setup checkpointing
        self.best_model_path = os.path.join(config.checkpoint_dir, "best_model.pth")
        self.latest_model_path = os.path.join(config.checkpoint_dir, "latest_model.pth")
        
        logger.info(f"Model trainer initialized on device: {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer based on configuration"""
        if self.config.optimizer.lower() == "adam":
            return optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer.lower() == "adamw":
            return optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer.lower() == "sgd":
            return optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                momentum=0.9
            )
        elif self.config.optimizer.lower() == "rmsprop":
            return optim.RMSprop(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
    
    def _create_scheduler(self) -> optim.lr_scheduler._LRScheduler:
        """Create learning rate scheduler"""
        if self.config.scheduler.lower() == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.epochs
            )
        elif self.config.scheduler.lower() == "step":
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1
            )
        elif self.config.scheduler.lower() == "plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                verbose=True
            )
        elif self.config.scheduler.lower() == "warmup_cosine":
            total_steps = len(self.data_loader.get_train_loader()) * self.config.epochs
            return get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.config.warmup_steps,
                num_training_steps=total_steps
            )
        else:
            raise ValueError(f"Unknown scheduler: {self.config.scheduler}")
    
    def _create_loss_function(self) -> nn.Module:
        """Create loss function based on configuration"""
        if self.config.loss_function.lower() == "cross_entropy":
            if self.config.class_weights is not None:
                return nn.CrossEntropyLoss(weight=self.config.class_weights.to(self.device))
            return nn.CrossEntropyLoss()
        elif self.config.loss_function.lower() == "focal":
            return FocalLoss(alpha=self.config.class_weights)
        elif self.config.loss_function.lower() == "mse":
            return nn.MSELoss()
        elif self.config.loss_function.lower() == "mae":
            return nn.L1Loss()
        else:
            raise ValueError(f"Unknown loss function: {self.config.loss_function}")
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        train_loader = self.data_loader.get_train_loader()
        if train_loader is None:
            raise ValueError("Train loader is not set up")
        
        # Set epoch for distributed training
        self.data_loader.set_epoch(epoch)
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config.epochs}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            if isinstance(batch, (list, tuple)):
                batch = [b.to(self.device) for b in batch]
            else:
                batch = batch.to(self.device)
            
            # Forward pass with mixed precision
            if self.config.use_mixed_precision:
                with autocast():
                    if isinstance(batch, (list, tuple)):
                        output = self.model(*batch[:-1])  # Assume last element is target
                        target = batch[-1]
                    else:
                        output = self.model(batch)
                        target = batch.get('target', batch.get('labels', None))
                    
                    loss = self.criterion(output, target)
                    loss = loss / self.config.gradient_accumulation_steps
                
                # Backward pass
                self.scaler.scale(loss).backward()
                
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    if self.config.gradient_clip_val > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config.gradient_clip_val
                        )
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                # Standard forward pass
                if isinstance(batch, (list, tuple)):
                    output = self.model(*batch[:-1])
                    target = batch[-1]
                else:
                    output = self.model(batch)
                    target = batch.get('target', batch.get('labels', None))
                
                loss = self.criterion(output, target)
                loss = loss / self.config.gradient_accumulation_steps
                
                # Backward pass
                loss.backward()
                
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    if self.config.gradient_clip_val > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config.gradient_clip_val
                        )
                    
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
            # Update metrics
            total_loss += loss.item() * self.config.gradient_accumulation_steps
            
            # Calculate accuracy
            if target is not None:
                if output.dim() > 1:
                    pred = output.argmax(dim=1)
                else:
                    pred = (output > 0.5).float()
                
                total_correct += (pred == target).sum().item()
                total_samples += target.size(0)
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })
        
        # Calculate epoch metrics
        avg_loss = total_loss / len(train_loader)
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy
        }
    
    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        val_loader = self.data_loader.get_val_loader()
        if val_loader is None:
            return {'loss': 0.0, 'accuracy': 0.0}
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                # Move batch to device
                if isinstance(batch, (list, tuple)):
                    batch = [b.to(self.device) for b in batch]
                else:
                    batch = batch.to(self.device)
                
                # Forward pass
                if self.config.use_mixed_precision:
                    with autocast():
                        if isinstance(batch, (list, tuple)):
                            output = self.model(*batch[:-1])
                            target = batch[-1]
                        else:
                            output = self.model(batch)
                            target = batch.get('target', batch.get('labels', None))
                        
                        loss = self.criterion(output, target)
                else:
                    if isinstance(batch, (list, tuple)):
                        output = self.model(*batch[:-1])
                        target = batch[-1]
                    else:
                        output = self.model(batch)
                        target = batch.get('target', batch.get('labels', None))
                    
                    loss = self.criterion(output, target)
                
                # Update metrics
                total_loss += loss.item()
                
                # Calculate accuracy
                if target is not None:
                    if output.dim() > 1:
                        pred = output.argmax(dim=1)
                    else:
                        pred = (output > 0.5).float()
                    
                    total_correct += (pred == target).sum().item()
                    total_samples += target.size(0)
        
        # Calculate epoch metrics
        avg_loss = total_loss / len(val_loader)
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy
        }
    
    def train(self) -> TrainingMetrics:
        """Main training loop"""
        logger.info("Starting training...")
        
        for epoch in range(self.config.epochs):
            epoch_start_time = time.time()
            
            # Train epoch
            train_metrics = self.train_epoch(epoch)
            
            # Validate epoch
            val_metrics = self.validate_epoch()
            
            # Update learning rate
            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_metrics['loss'])
            else:
                self.scheduler.step()
            
            # Update metrics
            self.metrics.train_loss.append(train_metrics['loss'])
            self.metrics.val_loss.append(val_metrics['loss'])
            self.metrics.train_accuracy.append(train_metrics['accuracy'])
            self.metrics.val_accuracy.append(val_metrics['accuracy'])
            self.metrics.learning_rate.append(self.optimizer.param_groups[0]['lr'])
            self.metrics.epoch_times.append(time.time() - epoch_start_time)
            self.metrics.current_epoch = epoch + 1
            
            # Log metrics
            logger.info(
                f"Epoch {epoch+1}/{self.config.epochs} - "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Train Acc: {train_metrics['accuracy']:.4f}, "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Val Acc: {val_metrics['accuracy']:.4f}, "
                f"LR: {self.optimizer.param_groups[0]['lr']:.6f}"
            )
            
            # TensorBoard logging
            if self.writer is not None:
                self.writer.add_scalar('Loss/Train', train_metrics['loss'], epoch)
                self.writer.add_scalar('Loss/Val', val_metrics['loss'], epoch)
                self.writer.add_scalar('Accuracy/Train', train_metrics['accuracy'], epoch)
                self.writer.add_scalar('Accuracy/Val', val_metrics['accuracy'], epoch)
                self.writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)
            
            # Save checkpoint
            if (epoch + 1) % self.config.save_interval == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch+1}.pth")
            
            # Save best model
            if val_metrics['loss'] < self.metrics.best_val_loss:
                self.metrics.best_val_loss = val_metrics['loss']
                self.save_checkpoint(self.best_model_path, is_best=True)
            
            if val_metrics['accuracy'] > self.metrics.best_val_accuracy:
                self.metrics.best_val_accuracy = val_metrics['accuracy']
            
            # Early stopping
            if self._should_stop_early():
                logger.info("Early stopping triggered")
                break
        
        # Save final model
        self.save_checkpoint(self.latest_model_path)
        
        logger.info("Training completed!")
        return self.metrics
    
    def _should_stop_early(self) -> bool:
        """Check if early stopping should be triggered"""
        if len(self.metrics.val_loss) < self.config.early_stopping_patience:
            return False
        
        recent_losses = self.metrics.val_loss[-self.config.early_stopping_patience:]
        min_loss = min(recent_losses)
        current_loss = recent_losses[-1]
        
        return (min_loss - current_loss) < self.config.early_stopping_min_delta
    
    def save_checkpoint(self, filename: str, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.metrics.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': self.metrics,
            'config': self.config
        }
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        filepath = os.path.join(self.config.checkpoint_dir, filename)
        torch.save(checkpoint, filepath)
        
        if is_best:
            logger.info(f"Best model saved: {filepath}")
        else:
            logger.info(f"Checkpoint saved: {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.metrics = checkpoint['metrics']
        
        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        logger.info(f"Checkpoint loaded: {filepath}")
        return checkpoint['epoch']

class ModelEvaluator:
    """Comprehensive model evaluation with multiple metrics"""
    
    def __init__(self, model: nn.Module, device: str = "cuda"):
        
    """__init__ function."""
self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()
    
    def evaluate(self, data_loader: DataLoader, task_type: str = "classification") -> Dict[str, float]:
        """Evaluate model on given data loader"""
        all_predictions = []
        all_targets = []
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Evaluating"):
                # Move batch to device
                if isinstance(batch, (list, tuple)):
                    batch = [b.to(self.device) for b in batch]
                else:
                    batch = batch.to(self.device)
                
                # Forward pass
                if isinstance(batch, (list, tuple)):
                    output = self.model(*batch[:-1])
                    target = batch[-1]
                else:
                    output = self.model(batch)
                    target = batch.get('target', batch.get('labels', None))
                
                # Store predictions and targets
                if output.dim() > 1:
                    pred = output.argmax(dim=1).cpu().numpy()
                else:
                    pred = (output > 0.5).float().cpu().numpy()
                
                target_np = target.cpu().numpy()
                
                all_predictions.extend(pred)
                all_targets.extend(target_np)
        
        # Calculate metrics based on task type
        if task_type == "classification":
            return self._calculate_classification_metrics(all_predictions, all_targets)
        elif task_type == "regression":
            return self._calculate_regression_metrics(all_predictions, all_targets)
        else:
            raise ValueError(f"Unknown task type: {task_type}")
    
    def _calculate_classification_metrics(self, predictions: List, targets: List) -> Dict[str, float]:
        """Calculate classification metrics"""
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        # Basic metrics
        accuracy = accuracy_score(targets, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            targets, predictions, average='weighted'
        )
        
        # ROC AUC (for binary classification)
        if len(np.unique(targets)) == 2:
            try:
                roc_auc = roc_auc_score(targets, predictions)
            except:
                roc_auc = 0.0
        else:
            roc_auc = 0.0
        
        # Confusion matrix
        cm = confusion_matrix(targets, predictions)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': cm
        }
    
    def _calculate_regression_metrics(self, predictions: List, targets: List) -> Dict[str, float]:
        """Calculate regression metrics"""
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        mse = mean_squared_error(targets, predictions)
        mae = mean_absolute_error(targets, predictions)
        r2 = r2_score(targets, predictions)
        
        return {
            'mse': mse,
            'mae': mae,
            'r2_score': r2,
            'rmse': np.sqrt(mse)
        }
    
    def generate_report(self, data_loader: DataLoader, task_type: str = "classification") -> str:
        """Generate detailed evaluation report"""
        metrics = self.evaluate(data_loader, task_type)
        
        report = f"Model Evaluation Report\n"
        report += f"=" * 50 + "\n"
        report += f"Task Type: {task_type}\n"
        report += f"Device: {self.device}\n\n"
        
        if task_type == "classification":
            report += f"Accuracy: {metrics['accuracy']:.4f}\n"
            report += f"Precision: {metrics['precision']:.4f}\n"
            report += f"Recall: {metrics['recall']:.4f}\n"
            report += f"F1 Score: {metrics['f1_score']:.4f}\n"
            if metrics['roc_auc'] > 0:
                report += f"ROC AUC: {metrics['roc_auc']:.4f}\n"
        else:
            report += f"MSE: {metrics['mse']:.4f}\n"
            report += f"MAE: {metrics['mae']:.4f}\n"
            report += f"R² Score: {metrics['r2_score']:.4f}\n"
            report += f"RMSE: {metrics['rmse']:.4f}\n"
        
        return report

# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example: Create a simple dataset and model
    class SimpleDataset(Dataset):
        def __init__(self, num_samples=1000, num_features=10, num_classes=2) -> Any:
            self.data = torch.randn(num_samples, num_features)
            self.labels = torch.randint(0, num_classes, (num_samples,))
        
        def __len__(self) -> Any:
            return len(self.data)
        
        def __getitem__(self, idx) -> Optional[Dict[str, Any]]:
            return self.data[idx], self.labels[idx]
    
    class SimpleModel(nn.Module):
        def __init__(self, input_size=10, num_classes=2) -> Any:
            super().__init__()
            self.fc = nn.Linear(input_size, num_classes)
        
        def forward(self, x) -> Any:
            return self.fc(x)
    
    # Create datasets
    train_dataset = SimpleDataset(1000, 10, 2)
    val_dataset = SimpleDataset(200, 10, 2)
    test_dataset = SimpleDataset(200, 10, 2)
    
    # Create model
    model = SimpleModel(10, 2)
    
    # Create training config
    config = TrainingConfig(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        epochs=10,
        batch_size=32,
        learning_rate=1e-3
    )
    
    # Create trainer and train
    trainer = ModelTrainer(config)
    metrics = trainer.train()
    
    # Evaluate on test set
    evaluator = ModelEvaluator(model)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    test_metrics = evaluator.evaluate(test_loader, task_type="classification")
    
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test F1 Score: {test_metrics['f1_score']:.4f}") 