"""
Training Management System
Unified training interface with optimization and monitoring
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass
import time
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Configuration for training"""
    # Training parameters
    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    
    # Optimizer settings
    optimizer: str = "adam"  # adam, sgd, adamw
    scheduler: str = "cosine"  # cosine, step, plateau
    warmup_steps: int = 100
    
    # Training options
    gradient_clip: float = 1.0
    accumulation_steps: int = 1
    mixed_precision: bool = False
    
    # Logging and saving
    save_interval: int = 1000
    log_interval: int = 100
    save_best: bool = True
    
    # Early stopping
    early_stopping_patience: int = 5
    min_delta: float = 0.001

class TrainingManager:
    """Unified training management system"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')
        self.training_history = []
        
        # Metrics tracking
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'epoch_time': []
        }
        
        logger.info(f"Initialized TrainingManager with {config.epochs} epochs")
    
    def setup_training(self, model: nn.Module, train_dataset: Dataset, val_dataset: Optional[Dataset] = None) -> None:
        """Setup training components"""
        self.model = model
        
        # Setup optimizer
        self.optimizer = self._create_optimizer()
        
        # Setup scheduler
        if self.config.scheduler:
            self.scheduler = self._create_scheduler()
        
        # Setup loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Setup data loaders
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=True,
            num_workers=2
        )
        
        if val_dataset:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=2
            )
        else:
            self.val_loader = None
            
        logger.info("Training setup completed")
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer based on config"""
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
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.optimizer}")
    
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
                step_size=self.config.epochs // 3,
                gamma=0.1
            )
        elif self.config.scheduler.lower() == "plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                patience=2,
                factor=0.5
            )
        else:
            raise ValueError(f"Unsupported scheduler: {self.config.scheduler}")
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        start_time = time.time()
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            # Move to device
            data, target = data.to(self.model.device), target.to(self.model.device)
            
            # Forward pass
            if self.config.mixed_precision:
                with torch.cuda.amp.autocast():
                    output = self.model(data)
                    loss = self.criterion(output, target)
            else:
                output = self.model(data)
                loss = self.criterion(output, target)
            
            # Scale loss for gradient accumulation
            loss = loss / self.config.accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.config.accumulation_steps == 0:
                # Gradient clipping
                if self.config.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.gradient_clip
                    )
                
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.global_step += 1
            
            total_loss += loss.item() * self.config.accumulation_steps
            num_batches += 1
            
            # Logging
            if batch_idx % self.config.log_interval == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                logger.info(
                    f"Epoch {self.current_epoch}, Batch {batch_idx}, "
                    f"Loss: {loss.item():.4f}, LR: {current_lr:.6f}"
                )
        
        epoch_time = time.time() - start_time
        avg_loss = total_loss / num_batches
        
        # Update metrics
        self.metrics['train_loss'].append(avg_loss)
        self.metrics['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
        self.metrics['epoch_time'].append(epoch_time)
        
        return {
            'train_loss': avg_loss,
            'epoch_time': epoch_time,
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
    
    def validate(self) -> Dict[str, float]:
        """Validate the model"""
        if self.val_loader is None:
            return {}
            
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.model.device), target.to(self.model.device)
                
                if self.config.mixed_precision:
                    with torch.cuda.amp.autocast():
                        output = self.model(data)
                        loss = self.criterion(output, target)
                else:
                    output = self.model(data)
                    loss = self.criterion(output, target)
                
                total_loss += loss.item()
                
                # Calculate accuracy
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100. * correct / total
        
        # Update metrics
        self.metrics['val_loss'].append(avg_loss)
        
        return {
            'val_loss': avg_loss,
            'val_accuracy': accuracy
        }
    
    def train(self, save_path: Optional[str] = None) -> Dict[str, Any]:
        """Main training loop"""
        logger.info(f"Starting training for {self.config.epochs} epochs")
        
        start_time = time.time()
        patience_counter = 0
        
        for epoch in range(self.config.epochs):
            self.current_epoch = epoch
            
            # Train epoch
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Update scheduler
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics.get('val_loss', train_metrics['train_loss']))
                else:
                    self.scheduler.step()
            
            # Log epoch results
            logger.info(
                f"Epoch {epoch}: Train Loss: {train_metrics['train_loss']:.4f}, "
                f"Val Loss: {val_metrics.get('val_loss', 'N/A'):.4f}, "
                f"Val Acc: {val_metrics.get('val_accuracy', 'N/A'):.2f}%"
            )
            
            # Save best model
            current_loss = val_metrics.get('val_loss', train_metrics['train_loss'])
            if self.config.save_best and current_loss < self.best_loss:
                self.best_loss = current_loss
                if save_path:
                    self.save_checkpoint(save_path, is_best=True)
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= self.config.early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break
        
        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time:.2f} seconds")
        
        return {
            'total_time': total_time,
            'best_loss': self.best_loss,
            'final_epoch': self.current_epoch,
            'metrics': self.metrics
        }
    
    def save_checkpoint(self, save_path: str, is_best: bool = False) -> None:
        """Save training checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_loss': self.best_loss,
            'metrics': self.metrics
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, save_path)
        
        if is_best:
            best_path = save_path.replace('.pth', '_best.pth')
            torch.save(checkpoint, best_path)
            logger.info(f"Best model saved to {best_path}")
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load training checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.model.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_loss = checkpoint['best_loss']
        self.metrics = checkpoint['metrics']
        
        if 'scheduler_state_dict' in checkpoint and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        logger.info(f"Checkpoint loaded from {checkpoint_path}")

