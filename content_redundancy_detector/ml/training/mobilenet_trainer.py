"""
MobileNet Training Utilities
Following PyTorch best practices for training with proper data loading, evaluation, and mixed precision
"""

import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from typing import Dict, Any, Optional
from tqdm import tqdm
import time
from pathlib import Path

from ..models.mobilenet.config import TrainingConfig
from .evaluation import ModelEvaluator
from .callbacks import CallbackList

logger = logging.getLogger(__name__)


class MobileNetTrainer:
    """
    MobileNet Training Class
    Handles training, validation, and evaluation with proper PyTorch patterns
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        training_config: Optional[TrainingConfig] = None,
    ):
        """
        Initialize trainer
        
        Args:
            model: MobileNet model
            device: Training device
            training_config: Training configuration
        """
        self.model = model.to(device)
        self.device = device
        
        if training_config is None:
            training_config = TrainingConfig()
        
        self.config = training_config
        self.use_mixed_precision = training_config.use_mixed_precision
        self.scaler = GradScaler() if self.use_mixed_precision and device.type == "cuda" else None
        
        # Initialize evaluator
        self.evaluator = ModelEvaluator(model, device, self.use_mixed_precision)
        
        # Initialize callbacks
        self.callbacks = CallbackList()
        
        logger.info(f"Initialized MobileNet trainer on {device}, mixed precision: {self.use_mixed_precision}")
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        epoch: int,
        gradient_clip: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Train for one epoch
        
        Args:
            train_loader: Training data loader
            criterion: Loss function
            optimizer: Optimizer
            epoch: Current epoch number
            gradient_clip: Gradient clipping value
            
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.use_mixed_precision and self.device.type == "cuda":
                with autocast():
                    outputs = self.model(inputs)
                    loss = criterion(outputs, targets)
                
                # Backward pass with scaling
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if gradient_clip:
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip)
                
                # Optimizer step
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                
                loss.backward()
                
                if gradient_clip:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip)
                
                optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{running_loss / (batch_idx + 1):.4f}',
                'acc': f'{100. * correct / total:.2f}%'
            })
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        
        return {
            "loss": epoch_loss,
            "accuracy": epoch_acc,
        }
    
    def validate(
        self,
        val_loader: DataLoader,
        criterion: nn.Module,
    ) -> Dict[str, float]:
        """
        Validate model
        
        Args:
            val_loader: Validation data loader
            criterion: Loss function
            
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc="Validation"):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                if self.use_mixed_precision and self.device.type == "cuda":
                    with autocast():
                        outputs = self.model(inputs)
                        loss = criterion(outputs, targets)
                else:
                    outputs = self.model(inputs)
                    loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = 100. * correct / total
        
        return {
            "loss": val_loss,
            "accuracy": val_acc,
        }
    
    def add_callback(self, callback) -> None:
        """Add a callback to the callback list"""
        self.callbacks.append(callback)
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        checkpoint_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Full training loop
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            checkpoint_dir: Directory to save checkpoints
            
        Returns:
            Dictionary with training history
        """
        # Setup optimizer
        optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.config.learning_rate,
            momentum=self.config.momentum,
            weight_decay=self.config.weight_decay
        )
        
        # Setup loss function
        criterion = nn.CrossEntropyLoss()
        
        # Setup scheduler
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.config.lr_scheduler_step,
            gamma=self.config.lr_scheduler_gamma
        )
        
        # Training history
        history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }
        
        best_val_acc = 0.0
        patience_counter = 0
        
        logger.info(f"Starting training for {self.config.num_epochs} epochs")
        start_time = time.time()
        
        # Call callbacks on train begin
        self.callbacks.on_train_begin(self, {'num_epochs': self.config.num_epochs})
        
        for epoch in range(self.config.num_epochs):
            # Call callbacks on epoch begin
            self.callbacks.on_epoch_begin(epoch)
            # Train
            train_metrics = self.train_epoch(
                train_loader,
                criterion,
                optimizer,
                epoch + 1,
                self.config.gradient_clip
            )
            history["train_loss"].append(train_metrics["loss"])
            history["train_acc"].append(train_metrics["accuracy"])
            
            # Validate
            if val_loader:
                val_metrics = self.validate(val_loader, criterion)
                history["val_loss"].append(val_metrics["loss"])
                history["val_acc"].append(val_metrics["accuracy"])
                
                # Check for improvement
                if val_metrics["accuracy"] > best_val_acc:
                    best_val_acc = val_metrics["accuracy"]
                    patience_counter = 0
                    
                    # Save checkpoint
                    if checkpoint_dir:
                        checkpoint_dir.mkdir(parents=True, exist_ok=True)
                        checkpoint_path = checkpoint_dir / f"best_model_epoch_{epoch + 1}.pth"
                        torch.save({
                            'epoch': epoch + 1,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'val_acc': val_metrics["accuracy"],
                            'val_loss': val_metrics["loss"],
                        }, checkpoint_path)
                        logger.info(f"Saved best model checkpoint: {checkpoint_path}")
                else:
                    patience_counter += 1
                
                logger.info(
                    f"Epoch {epoch + 1}/{self.config.num_epochs} - "
                    f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.2f}% - "
                    f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.2f}%"
                )
                
                # Early stopping
                if self.config.early_stopping_patience and patience_counter >= self.config.early_stopping_patience:
                    logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                    break
            else:
                logger.info(
                    f"Epoch {epoch + 1}/{self.config.num_epochs} - "
                    f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.2f}%"
                )
            
            # Update scheduler
            scheduler.step()
            
            # Call callbacks on epoch end
            epoch_logs = {
                'model': self.model,
                'optimizer': optimizer,
                'scheduler': scheduler,
                'train_loss': train_metrics['loss'],
                'train_acc': train_metrics['accuracy'],
            }
            if val_loader:
                epoch_logs.update({
                    'val_loss': val_metrics['loss'],
                    'val_acc': val_metrics['accuracy'],
                })
            
            should_stop = self.callbacks.on_epoch_end(epoch, epoch_logs)
            if should_stop:
                break
        
        # Call callbacks on train end
        self.callbacks.on_train_end(self, {'training_time': time.time() - start_time})
        
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f} seconds")
        
        return {
            "history": history,
            "best_val_acc": best_val_acc if val_loader else None,
            "training_time": training_time,
            "num_epochs": epoch + 1,
        }
    
    def evaluate(
        self,
        test_loader: DataLoader,
        criterion: Optional[nn.Module] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate model on test set
        
        Args:
            test_loader: Test data loader
            criterion: Loss function (optional)
            
        Returns:
            Dictionary with evaluation metrics
        """
        if criterion is None:
            criterion = nn.CrossEntropyLoss()
        
        return self.evaluator.evaluate(test_loader, criterion, return_predictions=False)

