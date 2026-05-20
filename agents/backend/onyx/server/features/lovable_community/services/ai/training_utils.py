"""
Training Utilities with Best Practices

Provides comprehensive training utilities including:
- Early stopping
- Model checkpointing
- Learning rate scheduling
- Gradient accumulation
- Multi-GPU support
- Training metrics
"""

import logging
import os
from typing import Dict, Any, Optional, List, Callable
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import numpy as np
from tqdm import tqdm

from .experiment_tracker import ExperimentTracker

logger = logging.getLogger(__name__)


class EarlyStopping:
    """
    Early stopping utility to stop training when validation loss stops improving
    
    Implements patience-based early stopping with best model saving.
    """
    
    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.0,
        mode: str = "min",
        restore_best_weights: bool = True
    ):
        """
        Initialize early stopping
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss, 'max' for accuracy
            restore_best_weights: Whether to restore best weights on stop
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        self.early_stop = False
        
        if mode == "min":
            self.is_better = lambda current, best: current < (best - min_delta)
        else:
            self.is_better = lambda current, best: current > (best + min_delta)
    
    def __call__(self, score: float, model: nn.Module) -> bool:
        """
        Check if training should stop
        
        Args:
            score: Current validation score
            model: Model to save weights from
            
        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = score
            self._save_checkpoint(model)
        elif self.is_better(score, self.best_score):
            self.best_score = score
            self.counter = 0
            self._save_checkpoint(model)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.restore_best_weights:
                    self._restore_checkpoint(model)
                logger.info(f"Early stopping triggered. Best score: {self.best_score:.4f}")
        
        return self.early_stop
    
    def _save_checkpoint(self, model: nn.Module) -> None:
        """Save model checkpoint"""
        self.best_weights = {
            k: v.cpu().clone() for k, v in model.state_dict().items()
        }
    
    def _restore_checkpoint(self, model: nn.Module) -> None:
        """Restore best model weights"""
        if self.best_weights:
            model.load_state_dict(self.best_weights)
            logger.info("Restored best model weights")


class ModelCheckpoint:
    """
    Model checkpointing utility
    
    Saves model checkpoints based on various criteria.
    """
    
    def __init__(
        self,
        save_dir: str,
        save_best: bool = True,
        save_last: bool = True,
        save_every_n_epochs: Optional[int] = None,
        monitor: str = "val_loss",
        mode: str = "min"
    ):
        """
        Initialize model checkpoint
        
        Args:
            save_dir: Directory to save checkpoints
            save_best: Whether to save best model
            save_last: Whether to save last model
            save_every_n_epochs: Save every N epochs (None to disable)
            monitor: Metric to monitor
            mode: 'min' or 'max'
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.save_best = save_best
        self.save_last = save_last
        self.save_every_n_epochs = save_every_n_epochs
        self.monitor = monitor
        self.mode = mode
        
        self.best_score = None
        self.last_epoch = 0
        
        if mode == "min":
            self.is_better = lambda current, best: current < best if best is not None else True
        else:
            self.is_better = lambda current, best: current > best if best is not None else True
    
    def save(
        self,
        model: nn.Module,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[_LRScheduler] = None,
        epoch: int = 0,
        metrics: Optional[Dict[str, float]] = None,
        tokenizer: Any = None
    ) -> None:
        """
        Save checkpoint
        
        Args:
            model: Model to save
            optimizer: Optimizer state
            scheduler: Scheduler state
            epoch: Current epoch
            metrics: Current metrics
            tokenizer: Tokenizer to save
        """
        self.last_epoch = epoch
        
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "metrics": metrics or {}
        }
        
        if optimizer:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        
        if scheduler:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()
        
        # Save last checkpoint
        if self.save_last:
            last_path = self.save_dir / "last_checkpoint.pt"
            torch.save(checkpoint, last_path)
            logger.info(f"Saved last checkpoint to {last_path}")
        
        # Save periodic checkpoint
        if self.save_every_n_epochs and epoch % self.save_every_n_epochs == 0:
            periodic_path = self.save_dir / f"checkpoint_epoch_{epoch}.pt"
            torch.save(checkpoint, periodic_path)
            logger.info(f"Saved periodic checkpoint to {periodic_path}")
        
        # Save best checkpoint
        if self.save_best and metrics and self.monitor in metrics:
            score = metrics[self.monitor]
            if self.is_better(score, self.best_score):
                self.best_score = score
                best_path = self.save_dir / "best_checkpoint.pt"
                torch.save(checkpoint, best_path)
                logger.info(f"Saved best checkpoint (score: {score:.4f}) to {best_path}")
        
        # Save tokenizer if provided
        if tokenizer and hasattr(tokenizer, "save_pretrained"):
            tokenizer_path = self.save_dir / "tokenizer"
            tokenizer.save_pretrained(str(tokenizer_path))
    
    def load(
        self,
        model: nn.Module,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[_LRScheduler] = None,
        checkpoint_path: Optional[str] = None,
        load_best: bool = False
    ) -> Dict[str, Any]:
        """
        Load checkpoint
        
        Args:
            model: Model to load weights into
            optimizer: Optimizer to load state into
            scheduler: Scheduler to load state into
            checkpoint_path: Path to checkpoint (None for best/last)
            load_best: Whether to load best checkpoint
            
        Returns:
            Dictionary with checkpoint info
        """
        if checkpoint_path is None:
            if load_best:
                checkpoint_path = self.save_dir / "best_checkpoint.pt"
            else:
                checkpoint_path = self.save_dir / "last_checkpoint.pt"
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        model.load_state_dict(checkpoint["model_state_dict"])
        
        if optimizer and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        if scheduler and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        logger.info(f"Epoch: {checkpoint.get('epoch', 'unknown')}")
        logger.info(f"Metrics: {checkpoint.get('metrics', {})}")
        
        return checkpoint


class TrainingMetrics:
    """
    Track and compute training metrics
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self) -> None:
        """Reset all metrics"""
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.learning_rates = []
    
    def update(
        self,
        train_loss: Optional[float] = None,
        val_loss: Optional[float] = None,
        val_accuracy: Optional[float] = None,
        learning_rate: Optional[float] = None
    ) -> None:
        """Update metrics"""
        if train_loss is not None:
            self.train_losses.append(train_loss)
        if val_loss is not None:
            self.val_losses.append(val_loss)
        if val_accuracy is not None:
            self.val_accuracies.append(val_accuracy)
        if learning_rate is not None:
            self.learning_rates.append(learning_rate)
    
    def get_average(self, metric: str, last_n: Optional[int] = None) -> float:
        """Get average of a metric"""
        values = getattr(self, metric, [])
        if not values:
            return 0.0
        
        if last_n:
            values = values[-last_n:]
        
        return np.mean(values)
    
    def get_summary(self) -> Dict[str, float]:
        """Get summary of all metrics"""
        return {
            "avg_train_loss": self.get_average("train_losses"),
            "avg_val_loss": self.get_average("val_losses"),
            "avg_val_accuracy": self.get_average("val_accuracies"),
            "current_lr": self.learning_rates[-1] if self.learning_rates else 0.0
        }


class Trainer:
    """
    Comprehensive trainer with all best practices
    
    Includes:
    - Early stopping
    - Checkpointing
    - Learning rate scheduling
    - Gradient accumulation
    - Mixed precision
    - Multi-GPU support
    - Metrics tracking
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        criterion: nn.Module,
        device: torch.device,
        scheduler: Optional[_LRScheduler] = None,
        use_mixed_precision: bool = True,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        early_stopping: Optional[EarlyStopping] = None,
        checkpoint: Optional[ModelCheckpoint] = None,
        tracker: Optional[ExperimentTracker] = None
    ):
        """
        Initialize trainer
        
        Args:
            model: Model to train
            optimizer: Optimizer
            criterion: Loss function
            device: Device to train on
            scheduler: Learning rate scheduler
            use_mixed_precision: Whether to use FP16
            gradient_accumulation_steps: Gradient accumulation steps
            max_grad_norm: Maximum gradient norm
            early_stopping: Early stopping callback
            checkpoint: Checkpoint callback
            tracker: Experiment tracker
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scheduler = scheduler
        self.use_mixed_precision = use_mixed_precision and device.type == "cuda"
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.early_stopping = early_stopping
        self.checkpoint = checkpoint
        self.tracker = tracker
        
        self.scaler = torch.cuda.amp.GradScaler() if self.use_mixed_precision else None
        self.metrics = TrainingMetrics()
        
        # Move model to device
        self.model.to(device)
    
    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int = 0
    ) -> float:
        """
        Train for one epoch
        
        Args:
            dataloader: Training data loader
            epoch: Current epoch number
            
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(dataloader, desc=f"Training Epoch {epoch + 1}")
        
        for step, batch in enumerate(progress_bar):
            # Move batch to device
            batch = self._move_to_device(batch)
            
            # Forward pass with mixed precision
            with torch.cuda.amp.autocast() if self.use_mixed_precision else torch.no_grad():
                outputs = self.model(**batch)
                loss = outputs.loss if hasattr(outputs, 'loss') else self.criterion(outputs, batch['labels'])
                loss = loss / self.gradient_accumulation_steps
            
            # Backward pass
            if self.use_mixed_precision:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient accumulation
            if (step + 1) % self.gradient_accumulation_steps == 0:
                # Clip gradients
                if self.use_mixed_precision:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                
                if self.scheduler:
                    self.scheduler.step()
            
            total_loss += loss.item() * self.gradient_accumulation_steps
            num_batches += 1
            
            # Update progress bar
            current_lr = self.scheduler.get_last_lr()[0] if self.scheduler else self.optimizer.param_groups[0]['lr']
            progress_bar.set_postfix({
                "loss": loss.item() * self.gradient_accumulation_steps,
                "lr": f"{current_lr:.2e}"
            })
        
        avg_loss = total_loss / num_batches
        self.metrics.update(train_loss=avg_loss, learning_rate=current_lr)
        
        return avg_loss
    
    def validate(
        self,
        dataloader: DataLoader
    ) -> Dict[str, float]:
        """
        Validate the model
        
        Args:
            dataloader: Validation data loader
            
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validating"):
                batch = self._move_to_device(batch)
                
                with torch.cuda.amp.autocast() if self.use_mixed_precision else torch.no_grad():
                    outputs = self.model(**batch)
                    loss = outputs.loss if hasattr(outputs, 'loss') else self.criterion(outputs, batch['labels'])
                    logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                
                total_loss += loss.item()
                
                if 'labels' in batch:
                    predictions = torch.argmax(logits, dim=-1)
                    labels = batch['labels']
                    correct_predictions += (predictions == labels).sum().item()
                    total_predictions += labels.size(0)
        
        metrics = {
            "val_loss": total_loss / len(dataloader),
            "val_accuracy": correct_predictions / total_predictions if total_predictions > 0 else 0.0
        }
        
        self.metrics.update(val_loss=metrics["val_loss"], val_accuracy=metrics["val_accuracy"])
        
        return metrics
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: int = 10,
        tokenizer: Any = None
    ) -> Dict[str, List[float]]:
        """
        Train the model
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs
            tokenizer: Tokenizer to save with checkpoints
            
        Returns:
            Training history
        """
        history = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": []
        }
        
        for epoch in range(num_epochs):
            # Train
            train_loss = self.train_epoch(train_loader, epoch)
            history["train_loss"].append(train_loss)
            
            # Validate
            if val_loader:
                val_metrics = self.validate(val_loader)
                history["val_loss"].append(val_metrics["val_loss"])
                history["val_accuracy"].append(val_metrics["val_accuracy"])
                
                # Log metrics
                if self.tracker:
                    self.tracker.log_metric("train_loss", train_loss, epoch)
                    self.tracker.log_metric("val_loss", val_metrics["val_loss"], epoch)
                    self.tracker.log_metric("val_accuracy", val_metrics["val_accuracy"], epoch)
                
                # Checkpoint
                if self.checkpoint:
                    self.checkpoint.save(
                        self.model,
                        self.optimizer,
                        self.scheduler,
                        epoch,
                        val_metrics,
                        tokenizer
                    )
                
                # Early stopping
                if self.early_stopping:
                    if self.early_stopping(val_metrics["val_loss"], self.model):
                        logger.info("Early stopping triggered")
                        break
            else:
                # Checkpoint without validation
                if self.checkpoint:
                    self.checkpoint.save(
                        self.model,
                        self.optimizer,
                        self.scheduler,
                        epoch,
                        {"train_loss": train_loss},
                        tokenizer
                    )
            
            logger.info(
                f"Epoch {epoch + 1}/{num_epochs} - "
                f"Train Loss: {train_loss:.4f}" +
                (f", Val Loss: {val_metrics['val_loss']:.4f}, Val Acc: {val_metrics['val_accuracy']:.4f}"
                 if val_loader else "")
            )
        
        return history
    
    def _move_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move batch to device"""
        return {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }















