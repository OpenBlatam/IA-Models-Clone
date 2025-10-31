"""
🚀 Base Trainer Class

Abstract base class for all training operations.
Provides common interface for model training and optimization.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import logging
import json
import time
from tqdm import tqdm

logger = logging.getLogger(__name__)


class BaseTrainer(ABC):
    """
    Abstract base class for all trainers.
    
    This class provides a common interface for:
    - Model training and optimization
    - Loss calculation and backpropagation
    - Training loop management
    - Checkpointing and model saving
    - Training monitoring and logging
    """
    
    def __init__(self, model: nn.Module, config: Dict[str, Any], name: str = "base_trainer"):
        """
        Initialize the base trainer.
        
        Args:
            model: Model to train
            config: Training configuration dictionary
            name: Trainer name for identification
        """
        self.model = model
        self.config = config
        self.name = name
        self.device = next(model.parameters()).device
        
        # Training components
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        
        # Training state
        self.current_epoch = 0
        self.current_step = 0
        self.best_metric = float('inf')
        self.training_history = {
            "train_loss": [],
            "val_loss": [],
            "train_metrics": [],
            "val_metrics": [],
            "learning_rates": []
        }
        
        # Initialize training components
        self._setup_training_components()
        self._log_training_info()
    
    @abstractmethod
    def _setup_training_components(self) -> None:
        """
        Setup optimizer, scheduler, and loss function.
        Must be implemented by subclasses.
        """
        pass
    
    def _log_training_info(self) -> None:
        """Log training setup information."""
        logger.info(f"Trainer {self.name} initialized:")
        logger.info(f"  Model: {self.model.__class__.__name__}")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Optimizer: {self.optimizer.__class__.__name__ if self.optimizer else 'None'}")
        logger.info(f"  Scheduler: {self.scheduler.__class__.__name__ if self.scheduler else 'None'}")
        logger.info(f"  Criterion: {self.criterion.__class__.__name__ if self.criterion else 'None'}")
    
    def get_training_info(self) -> Dict[str, Any]:
        """Get comprehensive training information."""
        return {
            "name": self.name,
            "model": self.model.__class__.__name__,
            "device": str(self.device),
            "optimizer": self.optimizer.__class__.__name__ if self.optimizer else None,
            "scheduler": self.scheduler.__class__.__name__ if self.scheduler else None,
            "criterion": self.criterion.__class__.__name__ if self.criterion else None,
            "current_epoch": self.current_epoch,
            "current_step": self.current_step,
            "best_metric": self.best_metric,
            "config": self.config
        }
    
    def get_training_history(self) -> Dict[str, List]:
        """Get training history."""
        return self.training_history.copy()
    
    def get_current_state(self) -> Dict[str, Any]:
        """Get current training state."""
        return {
            "epoch": self.current_epoch,
            "step": self.current_step,
            "best_metric": self.best_metric,
            "learning_rate": self.optimizer.param_groups[0]['lr'] if self.optimizer else None
        }
    
    def save_checkpoint(self, path: Union[str, Path], save_optimizer: bool = True, 
                       save_scheduler: bool = True) -> None:
        """
        Save training checkpoint.
        
        Args:
            path: Path to save the checkpoint
            save_optimizer: Whether to save optimizer state
            save_scheduler: Whether to save scheduler state
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict() if save_optimizer and self.optimizer else None,
            "scheduler_state_dict": self.scheduler.state_dict() if save_scheduler and self.scheduler else None,
            "current_epoch": self.current_epoch,
            "current_step": self.current_step,
            "best_metric": self.best_metric,
            "training_history": self.training_history,
            "config": self.config
        }
        
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to: {path}")
    
    def load_checkpoint(self, path: Union[str, Path], load_optimizer: bool = True, 
                       load_scheduler: bool = True) -> None:
        """
        Load training checkpoint.
        
        Args:
            path: Path to load the checkpoint from
            load_optimizer: Whether to load optimizer state
            load_scheduler: Whether to load scheduler state
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {path}")
        
        checkpoint = torch.load(path, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint["model_state_dict"])
        
        # Load optimizer state
        if load_optimizer and checkpoint["optimizer_state_dict"] and self.optimizer:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        # Load scheduler state
        if load_scheduler and checkpoint["scheduler_state_dict"] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        # Load training state
        self.current_epoch = checkpoint.get("current_epoch", 0)
        self.current_step = checkpoint.get("current_step", 0)
        self.best_metric = checkpoint.get("best_metric", float('inf'))
        
        # Load training history
        if "training_history" in checkpoint:
            self.training_history.update(checkpoint["training_history"])
        
        logger.info(f"Checkpoint loaded from: {path}")
        logger.info(f"Resuming from epoch {self.current_epoch}, step {self.current_step}")
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Dictionary containing training metrics
        """
        self.model.train()
        epoch_loss = 0.0
        epoch_metrics = {}
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {self.current_epoch + 1}")
        
        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            # Move data to device
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            
            # Calculate loss
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping if specified
            if self.config.get("gradient_clip_norm"):
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config["gradient_clip_norm"]
                )
            
            # Optimizer step
            self.optimizer.step()
            
            # Update metrics
            epoch_loss += loss.item()
            batch_metrics = self._calculate_batch_metrics(outputs, targets)
            
            for key, value in batch_metrics.items():
                if key not in epoch_metrics:
                    epoch_metrics[key] = []
                epoch_metrics[key].append(value)
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.6f}"
            })
            
            self.current_step += 1
        
        # Calculate epoch averages
        epoch_metrics = {key: np.mean(values) for key, values in epoch_metrics.items()}
        epoch_metrics['loss'] = epoch_loss / len(train_loader)
        
        # Update training history
        self.training_history["train_loss"].append(epoch_metrics['loss'])
        self.training_history["train_metrics"].append(epoch_metrics)
        self.training_history["learning_rates"].append(self.optimizer.param_groups[0]['lr'])
        
        return epoch_metrics
    
    def validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Validate for one epoch.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Dictionary containing validation metrics
        """
        self.model.eval()
        epoch_loss = 0.0
        epoch_metrics = {}
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                # Move data to device
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                
                # Calculate loss
                loss = self.criterion(outputs, targets)
                
                # Update metrics
                epoch_loss += loss.item()
                batch_metrics = self._calculate_batch_metrics(outputs, targets)
                
                for key, value in batch_metrics.items():
                    if key not in epoch_metrics:
                        epoch_metrics[key] = []
                    epoch_metrics[key].append(value)
        
        # Calculate epoch averages
        epoch_metrics = {key: np.mean(values) for key, values in epoch_metrics.items()}
        epoch_metrics['loss'] = epoch_loss / len(val_loader)
        
        # Update training history
        self.training_history["val_loss"].append(epoch_metrics['loss'])
        self.training_history["val_metrics"].append(epoch_metrics)
        
        return epoch_metrics
    
    @abstractmethod
    def _calculate_batch_metrics(self, outputs: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """
        Calculate metrics for a single batch.
        Must be implemented by subclasses.
        
        Args:
            outputs: Model outputs
            targets: Ground truth targets
            
        Returns:
            Dictionary containing batch metrics
        """
        pass
    
    def update_learning_rate(self) -> None:
        """Update learning rate using scheduler if available."""
        if self.scheduler:
            self.scheduler.step()
    
    def is_best_model(self, current_metric: float, metric_name: str = "loss") -> bool:
        """
        Check if current model is the best based on metric.
        
        Args:
            current_metric: Current metric value
            metric_name: Name of the metric to compare
            
        Returns:
            True if current model is the best
        """
        if metric_name == "loss":
            is_best = current_metric < self.best_metric
        else:
            is_best = current_metric > self.best_metric
        
        if is_best:
            self.best_metric = current_metric
        
        return is_best
    
    def get_training_summary(self) -> str:
        """
        Generate a summary of the training.
        
        Returns:
            String summary of the training
        """
        summary = f"""
Training Summary: {self.name}
{'=' * 50}
Current Epoch: {self.current_epoch}
Current Step: {self.current_step}
Best Metric: {self.best_metric}
Device: {self.device}

Training History:
  - Train Loss: {len(self.training_history['train_loss'])} epochs
  - Val Loss: {len(self.training_history['val_loss'])} epochs
  - Learning Rates: {len(self.training_history['learning_rates'])} epochs

Latest Metrics:
  - Train Loss: {self.training_history['train_loss'][-1] if self.training_history['train_loss'] else 'N/A'}
  - Val Loss: {self.training_history['val_loss'][-1] if self.training_history['val_loss'] else 'N/A'}
  - Learning Rate: {self.training_history['learning_rates'][-1] if self.training_history['learning_rates'] else 'N/A'}
        """.strip()
        
        return summary
    
    def save_training_history(self, path: Union[str, Path]) -> None:
        """
        Save training history to JSON file.
        
        Args:
            path: Path to save the training history
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        save_dict = {
            "training_history": self.training_history,
            "training_info": self.get_training_info(),
            "current_state": self.get_current_state()
        }
        
        with open(path, 'w') as f:
            json.dump(save_dict, f, indent=2, default=str)
        
        logger.info(f"Training history saved to: {path}")
    
    def load_training_history(self, path: Union[str, Path]) -> None:
        """
        Load training history from JSON file.
        
        Args:
            path: Path to load the training history from
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Training history file not found: {path}")
        
        with open(path, 'r') as f:
            data_dict = json.load(f)
        
        # Update training history
        if "training_history" in data_dict:
            self.training_history.update(data_dict["training_history"])
        
        logger.info(f"Training history loaded from: {path}")






