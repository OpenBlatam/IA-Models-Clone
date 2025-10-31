"""
PyTorch trainer implementation for the ads training system.

This module consolidates all PyTorch training functionality into a unified,
clean architecture following the base trainer interface.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
from torch.nn.utils import clip_grad_norm_
import numpy as np
import logging
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass

from .base_trainer import BaseTrainer, TrainingConfig, TrainingMetrics, TrainingResult

logger = logging.getLogger(__name__)

@dataclass
class PyTorchModelConfig:
    """Configuration for PyTorch models."""
    input_size: int = 10
    hidden_size: int = 50
    output_size: int = 1
    num_layers: int = 3
    dropout_rate: float = 0.1
    activation: str = "relu"  # relu, tanh, sigmoid, gelu

@dataclass
class PyTorchDataConfig:
    """Configuration for PyTorch data handling."""
    num_samples: int = 1000
    train_split: float = 0.8
    val_split: float = 0.2
    shuffle: bool = True
    persistent_workers: bool = True
    drop_last: bool = True

class SimpleModel(nn.Module):
    """Simple neural network model for demonstration."""
    
    def __init__(self, config: PyTorchModelConfig):
        super().__init__()
        self.config = config
        
        # Build layers dynamically
        layers = []
        input_dim = config.input_size
        
        for i in range(config.num_layers - 1):
            layers.extend([
                nn.Linear(input_dim, config.hidden_size),
                nn.ReLU() if config.activation == "relu" else
                nn.Tanh() if config.activation == "tanh" else
                nn.Sigmoid() if config.activation == "sigmoid" else
                nn.GELU(),
                nn.Dropout(config.dropout_rate)
            ])
            input_dim = config.hidden_size
        
        # Output layer
        layers.append(nn.Linear(input_dim, config.output_size))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        return self.network(x)

class PyTorchTrainer(BaseTrainer):
    """
    PyTorch-specific trainer implementation.
    
    This trainer consolidates all PyTorch training functionality including:
    - Model management
    - Data handling
    - Training loops with mixed precision
    - Checkpointing
    - Performance optimization
    """
    
    def __init__(self, config: TrainingConfig, 
                 model_config: Optional[PyTorchModelConfig] = None,
                 data_config: Optional[PyTorchDataConfig] = None):
        """Initialize the PyTorch trainer."""
        super().__init__(config)
        
        self.model_config = model_config or PyTorchModelConfig()
        self.data_config = data_config or PyTorchDataConfig()
        
        # PyTorch-specific components
        self.model: Optional[nn.Module] = None
        self.optimizer: Optional[optim.Optimizer] = None
        self.scheduler: Optional[optim.lr_scheduler._LRScheduler] = None
        self.criterion: Optional[nn.Module] = None
        self.scaler: Optional[GradScaler] = None
        
        # Data components
        self.train_loader: Optional[DataLoader] = None
        self.val_loader: Optional[DataLoader] = None
        
        # Device management
        self.device = self._setup_device()
        
        logger.info(f"PyTorch trainer initialized on device: {self.device}")
    
    def _setup_device(self) -> torch.device:
        """Set up the training device."""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
                logger.info("CUDA device detected and selected")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = torch.device("mps")
                logger.info("MPS device detected and selected")
            else:
                device = torch.device("cpu")
                logger.info("CPU device selected")
        else:
            device = torch.device(self.config.device)
        
        return device
    
    async def setup_training(self) -> bool:
        """Set up the training environment and resources."""
        try:
            # Create model
            self.model = SimpleModel(self.model_config).to(self.device)
            
            # Create criterion
            if self.config.model_name == "classification":
                self.criterion = nn.CrossEntropyLoss()
            else:
                self.criterion = nn.MSELoss()
            
            # Create optimizer
            self.optimizer = self._create_optimizer()
            
            # Create scheduler
            self.scheduler = self._create_scheduler()
            
            # Setup mixed precision
            if self.config.mixed_precision and self.device.type == "cuda":
                self.scaler = GradScaler()
            
            # Create data loaders
            await self._setup_data()
            
            logger.info("PyTorch training setup completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup PyTorch training: {e}")
            return False
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create the optimizer based on configuration."""
        if self.config.optimizer_name.lower() == "adam":
            return optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer_name.lower() == "sgd":
            return optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                momentum=0.9
            )
        elif self.config.optimizer_name.lower() == "adamw":
            return optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        else:
            logger.warning(f"Unknown optimizer: {self.config.optimizer_name}, using Adam")
            return optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
    
    def _create_scheduler(self) -> optim.lr_scheduler._LRScheduler:
        """Create the learning rate scheduler."""
        if self.config.scheduler_name.lower() == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.num_epochs
            )
        elif self.config.scheduler_name.lower() == "step":
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1
            )
        elif self.config.scheduler_name.lower() == "exponential":
            return optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma=0.95
            )
        else:
            logger.warning(f"Unknown scheduler: {self.config.scheduler_name}, using cosine")
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.num_epochs
            )
    
    async def _setup_data(self):
        """Set up training and validation data loaders."""
        # Generate synthetic data for demonstration
        # In production, this would load from actual datasets
        X = torch.randn(self.data_config.num_samples, self.model_config.input_size)
        y = torch.randn(self.data_config.num_samples, self.model_config.output_size)
        
        # Split data
        train_size = int(self.data_config.train_split * len(X))
        val_size = len(X) - train_size
        
        X_train, X_val = torch.split(X, [train_size, val_size])
        y_train, y_val = torch.split(y, [train_size, val_size])
        
        # Create datasets
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=self.data_config.shuffle,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=self.data_config.persistent_workers,
            drop_last=self.data_config.drop_last
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=self.data_config.persistent_workers,
            drop_last=False
        )
        
        logger.info(f"Data loaders created - Train: {len(self.train_loader)}, Val: {len(self.val_loader)}")
    
    async def train_epoch(self, epoch: int) -> TrainingMetrics:
        """Train for one epoch."""
        if not self.model or not self.optimizer or not self.criterion:
            raise RuntimeError("Training not properly initialized")
        
        self.model.train()
        total_loss = 0.0
        total_steps = 0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            if self.scaler and self.config.mixed_precision:
                with autocast():
                    output = self.model(data)
                    loss = self.criterion(output, target)
                
                # Backward pass with scaling
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.config.gradient_clipping:
                    self.scaler.unscale_(self.optimizer)
                    clip_grad_norm_(self.model.parameters(), self.config.gradient_clipping)
                
                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                output = self.model(data)
                loss = self.criterion(output, target)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                if self.config.gradient_clipping:
                    clip_grad_norm_(self.model.parameters(), self.config.gradient_clipping)
                
                # Optimizer step
                self.optimizer.step()
            
            total_loss += loss.item()
            total_steps += 1
        
        # Update learning rate
        if self.scheduler:
            self.scheduler.step()
        
        # Calculate average loss
        avg_loss = total_loss / total_steps if total_steps > 0 else 0.0
        
        # Create metrics
        metrics = TrainingMetrics(
            epoch=epoch,
            step=total_steps,
            loss=avg_loss,
            learning_rate=self.optimizer.param_groups[0]['lr']
        )
        
        return metrics
    
    async def validate(self, epoch: int) -> TrainingMetrics:
        """Validate the model."""
        if not self.model or not self.criterion:
            raise RuntimeError("Training not properly initialized")
        
        self.model.eval()
        total_loss = 0.0
        total_steps = 0
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                total_steps += 1
        
        # Calculate average validation loss
        avg_val_loss = total_loss / total_steps if total_steps > 0 else 0.0
        
        # Create validation metrics
        metrics = TrainingMetrics(
            epoch=epoch,
            validation_loss=avg_val_loss
        )
        
        return metrics
    
    async def save_checkpoint(self, epoch: int, metrics: TrainingMetrics) -> str:
        """Save a training checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict() if self.model else None,
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'metrics': metrics.to_dict(),
            'config': self.config.to_dict(),
            'model_config': self.model_config.__dict__,
            'data_config': self.data_config.__dict__
        }
        
        checkpoint_path = f"{self.config.checkpoint_path}/checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        logger.info(f"Checkpoint saved: {checkpoint_path}")
        return checkpoint_path
    
    async def load_checkpoint(self, checkpoint_path: str) -> bool:
        """Load a training checkpoint."""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Load model state
            if self.model and checkpoint['model_state_dict']:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # Load optimizer state
            if self.optimizer and checkpoint['optimizer_state_dict']:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Load scheduler state
            if self.scheduler and checkpoint['scheduler_state_dict']:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            # Load scaler state
            if self.scaler and checkpoint['scaler_state_dict']:
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
            logger.info(f"Checkpoint loaded successfully: {checkpoint_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return False
    
    async def _get_final_model_path(self) -> Optional[str]:
        """Get the path to the final trained model."""
        if not self.model:
            return None
        
        model_path = f"{self.config.model_save_path}/{self.config.model_name}_final.pt"
        torch.save(self.model.state_dict(), model_path)
        return model_path
    
    async def _get_final_checkpoint_path(self) -> Optional[str]:
        """Get the path to the final checkpoint."""
        if not self.training_history:
            return None
        
        final_epoch = len(self.training_history) - 1
        return f"{self.config.checkpoint_path}/checkpoint_epoch_{final_epoch}.pt"
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        if not self.model:
            return {"error": "No model initialized"}
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            "model_type": self.model.__class__.__name__,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "device": str(self.device),
            "mixed_precision": self.scaler is not None,
            "optimizer": self.optimizer.__class__.__name__ if self.optimizer else None,
            "scheduler": self.scheduler.__class__.__name__ if self.scheduler else None
        }
