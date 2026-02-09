"""
Base Model - Abstract base class for all model architectures
============================================================

Provides common functionality for all models including:
- Weight initialization
- Forward pass structure
- Model checkpointing
- Device management
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class BaseModel(nn.Module, ABC):
    """
    Abstract base class for all model architectures.
    
    All custom models should inherit from this class to ensure
    consistent behavior and best practices.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize base model.
        
        Args:
            config: Model configuration dictionary
        """
        super().__init__()
        self.config = config or {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._initialize_weights()
    
    @abstractmethod
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        Forward pass - must be implemented by subclasses.
        
        Returns:
            Model output tensor
        """
        raise NotImplementedError("Subclasses must implement forward method")
    
    def _initialize_weights(self) -> None:
        """
        Initialize model weights using best practices.
        
        Applies proper initialization based on layer types.
        """
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, (nn.LSTM, nn.GRU)):
                for name, param in module.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        nn.init.constant_(param.data, 0)
    
    def to_device(self, device: Optional[torch.device] = None) -> 'BaseModel':
        """
        Move model to specified device.
        
        Args:
            device: Target device (defaults to CUDA if available)
            
        Returns:
            Self for method chaining
        """
        if device is None:
            device = self.device
        self.device = device
        return self.to(device)
    
    def save_checkpoint(
        self,
        filepath: Path,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        epoch: Optional[int] = None,
        metrics: Optional[Dict[str, float]] = None
    ) -> None:
        """
        Save model checkpoint.
        
        Args:
            filepath: Path to save checkpoint
            optimizer: Optimizer state (optional)
            scheduler: Learning rate scheduler state (optional)
            epoch: Current epoch (optional)
            metrics: Training metrics (optional)
        """
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'config': self.config,
            'epoch': epoch,
            'metrics': metrics or {},
        }
        
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        filepath.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, filepath)
        logger.info(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(
        self,
        filepath: Path,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        strict: bool = True
    ) -> Dict[str, Any]:
        """
        Load model checkpoint.
        
        Args:
            filepath: Path to checkpoint file
            optimizer: Optimizer to load state into (optional)
            scheduler: Scheduler to load state into (optional)
            strict: Whether to strictly enforce state dict keys match
            
        Returns:
            Dictionary containing loaded information (epoch, metrics, etc.)
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.load_state_dict(checkpoint['model_state_dict'], strict=strict)
        self.config = checkpoint.get('config', {})
        
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        logger.info(f"Checkpoint loaded from {filepath}")
        return {
            'epoch': checkpoint.get('epoch'),
            'metrics': checkpoint.get('metrics', {}),
        }
    
    def get_num_parameters(self, trainable_only: bool = True) -> int:
        """
        Get number of model parameters.
        
        Args:
            trainable_only: Count only trainable parameters
            
        Returns:
            Number of parameters
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
    
    def freeze(self) -> None:
        """Freeze all model parameters."""
        for param in self.parameters():
            param.requires_grad = False
        logger.info("Model parameters frozen")
    
    def unfreeze(self) -> None:
        """Unfreeze all model parameters."""
        for param in self.parameters():
            param.requires_grad = True
        logger.info("Model parameters unfrozen")



