"""
Base Model Classes
==================
Base classes for all models following best practices
"""

from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import structlog

logger = structlog.get_logger()


class BaseModel(nn.Module):
    """
    Base class for all models
    Follows PyTorch best practices
    """
    
    def __init__(self, device: Optional[torch.device] = None):
        """
        Initialize base model
        
        Args:
            device: Device to use (auto-detect if None)
        """
        super().__init__()
        self.device = device or self._get_device()
        logger.debug("BaseModel initialized", device=str(self.device))
    
    def _get_device(self) -> torch.device:
        """
        Get appropriate device
        
        Returns:
            torch.device
        """
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    
    def to_device(self, device: Optional[torch.device] = None) -> 'BaseModel':
        """
        Move model to device
        
        Args:
            device: Target device
            
        Returns:
            Self for chaining
        """
        target_device = device or self.device
        self.device = target_device
        return self.to(target_device)
    
    def get_num_parameters(self, trainable_only: bool = False) -> int:
        """
        Get number of parameters
        
        Args:
            trainable_only: Count only trainable parameters
            
        Returns:
            Number of parameters
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
    
    def get_model_size_mb(self) -> float:
        """
        Get model size in MB
        
        Returns:
            Model size in MB
        """
        param_size = sum(p.numel() * p.element_size() for p in self.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.buffers())
        return (param_size + buffer_size) / (1024 ** 2)
    
    def freeze(self) -> None:
        """Freeze all parameters"""
        for param in self.parameters():
            param.requires_grad = False
        logger.info("Model frozen")
    
    def unfreeze(self) -> None:
        """Unfreeze all parameters"""
        for param in self.parameters():
            param.requires_grad = True
        logger.info("Model unfrozen")


class BaseClassifier(BaseModel):
    """
    Base class for classification models
    """
    
    def __init__(
        self,
        num_classes: int,
        device: Optional[torch.device] = None
    ):
        """
        Initialize base classifier
        
        Args:
            num_classes: Number of classes
            device: Device to use
        """
        super().__init__(device)
        self.num_classes = num_classes
    
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        Forward pass (to be implemented by subclasses)
        
        Returns:
            Logits tensor
        """
        raise NotImplementedError("Subclasses must implement forward method")
    
    def predict(self, *args, **kwargs) -> torch.Tensor:
        """
        Make predictions
        
        Returns:
            Predictions tensor
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(*args, **kwargs)
            return torch.argmax(logits, dim=-1)
    
    def predict_proba(self, *args, **kwargs) -> torch.Tensor:
        """
        Get prediction probabilities
        
        Returns:
            Probabilities tensor
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(*args, **kwargs)
            return torch.softmax(logits, dim=-1)




