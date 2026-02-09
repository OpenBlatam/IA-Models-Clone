"""
Models Module for HeyGen AI
===========================

This module contains PyTorch model architectures following best practices:
- Custom nn.Module classes for model architectures
- Proper weight initialization
- Type hints and documentation
- Modular design
"""

from typing import Optional

import torch
import torch.nn as nn

__all__ = ["BaseModel"]


class BaseModel(nn.Module):
    """Base class for all model architectures.
    
    Provides common functionality:
    - Weight initialization
    - Device management
    - Model saving/loading
    """
    
    def __init__(self):
        """Initialize base model."""
        super().__init__()
        self._initialized = False
    
    def _init_weights(self):
        """Initialize model weights using Xavier/Glorot initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, *args, **kwargs):
        """Forward pass - to be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement forward method")
    
    def save(self, path: str):
        """Save model to file.
        
        Args:
            path: Path to save model
        """
        torch.save({
            'model_state_dict': self.state_dict(),
            'model_class': self.__class__.__name__,
        }, path)
    
    @classmethod
    def load(cls, path: str, device: Optional[torch.device] = None):
        """Load model from file.
        
        Args:
            path: Path to model file
            device: Device to load model on
        
        Returns:
            Loaded model instance
        """
        checkpoint = torch.load(path, map_location=device)
        model = cls()
        model.load_state_dict(checkpoint['model_state_dict'])
        if device:
            model = model.to(device)
        return model



