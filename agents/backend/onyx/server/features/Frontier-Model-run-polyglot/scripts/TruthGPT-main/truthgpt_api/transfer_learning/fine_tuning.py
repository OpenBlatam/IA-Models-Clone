"""
Fine-tuning for TruthGPT API
============================

TensorFlow-like fine-tuning implementation.
"""

import torch
import torch.nn as nn
from typing import Any, Optional, Dict, List, Tuple
import numpy as np


class FineTuner:
    """
    Fine-tune pretrained models.
    
    Similar to tf.keras.applications, this class
    provides fine-tuning capabilities for pretrained models.
    """
    
    def __init__(self, 
                 model: Any,
                 freeze_base: bool = True,
                 custom_head: Optional[nn.Module] = None,
                 name: Optional[str] = None):
        """
        Initialize FineTuner.
        
        Args:
            model: Pretrained model to fine-tune
            freeze_base: Whether to freeze base layers
            custom_head: Custom classification head
            name: Optional name for the fine-tuner
        """
        self.model = model
        self.freeze_base = freeze_base
        self.custom_head = custom_head
        self.name = name or "fine_tuner"
        
        # Freeze base layers if specified
        if freeze_base:
            self._freeze_base_layers()
        
        # Add custom head if provided
        if custom_head is not None:
            self._add_custom_head(custom_head)
    
    def _freeze_base_layers(self):
        """Freeze base layers of the model."""
        print(f"🔒 Freezing base layers...")
        
        for name, param in self.model.named_parameters():
            if 'fc' not in name and 'classifier' not in name:
                param.requires_grad = False
        
        print(f"✅ Base layers frozen!")
    
    def _add_custom_head(self, custom_head: nn.Module):
        """Add custom classification head."""
        # Remove existing head
        if hasattr(self.model, 'fc'):
            self.model.fc = custom_head
        elif hasattr(self.model, 'classifier'):
            self.model.classifier = custom_head
        else:
            raise ValueError("Could not find classification head to replace")
        
        print(f"✅ Custom head added!")
    
    def freeze_layers(self, layers: List[str]):
        """
        Freeze specific layers.
        
        Args:
            layers: List of layer names to freeze
        """
        for name, param in self.model.named_parameters():
            if any(layer in name for layer in layers):
                param.requires_grad = False
    
    def unfreeze_layers(self, layers: List[str]):
        """
        Unfreeze specific layers.
        
        Args:
            layers: List of layer names to unfreeze
        """
        for name, param in self.model.named_parameters():
            if any(layer in name for layer in layers):
                param.requires_grad = True
    
    def make_trainable(self):
        """Make all layers trainable."""
        for param in self.model.parameters():
            param.requires_grad = True
    
    def get_trainable_params(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def get_total_params(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.model.parameters())
    
    def get_trainable_summary(self) -> Dict[str, Any]:
        """Get summary of trainable parameters."""
        trainable_params = self.get_trainable_params()
        total_params = self.get_total_params()
        
        summary = {
            'trainable_params': trainable_params,
            'total_params': total_params,
            'trainable_percentage': (trainable_params / total_params) * 100 if total_params > 0 else 0
        }
        
        return summary
    
    def __repr__(self):
        return f"FineTuner(freeze_base={self.freeze_base}, trainable_params={self.get_trainable_params()})"







