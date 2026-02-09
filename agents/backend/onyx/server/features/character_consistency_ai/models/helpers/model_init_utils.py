"""
Model Initialization Utilities
===============================

Utilities for model weight initialization and component setup.
"""

import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


class ModelInitializer:
    """Handles model initialization and weight setup."""
    
    @staticmethod
    def initialize_weights(module: nn.Module) -> None:
        """
        Initialize model weights using Xavier uniform for Linear layers.
        
        Args:
            module: PyTorch module to initialize
        """
        for module_item in module.modules():
            if isinstance(module_item, nn.Linear):
                nn.init.xavier_uniform_(module_item.weight)
                if module_item.bias is not None:
                    nn.init.zeros_(module_item.bias)
            elif isinstance(module_item, nn.LayerNorm):
                nn.init.ones_(module_item.weight)
                nn.init.zeros_(module_item.bias)
    
    @staticmethod
    def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
        """
        Count model parameters.
        
        Args:
            model: PyTorch model
            trainable_only: If True, only count trainable parameters
            
        Returns:
            Number of parameters
        """
        if trainable_only:
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        return sum(p.numel() for p in model.parameters())
    
    @staticmethod
    def get_model_info(model: nn.Module) -> dict:
        """
        Get model information including parameter counts.
        
        Args:
            model: PyTorch model
            
        Returns:
            Dict with model information
        """
        total_params = ModelInitializer.count_parameters(model, trainable_only=False)
        trainable_params = ModelInitializer.count_parameters(model, trainable_only=True)
        
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "non_trainable_parameters": total_params - trainable_params,
        }
    
    @staticmethod
    def move_to_device(module: nn.Module, device: torch.device) -> nn.Module:
        """
        Move module to device.
        
        Args:
            module: PyTorch module
            device: Target device
            
        Returns:
            Module moved to device
        """
        return module.to(device)


