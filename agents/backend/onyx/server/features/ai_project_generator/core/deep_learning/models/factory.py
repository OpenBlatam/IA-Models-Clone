"""
Model Factory - Factory for Creating Model Instances
=====================================================

Provides factory functions for creating different model architectures.
"""

import logging
from typing import Dict, Any, Optional, Type
import torch.nn as nn

from .base_model import BaseModel
from .transformer_model import TransformerModel

logger = logging.getLogger(__name__)


def create_model(
    model_type: str,
    config: Dict[str, Any],
    **kwargs
) -> BaseModel:
    """
    Create a model instance based on type and configuration.
    
    Args:
        model_type: Type of model ('transformer', 'cnn', 'rnn', etc.)
        config: Model configuration dictionary
        **kwargs: Additional model-specific parameters
        
    Returns:
        Model instance
        
    Raises:
        ValueError: If model type is not supported
    """
    model_type = model_type.lower()
    config = {**config, **kwargs}
    
    if model_type == 'transformer':
        return TransformerModel(**config)
    elif model_type == 'cnn':
        # Import here to avoid circular dependencies
        try:
            from .cnn_model import CNNModel
            return CNNModel(**config)
        except ImportError:
            logger.warning("CNNModel not available, using base model")
            return BaseModel(config)
    elif model_type == 'rnn':
        try:
            from .rnn_model import RNNModel
            return RNNModel(**config)
        except ImportError:
            logger.warning("RNNModel not available, using base model")
            return BaseModel(config)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def initialize_weights(model: nn.Module, method: str = 'kaiming') -> None:
    """
    Initialize model weights using specified method.
    
    Args:
        model: PyTorch model
        method: Initialization method ('kaiming', 'xavier', 'normal', 'zeros')
    """
    method = method.lower()
    
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            if method == 'kaiming':
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif method == 'xavier':
                nn.init.xavier_normal_(module.weight)
            elif method == 'normal':
                nn.init.normal_(module.weight, 0, 0.02)
            elif method == 'zeros':
                nn.init.zeros_(module.weight)
            
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        
        elif isinstance(module, nn.Linear):
            if method == 'kaiming':
                nn.init.kaiming_normal_(module.weight)
            elif method == 'xavier':
                nn.init.xavier_normal_(module.weight)
            elif method == 'normal':
                nn.init.normal_(module.weight, 0, 0.01)
            elif method == 'zeros':
                nn.init.zeros_(module.weight)
            
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        
        elif isinstance(module, (nn.BatchNorm2d, nn.LayerNorm)):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)



