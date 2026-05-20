"""
Model Summary Module

Model summary and analysis utilities.
"""

from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

try:
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .parameters import count_parameters, get_model_size_mb


def initialize_weights(model: nn.Module, method: str = "xavier_uniform"):
    """
    Initialize model weights.
    
    Args:
        model: Model to initialize
        method: Initialization method
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required")
    
    for module in model.modules():
        if isinstance(module, nn.Linear):
            if method == "xavier_uniform":
                nn.init.xavier_uniform_(module.weight)
            elif method == "xavier_normal":
                nn.init.xavier_normal_(module.weight)
            elif method == "kaiming_uniform":
                nn.init.kaiming_uniform_(module.weight)
            elif method == "kaiming_normal":
                nn.init.kaiming_normal_(module.weight)
            else:
                raise ValueError(f"Unknown initialization method: {method}")
            
            if module.bias is not None:
                nn.init.zeros_(module.bias)


def get_model_summary(model: nn.Module) -> Dict[str, Any]:
    """
    Get model summary.
    
    Args:
        model: Model to summarize
    
    Returns:
        Dictionary with model summary
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required")
    
    param_info = count_parameters(model)
    model_size = get_model_size_mb(model)
    
    # Count layers
    num_layers = sum(1 for _ in model.modules() if isinstance(_, (nn.Linear, nn.Conv1d, nn.Conv2d)))
    
    return {
        "total_parameters": param_info["total_parameters"],
        "trainable_parameters": param_info["trainable_parameters"],
        "model_size_mb": model_size,
        "num_layers": num_layers,
        "architecture": str(type(model).__name__)
    }



