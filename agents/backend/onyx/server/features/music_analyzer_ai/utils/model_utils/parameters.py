"""
Model Parameters Module

Parameter counting and analysis utilities.
"""

from typing import Dict
import logging

logger = logging.getLogger(__name__)

try:
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def count_parameters(model: nn.Module, trainable_only: bool = False) -> Dict[str, int]:
    """
    Count model parameters.
    
    Args:
        model: Model to count parameters for
        trainable_only: Only count trainable parameters
    
    Returns:
        Dictionary with parameter counts
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required")
    
    total_params = 0
    trainable_params = 0
    
    for param in model.parameters():
        num_params = param.numel()
        total_params += num_params
        if param.requires_grad:
            trainable_params += num_params
    
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params if trainable_only else total_params,
        "non_trainable_parameters": total_params - trainable_params
    }


def get_model_size_mb(model: nn.Module) -> float:
    """
    Get model size in MB.
    
    Args:
        model: Model to get size for
    
    Returns:
        Model size in MB
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required")
    
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb



