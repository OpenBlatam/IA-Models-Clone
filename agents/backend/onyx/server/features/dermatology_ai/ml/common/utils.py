"""
Common Utilities
Shared utility functions across ML components
"""

import torch
import numpy as np
from typing import Union, Tuple, Optional, Dict, Any
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def ensure_tensor(
    data: Union[torch.Tensor, np.ndarray, list],
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Ensure data is a torch tensor
    
    Args:
        data: Input data
        dtype: Desired dtype
        device: Desired device
        
    Returns:
        torch.Tensor
    """
    if isinstance(data, torch.Tensor):
        tensor = data
    elif isinstance(data, np.ndarray):
        tensor = torch.from_numpy(data)
    elif isinstance(data, list):
        tensor = torch.tensor(data)
    else:
        raise TypeError(f"Cannot convert {type(data)} to tensor")
    
    if dtype:
        tensor = tensor.to(dtype)
    if device:
        tensor = tensor.to(device)
    
    return tensor


def ensure_numpy(
    data: Union[torch.Tensor, np.ndarray, list]
) -> np.ndarray:
    """Ensure data is numpy array"""
    if isinstance(data, torch.Tensor):
        return data.cpu().numpy()
    elif isinstance(data, np.ndarray):
        return data
    elif isinstance(data, list):
        return np.array(data)
    else:
        raise TypeError(f"Cannot convert {type(data)} to numpy array")


def get_device(device: Optional[str] = None) -> torch.device:
    """Get torch device"""
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """Count model parameters"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total,
        'trainable': trainable,
        'non_trainable': total - trainable
    }


def get_model_size(model: torch.nn.Module) -> float:
    """Get model size in MB"""
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / (1024 ** 2)


def set_seed(seed: int = 42):
    """Set random seed for reproducibility"""
    import random
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    logger.info(f"Random seed set to {seed}")


def format_time(seconds: float) -> str:
    """Format time in human-readable format"""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"


def format_size(size_bytes: float) -> str:
    """Format size in human-readable format"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f}{unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f}PB"


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safe division with default value"""
    if denominator == 0:
        return default
    return numerator / denominator


def clip_values(
    values: Union[torch.Tensor, np.ndarray, list],
    min_val: Optional[float] = None,
    max_val: Optional[float] = None
) -> Union[torch.Tensor, np.ndarray]:
    """Clip values to range"""
    if isinstance(values, torch.Tensor):
        return torch.clamp(values, min=min_val, max=max_val)
    elif isinstance(values, np.ndarray):
        return np.clip(values, a_min=min_val, a_max=max_val)
    else:
        arr = np.array(values)
        return np.clip(arr, a_min=min_val, a_max=max_val)


def normalize_tensor(
    tensor: torch.Tensor,
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    std: Tuple[float, ...] = (0.229, 0.224, 0.225)
) -> torch.Tensor:
    """Normalize tensor with ImageNet stats"""
    mean_tensor = torch.tensor(mean).view(-1, 1, 1).to(tensor.device)
    std_tensor = torch.tensor(std).view(-1, 1, 1).to(tensor.device)
    return (tensor - mean_tensor) / std_tensor


def denormalize_tensor(
    tensor: torch.Tensor,
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    std: Tuple[float, ...] = (0.229, 0.224, 0.225)
) -> torch.Tensor:
    """Denormalize tensor"""
    mean_tensor = torch.tensor(mean).view(-1, 1, 1).to(tensor.device)
    std_tensor = torch.tensor(std).view(-1, 1, 1).to(tensor.device)
    return tensor * std_tensor + mean_tensor













