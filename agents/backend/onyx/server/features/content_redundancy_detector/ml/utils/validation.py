"""
Validation Utilities
Input validation and sanitization
"""

import torch
import numpy as np
from typing import Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def validate_tensor(
    tensor: Any,
    expected_shape: Optional[Tuple[int, ...]] = None,
    expected_dtype: Optional[torch.dtype] = None,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    allow_nan: bool = False,
    allow_inf: bool = False,
) -> torch.Tensor:
    """
    Validate tensor input
    
    Args:
        tensor: Input tensor
        expected_shape: Expected shape
        expected_dtype: Expected dtype
        min_value: Minimum value
        max_value: Maximum value
        allow_nan: Allow NaN values
        allow_inf: Allow Inf values
        
    Returns:
        Validated tensor
        
    Raises:
        ValueError: If validation fails
    """
    if not isinstance(tensor, torch.Tensor):
        if isinstance(tensor, np.ndarray):
            tensor = torch.from_numpy(tensor)
        else:
            raise ValueError(f"Input must be torch.Tensor or numpy.ndarray, got {type(tensor)}")
    
    # Check shape
    if expected_shape is not None:
        if tensor.shape != expected_shape:
            raise ValueError(f"Expected shape {expected_shape}, got {tensor.shape}")
    
    # Check dtype
    if expected_dtype is not None:
        if tensor.dtype != expected_dtype:
            tensor = tensor.to(expected_dtype)
    
    # Check for NaN
    if not allow_nan and torch.isnan(tensor).any():
        raise ValueError("Tensor contains NaN values")
    
    # Check for Inf
    if not allow_inf and torch.isinf(tensor).any():
        raise ValueError("Tensor contains Inf values")
    
    # Check value range
    if min_value is not None:
        if (tensor < min_value).any():
            raise ValueError(f"Tensor contains values below {min_value}")
    
    if max_value is not None:
        if (tensor > max_value).any():
            raise ValueError(f"Tensor contains values above {max_value}")
    
    return tensor


def validate_image_tensor(
    tensor: torch.Tensor,
    image_size: Optional[int] = None,
) -> torch.Tensor:
    """
    Validate image tensor
    
    Args:
        tensor: Image tensor
        image_size: Expected image size (H=W)
        
    Returns:
        Validated tensor
    """
    # Check dimensions
    if tensor.dim() not in [3, 4]:
        raise ValueError(f"Image tensor must be 3D (C, H, W) or 4D (B, C, H, W), got {tensor.dim()}D")
    
    # Check channels
    if tensor.dim() == 3:
        if tensor.shape[0] not in [1, 3]:
            raise ValueError(f"Expected 1 or 3 channels, got {tensor.shape[0]}")
    else:
        if tensor.shape[1] not in [1, 3]:
            raise ValueError(f"Expected 1 or 3 channels, got {tensor.shape[1]}")
    
    # Check size
    if image_size is not None:
        if tensor.dim() == 3:
            h, w = tensor.shape[1], tensor.shape[2]
        else:
            h, w = tensor.shape[2], tensor.shape[3]
        
        if h != image_size or w != image_size:
            raise ValueError(f"Expected image size {image_size}x{image_size}, got {h}x{w}")
    
    # Check value range (should be [0, 1] or [0, 255])
    if tensor.dim() == 3:
        min_val, max_val = tensor.min().item(), tensor.max().item()
    else:
        min_val, max_val = tensor.min().item(), tensor.max().item()
    
    if max_val > 1.1:  # Allow small floating point errors
        # Assume [0, 255] range, normalize to [0, 1]
        tensor = tensor / 255.0
    elif min_val < 0:
        raise ValueError("Image tensor contains negative values")
    
    return tensor


def sanitize_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """
    Sanitize tensor (handle NaN/Inf)
    
    Args:
        tensor: Input tensor
        
    Returns:
        Sanitized tensor
    """
    # Replace NaN with 0
    tensor = torch.where(torch.isnan(tensor), torch.zeros_like(tensor), tensor)
    
    # Replace Inf with large finite value
    tensor = torch.where(torch.isinf(tensor), torch.full_like(tensor, 1e6), tensor)
    
    # Clip extreme values
    tensor = torch.clamp(tensor, -1e6, 1e6)
    
    return tensor



