"""
Tensor Validation Utilities

Validates tensors for NaN, Inf, and other issues.
"""

import logging
from typing import Optional
import torch
import numpy as np

logger = logging.getLogger(__name__)


def check_for_nan_inf(
    tensor: torch.Tensor,
    name: str = "tensor",
    raise_on_error: bool = False
) -> bool:
    """
    Check for NaN or Inf values in tensor.
    
    Args:
        tensor: Tensor to check
        name: Name for logging
        raise_on_error: Whether to raise exception on NaN/Inf
        
    Returns:
        True if NaN/Inf found, False otherwise
        
    Raises:
        ValueError: If raise_on_error is True and NaN/Inf detected
    """
    has_nan = torch.isnan(tensor).any()
    has_inf = torch.isinf(tensor).any()
    
    if has_nan:
        logger.warning(f"NaN detected in {name}")
        if raise_on_error:
            raise ValueError(f"NaN detected in {name}")
        return True
    
    if has_inf:
        logger.warning(f"Inf detected in {name}")
        if raise_on_error:
            raise ValueError(f"Inf detected in {name}")
        return True
    
    return False


def validate_tensor(
    tensor: torch.Tensor,
    name: str = "tensor",
    check_nan: bool = True,
    check_inf: bool = True,
    check_shape: Optional[tuple] = None,
    check_range: Optional[tuple] = None
) -> bool:
    """
    Comprehensive tensor validation.
    
    Args:
        tensor: Tensor to validate
        name: Name for logging
        check_nan: Check for NaN
        check_inf: Check for Inf
        check_shape: Optional expected shape
        check_range: Optional (min, max) range
        
    Returns:
        True if valid, False otherwise
    """
    # Check NaN/Inf
    if check_nan or check_inf:
        if check_for_nan_inf(tensor, name):
            return False
    
    # Check shape
    if check_shape is not None:
        if tensor.shape != check_shape:
            logger.warning(
                f"Shape mismatch in {name}: "
                f"expected {check_shape}, got {tensor.shape}"
            )
            return False
    
    # Check range
    if check_range is not None:
        min_val, max_val = check_range
        actual_min = tensor.min().item()
        actual_max = tensor.max().item()
        
        if actual_min < min_val or actual_max > max_val:
            logger.warning(
                f"Value out of range in {name}: "
                f"expected [{min_val}, {max_val}], got [{actual_min}, {actual_max}]"
            )
            return False
    
    return True


def validate_audio(
    audio: np.ndarray,
    name: str = "audio",
    sample_rate: Optional[int] = None,
    max_duration: Optional[float] = None
) -> bool:
    """
    Validate audio array.
    
    Args:
        audio: Audio array
        name: Name for logging
        sample_rate: Optional sample rate
        max_duration: Optional maximum duration in seconds
        
    Returns:
        True if valid, False otherwise
    """
    # Check if empty
    if audio is None or len(audio) == 0:
        logger.warning(f"Empty audio array: {name}")
        return False
    
    # Check for NaN/Inf
    if np.isnan(audio).any():
        logger.warning(f"NaN detected in audio: {name}")
        return False
    
    if np.isinf(audio).any():
        logger.warning(f"Inf detected in audio: {name}")
        return False
    
    # Check duration
    if sample_rate and max_duration:
        duration = len(audio) / sample_rate
        if duration > max_duration:
            logger.warning(
                f"Audio duration too long in {name}: "
                f"{duration:.2f}s > {max_duration}s"
            )
            return False
    
    return True



