"""
Tensor Validator Module

Validates and sanitizes PyTorch tensors.
"""

from typing import Optional
import logging

logger = logging.getLogger(__name__)

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available")


class TensorValidator:
    """Validate and sanitize tensors."""
    
    @staticmethod
    def check_nan_inf(tensor: torch.Tensor, name: str = "tensor") -> bool:
        """
        Check for NaN/Inf values in tensor.
        
        Args:
            tensor: Input tensor.
            name: Name for logging.
        
        Returns:
            True if NaN/Inf detected, False otherwise.
        """
        if not TORCH_AVAILABLE:
            return False
        
        has_nan = torch.isnan(tensor).any()
        has_inf = torch.isinf(tensor).any()
        
        if has_nan:
            logger.warning(f"NaN detected in {name}")
        if has_inf:
            logger.warning(f"Inf detected in {name}")
        
        return has_nan or has_inf
    
    @staticmethod
    def fix_nan_inf(
        tensor: torch.Tensor,
        nan_value: float = 0.0,
        posinf_value: float = 1.0,
        neginf_value: float = -1.0
    ) -> torch.Tensor:
        """
        Fix NaN/Inf values in tensor.
        
        Args:
            tensor: Input tensor.
            nan_value: Value to replace NaN.
            posinf_value: Value to replace positive infinity.
            neginf_value: Value to replace negative infinity.
        
        Returns:
            Fixed tensor.
        """
        if not TORCH_AVAILABLE:
            return tensor
        
        return torch.nan_to_num(
            tensor,
            nan=nan_value,
            posinf=posinf_value,
            neginf=neginf_value
        )
    
    @staticmethod
    def validate_shape(
        tensor: torch.Tensor,
        expected_shape: tuple,
        name: str = "tensor"
    ) -> bool:
        """
        Validate tensor shape.
        
        Args:
            tensor: Input tensor.
            expected_shape: Expected shape tuple.
            name: Name for logging.
        
        Returns:
            True if shape matches, False otherwise.
        """
        if not TORCH_AVAILABLE:
            return False
        
        if tensor.shape != expected_shape:
            logger.warning(
                f"Shape mismatch in {name}: "
                f"expected {expected_shape}, got {tensor.shape}"
            )
            return False
        return True
    
    @staticmethod
    def validate_range(
        tensor: torch.Tensor,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
        name: str = "tensor"
    ) -> bool:
        """
        Validate tensor value range.
        
        Args:
            tensor: Input tensor.
            min_val: Minimum allowed value.
            max_val: Maximum allowed value.
            name: Name for logging.
        
        Returns:
            True if values in range, False otherwise.
        """
        if not TORCH_AVAILABLE:
            return False
        
        if min_val is not None and (tensor < min_val).any():
            logger.warning(f"Values below {min_val} in {name}")
            return False
        
        if max_val is not None and (tensor > max_val).any():
            logger.warning(f"Values above {max_val} in {name}")
            return False
        
        return True



