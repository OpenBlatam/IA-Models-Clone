"""
Validation module for KV Cache.

Provides validation functions following best practices.
"""
import logging
from typing import Tuple
import torch

logger = logging.getLogger(__name__)


class CacheValidator:
    """
    Validates inputs and operations for KV cache.
    
    Responsibilities:
    - Validate tensor shapes and dtypes
    - Validate configuration
    - Validate cache operations
    - Provide helpful error messages
    """
    
    @staticmethod
    def validate_tensors(
        key: torch.Tensor,
        value: torch.Tensor,
        expected_dims: Optional[int] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate key and value tensors.
        
        Args:
            key: Key tensor
            value: Value tensor
            expected_dims: Expected number of dimensions (optional)
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check shapes match
        if key.shape != value.shape:
            return False, (
                f"Key and value shapes must match. "
                f"Got key={key.shape}, value={value.shape}"
            )
        
        # Check expected dimensions
        if expected_dims is not None:
            if len(key.shape) != expected_dims:
                return False, (
                    f"Unexpected number of dimensions. "
                    f"Got {len(key.shape)}, expected {expected_dims}"
                )
        
        # Check tensors are valid (not empty, finite values)
        if key.numel() == 0:
            return False, "Key tensor is empty"
        
        if value.numel() == 0:
            return False, "Value tensor is empty"
        
        # Check for NaN/Inf
        if not torch.isfinite(key).all():
            return False, "Key tensor contains NaN or Inf values"
        
        if not torch.isfinite(value).all():
            return False, "Value tensor contains NaN or Inf values"
        
        return True, None
    
    @staticmethod
    def validate_position(position: int) -> Tuple[bool, Optional[str]]:
        """
        Validate cache position.
        
        Args:
            position: Cache position
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not isinstance(position, int):
            return False, f"Position must be integer, got {type(position)}"
        
        if position < 0:
            return False, f"Position must be non-negative, got {position}"
        
        return True, None
    
    @staticmethod
    def validate_config(config) -> Tuple[bool, Optional[str]]:
        """
        Validate cache configuration.
        
        Args:
            config: KVCacheConfig instance
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            config.validate()
            return True, None
        except ValueError as e:
            return False, str(e)
        except Exception as e:
            return False, f"Configuration validation error: {e}"
    
    @staticmethod
    def validate_device_transfer(
        tensor: torch.Tensor,
        target_device: torch.device
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate device transfer is possible.
        
        Args:
            tensor: Tensor to transfer
            target_device: Target device
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if tensor.device == target_device:
            return True, None
        
        if target_device.type == "cuda":
            if not torch.cuda.is_available():
                return False, "CUDA device requested but CUDA not available"
            
            if target_device.index is not None:
                if target_device.index >= torch.cuda.device_count():
                    return False, (
                        f"CUDA device index {target_device.index} "
                        f"exceeds available devices ({torch.cuda.device_count()})"
                    )
        
        elif target_device.type == "mps":
            if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
                return False, "MPS device requested but MPS not available"
        
        return True, None

