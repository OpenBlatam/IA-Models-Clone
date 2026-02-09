"""
Device Utilities
================

Utilities for device and dtype configuration in PyTorch models.
"""

import torch
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class DeviceManager:
    """Manages device and dtype configuration for PyTorch models."""
    
    @staticmethod
    def setup_device(device: Optional[str] = None) -> torch.device:
        """
        Setup computation device.
        
        Args:
            device: Device string (cuda/cpu/auto) or None for auto-detection
            
        Returns:
            Configured torch.device
        """
        if device is None or device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        device_obj = torch.device(device)
        logger.debug(f"Device configured: {device_obj}")
        return device_obj
    
    @staticmethod
    def setup_dtype(
        device: torch.device,
        dtype: Optional[torch.dtype] = None,
        prefer_float16: bool = True
    ) -> torch.dtype:
        """
        Setup data type based on device.
        
        Args:
            device: Target device
            dtype: Explicit dtype or None for auto-detection
            prefer_float16: Whether to prefer float16 on CUDA
            
        Returns:
            Configured torch.dtype
        """
        if dtype is not None:
            return dtype
        
        if device.type == "cuda" and prefer_float16:
            if torch.cuda.is_available():
                return torch.float16
        
        return torch.float32
    
    @staticmethod
    def move_to_device(
        model_or_tensor,
        device: torch.device,
        dtype: Optional[torch.dtype] = None
    ):
        """
        Move model or tensor to device with optional dtype conversion.
        
        Args:
            model_or_tensor: PyTorch model or tensor
            device: Target device
            dtype: Optional dtype to convert to
            
        Returns:
            Model or tensor on target device
        """
        if dtype is not None:
            if hasattr(model_or_tensor, "to"):
                return model_or_tensor.to(device=device, dtype=dtype)
        else:
            if hasattr(model_or_tensor, "to"):
                return model_or_tensor.to(device)
        
        return model_or_tensor


