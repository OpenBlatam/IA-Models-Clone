"""
Device Manager
==============

Manages device configuration and selection for models.
"""

import logging
from typing import Optional
import torch

logger = logging.getLogger(__name__)


class DeviceManager:
    """Manages device configuration and selection."""
    
    @staticmethod
    def get_device(device: Optional[str] = None) -> torch.device:
        """
        Get or create a torch device.
        
        Args:
            device: Device string (cuda/cpu/auto) or None for auto
            
        Returns:
            Torch device
        """
        if device is None or device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        return torch.device(device)
    
    @staticmethod
    def get_dtype(device: torch.device, dtype: Optional[torch.dtype] = None) -> torch.dtype:
        """
        Get appropriate dtype for device.
        
        Args:
            device: Torch device
            dtype: Optional dtype override
            
        Returns:
            Torch dtype
        """
        if dtype is not None:
            return dtype
        
        return torch.float16 if device.type == "cuda" else torch.float32
    
    @staticmethod
    def get_device_info(device: torch.device) -> dict:
        """
        Get information about the device.
        
        Args:
            device: Torch device
            
        Returns:
            Device information dictionary
        """
        info = {
            "device": str(device),
            "type": device.type,
        }
        
        if device.type == "cuda":
            info.update({
                "cuda_available": torch.cuda.is_available(),
                "cuda_device_count": torch.cuda.device_count(),
                "cuda_current_device": torch.cuda.current_device() if torch.cuda.is_available() else None,
                "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            })
        
        return info


