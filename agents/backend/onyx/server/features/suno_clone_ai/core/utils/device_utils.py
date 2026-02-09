"""
Device Utilities

Advanced utilities for device management and GPU operations.
"""

import logging
import torch
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)


class DeviceManager:
    """Manage devices and GPU operations."""
    
    def __init__(self, device: Optional[torch.device] = None):
        """
        Initialize device manager.
        
        Args:
            device: Preferred device
        """
        self.device = device or self._get_best_device()
        logger.info(f"Device manager initialized: {self.device}")
    
    @staticmethod
    def _get_best_device() -> torch.device:
        """
        Get best available device.
        
        Returns:
            Best device
        """
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    
    def get_device_info(self) -> Dict[str, Any]:
        """
        Get device information.
        
        Returns:
            Device information dictionary
        """
        info = {
            'device': str(self.device),
            'type': self.device.type
        }
        
        if self.device.type == "cuda":
            info.update({
                'device_name': torch.cuda.get_device_name(self.device),
                'device_count': torch.cuda.device_count(),
                'memory_allocated': torch.cuda.memory_allocated(self.device) / (1024 ** 3),
                'memory_reserved': torch.cuda.memory_reserved(self.device) / (1024 ** 3),
                'memory_total': torch.cuda.get_device_properties(self.device).total_memory / (1024 ** 3)
            })
        
        return info
    
    def clear_cache(self) -> None:
        """Clear GPU cache."""
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            logger.debug("GPU cache cleared")
    
    def synchronize(self) -> None:
        """Synchronize device operations."""
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
    
    def set_memory_fraction(self, fraction: float) -> None:
        """
        Set memory fraction for CUDA.
        
        Args:
            fraction: Memory fraction (0.0 to 1.0)
        """
        if self.device.type == "cuda":
            torch.cuda.set_per_process_memory_fraction(fraction, self.device)
            logger.info(f"Set memory fraction to {fraction}")


def get_best_device() -> torch.device:
    """Get best available device."""
    return DeviceManager._get_best_device()


def get_device_info(device: Optional[torch.device] = None) -> Dict[str, Any]:
    """Get device information."""
    manager = DeviceManager(device)
    return manager.get_device_info()


def clear_gpu_cache(device: Optional[torch.device] = None) -> None:
    """Clear GPU cache."""
    manager = DeviceManager(device)
    manager.clear_cache()



