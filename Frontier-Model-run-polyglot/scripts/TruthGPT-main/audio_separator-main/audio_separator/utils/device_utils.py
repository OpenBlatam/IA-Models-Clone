"""
Device utilities for GPU/CPU management.
Refactored to use constants and improve organization.
"""

import torch
from typing import Optional, Dict, Any

from .constants import DEFAULT_DEVICE, VALID_DEVICES
from ..logger import logger


def get_device(device: str = DEFAULT_DEVICE) -> torch.device:
    """
    Get the appropriate device for computation.
    
    Args:
        device: Device preference ("auto", "cpu", "cuda", "mps")
        
    Returns:
        torch.device object
    """
    if device not in VALID_DEVICES:
        logger.warning(f"Invalid device '{device}', using '{DEFAULT_DEVICE}'")
        device = DEFAULT_DEVICE
    
    if device == "auto":
        device = _detect_best_device()
    elif device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available, falling back to CPU")
        device = "cpu"
    elif device == "mps" and not _is_mps_available():
        logger.warning("MPS requested but not available, falling back to CPU")
        device = "cpu"
    
    return torch.device(device)


def _detect_best_device() -> str:
    """
    Detect the best available device.
    
    Returns:
        Device name ("cuda", "mps", or "cpu")
    """
    if torch.cuda.is_available():
        logger.info("Using CUDA device")
        return "cuda"
    elif _is_mps_available():
        logger.info("Using MPS device (Apple Silicon)")
        return "mps"
    else:
        logger.info("Using CPU device")
        return "cpu"


def _is_mps_available() -> bool:
    """
    Check if MPS (Apple Silicon) is available.
    
    Returns:
        True if MPS is available
    """
    return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


def move_to_device(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Move tensor to specified device.
    
    Args:
        tensor: Input tensor
        device: Target device
        
    Returns:
        Tensor on target device
    """
    return tensor.to(device)


def get_device_info() -> Dict[str, Any]:
    """
    Get information about available devices.
    
    Returns:
        Dictionary with device information
    """
    info: Dict[str, Any] = {
        "cpu": True,
        "cuda": torch.cuda.is_available(),
        "mps": _is_mps_available(),
        "cuda_devices": []
    }
    
    if torch.cuda.is_available():
        info["cuda_devices"] = [
            {
                "index": i,
                "name": torch.cuda.get_device_name(i),
                "memory": f"{torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB"
            }
            for i in range(torch.cuda.device_count())
        ]
    
    return info
