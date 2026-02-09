"""
Device Management Utilities

Handles device selection, GPU optimization, and device information.
"""

import logging
from typing import Optional, Dict, Any
import torch

logger = logging.getLogger(__name__)


def get_device(use_gpu: bool = True, device_id: Optional[int] = None) -> torch.device:
    """
    Get the appropriate device for computation.
    
    Args:
        use_gpu: Whether to use GPU if available
        device_id: Specific GPU device ID
        
    Returns:
        PyTorch device
    """
    if use_gpu and torch.cuda.is_available():
        if device_id is not None:
            device = torch.device(f"cuda:{device_id}")
        else:
            device = torch.device("cuda")
        logger.info(f"Using GPU: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")
    
    return device


def setup_gpu_optimizations() -> None:
    """Setup optimal GPU settings for performance."""
    if not torch.cuda.is_available():
        return
    
    # Enable cuDNN benchmarking for faster convolutions
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    
    # Enable TensorFloat-32 for faster operations on Ampere GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    logger.info("GPU optimizations enabled")


def clear_gpu_cache() -> None:
    """Clear GPU cache and synchronize."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        logger.debug("GPU cache cleared")


def get_device_info(device: Optional[torch.device] = None) -> Dict[str, Any]:
    """
    Get information about the device.
    
    Args:
        device: Device to get info for (uses current if None)
        
    Returns:
        Dictionary with device information
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    info = {
        "device": str(device),
        "type": device.type
    }
    
    if device.type == "cuda":
        info.update({
            "name": torch.cuda.get_device_name(device),
            "capability": torch.cuda.get_device_capability(device),
            "memory_allocated_gb": torch.cuda.memory_allocated(device) / (1024**3),
            "memory_reserved_gb": torch.cuda.memory_reserved(device) / (1024**3),
            "cuda_version": torch.version.cuda
        })
    
    return info



