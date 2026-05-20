"""
Device Management Utilities
============================

Shared utilities for device detection and management across all modules.
Follows best practices for GPU utilization and mixed precision.
"""

import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)


def detect_device(device: Optional[torch.device] = None) -> torch.device:
    """Detect and return appropriate PyTorch device.
    
    Args:
        device: Optional device to use. If None, auto-detects.
    
    Returns:
        PyTorch device (cuda, mps, or cpu)
    """
    if device is not None:
        return device
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using MPS device")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU device")
    
    return device


def get_torch_dtype(device: torch.device) -> torch.dtype:
    """Get appropriate torch dtype based on device.
    
    Args:
        device: PyTorch device
    
    Returns:
        Appropriate torch dtype (float16 for CUDA, float32 otherwise)
    """
    if device.type == "cuda":
        return torch.float16  # Use FP16 on CUDA for better performance
    elif device.type == "mps":
        return torch.float32  # MPS doesn't support FP16
    else:
        return torch.float32  # Use FP32 on CPU


def clear_cuda_cache() -> None:
    """Clear CUDA cache and run garbage collection."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        import gc
        gc.collect()
        logger.debug("CUDA cache cleared")


def get_device_info(device: torch.device) -> dict:
    """Get information about the device.
    
    Args:
        device: PyTorch device
    
    Returns:
        Dictionary with device information
    """
    info = {
        "type": device.type,
        "index": device.index if device.index is not None else 0,
    }
    
    if device.type == "cuda":
        info["name"] = torch.cuda.get_device_name(device.index or 0)
        info["memory_total"] = torch.cuda.get_device_properties(
            device.index or 0
        ).total_memory
        info["memory_allocated"] = torch.cuda.memory_allocated(device.index or 0)
        info["memory_reserved"] = torch.cuda.memory_reserved(device.index or 0)
    
    return info



