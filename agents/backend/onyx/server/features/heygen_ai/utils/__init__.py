"""
Utils Module for HeyGen AI
===========================

This module contains utility functions following best practices:
- Device detection and management
- Logging utilities
- Configuration loading
- Common helper functions
- Gradio integration utilities
"""

from typing import Optional

import torch

# Import device management
from .device_manager import (
    detect_device,
    get_torch_dtype,
    clear_cuda_cache,
    get_device_info,
)


def get_device(device: Optional[str] = None) -> torch.device:
    """Get appropriate PyTorch device.
    
    Args:
        device: Device string ('cuda', 'cpu', 'mps', or None for auto)
    
    Returns:
        PyTorch device
    """
    if device:
        return torch.device(device)
    
    return detect_device()


def setup_logging(level: str = "INFO"):
    """Setup logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    import logging
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


# Import Gradio utilities
from .gradio_utils import (
    GradioInterface,
    validate_input,
    handle_errors,
)

__all__ = [
    "get_device",
    "setup_logging",
    "detect_device",
    "get_torch_dtype",
    "clear_cuda_cache",
    "get_device_info",
    "GradioInterface",
    "validate_input",
    "handle_errors",
]
