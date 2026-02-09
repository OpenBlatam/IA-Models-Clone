"""
Device Utilities - GPU/CPU Management
=====================================

Utilities for managing devices and GPU utilization.
"""

import logging
from typing import Optional
import torch

logger = logging.getLogger(__name__)


def get_device(device: Optional[str] = None) -> torch.device:
    """
    Get appropriate device for computation.
    
    Args:
        device: Device string ('cuda', 'cpu', 'mps') or None for auto-detect
        
    Returns:
        torch.device instance
    """
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    
    device_obj = torch.device(device)
    
    if device_obj.type == 'cuda':
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    elif device_obj.type == 'mps':
        logger.info("Using MPS (Apple Silicon)")
    else:
        logger.info("Using CPU")
    
    return device_obj


def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    logger.info(f"Random seed set to {seed}")


def enable_anomaly_detection(enabled: bool = True) -> None:
    """
    Enable/disable PyTorch anomaly detection for debugging.
    
    Args:
        enabled: Whether to enable anomaly detection
    """
    torch.autograd.set_detect_anomaly(enabled)
    if enabled:
        logger.warning("Anomaly detection enabled - this will slow down training!")



