"""
Helper Utilities - Common Utilities
===================================

Common helper functions for deep learning workflows.
"""

import torch
import numpy as np
from typing import Optional, Tuple, Dict, Any, Union
import logging
from pathlib import Path

try:
    import structlog
    logger = structlog.get_logger(__name__)
except ImportError:
    logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info(f"✅ Random seed set to {seed}")


def get_device(device_name: Optional[str] = None) -> torch.device:
    """
    Get optimal device.
    
    Args:
        device_name: Device name (cuda, cpu, or None for auto)
    
    Returns:
        torch.device
    """
    if device_name is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    
    if device.type == "cuda":
        logger.info(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.info("✅ Using CPU")
    
    return device


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """
    Count model parameters.
    
    Args:
        model: PyTorch model
    
    Returns:
        Dictionary with parameter counts
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    
    return {
        "trainable": trainable,
        "total": total,
        "non_trainable": total - trainable,
        "trainable_percentage": 100.0 * trainable / total if total > 0 else 0.0
    }


def format_size(size_bytes: int) -> str:
    """
    Format bytes to human-readable size.
    
    Args:
        size_bytes: Size in bytes
    
    Returns:
        Formatted string
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def get_model_size(model: torch.nn.Module) -> Dict[str, Any]:
    """
    Get model size information.
    
    Args:
        model: PyTorch model
    
    Returns:
        Dictionary with size information
    """
    param_count = count_parameters(model)
    
    # Estimate memory usage
    total_params = param_count["total"]
    # Assume float32 (4 bytes per parameter)
    memory_bytes = total_params * 4
    
    return {
        **param_count,
        "estimated_memory_mb": memory_bytes / (1024 ** 2),
        "estimated_memory": format_size(memory_bytes)
    }


def save_model_summary(model: torch.nn.Module, path: Union[str, Path]) -> None:
    """
    Save model summary to file.
    
    Args:
        model: PyTorch model
        path: Output path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    summary = {
        "model_class": model.__class__.__name__,
        "parameters": count_parameters(model),
        "size": get_model_size(model)
    }
    
    with open(path, 'w') as f:
        import json
        json.dump(summary, f, indent=2)
    
    logger.info(f"✅ Model summary saved to {path}")

