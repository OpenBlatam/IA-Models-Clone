"""
Helper Utilities Module
========================

Utility functions for validation, estimation, and formatting.

Author: BUL System
Date: 2024
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


def validate_dataset_path(dataset_path: str) -> Path:
    """
    Validate and return dataset path.
    
    Args:
        dataset_path: Path to dataset file
        
    Returns:
        Validated Path object
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If path is invalid
    """
    path = Path(dataset_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    
    if not path.is_file():
        raise ValueError(f"Dataset path is not a file: {dataset_path}")
    
    if path.suffix.lower() != '.json':
        logger.warning(f"Dataset file doesn't have .json extension: {dataset_path}")
    
    return path


def validate_model_name(model_name: str) -> bool:
    """
    Validate model name format.
    
    Args:
        model_name: Model name to validate
        
    Returns:
        True if valid, False otherwise
    """
    if not model_name or not isinstance(model_name, str):
        return False
    
    if len(model_name) < 2:
        return False
    
    # Check if it's a valid path or model name
    if Path(model_name).exists():
        return True
    
    # Model names typically contain alphanumeric, dashes, underscores, slashes
    import re
    pattern = r'^[a-zA-Z0-9_\-/]+$'
    return bool(re.match(pattern, model_name))


def estimate_training_time(
    num_samples: int,
    batch_size: int,
    num_epochs: int,
    steps_per_second: float = 1.0
) -> Dict[str, float]:
    """
    Estimate training time.
    
    Args:
        num_samples: Number of training samples
        batch_size: Batch size
        num_epochs: Number of epochs
        steps_per_second: Estimated steps per second
        
    Returns:
        Dictionary with time estimates
    """
    steps_per_epoch = num_samples / batch_size
    total_steps = steps_per_epoch * num_epochs
    
    total_seconds = total_steps / steps_per_second if steps_per_second > 0 else 0
    total_minutes = total_seconds / 60
    total_hours = total_minutes / 60
    
    return {
        "total_steps": total_steps,
        "steps_per_epoch": steps_per_epoch,
        "total_seconds": total_seconds,
        "total_minutes": total_minutes,
        "total_hours": total_hours,
    }


def calculate_model_size(model) -> Dict[str, float]:
    """
    Calculate model size in different units.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with size information
    """
    import torch
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Estimate size in bytes (assuming float32)
    size_bytes = total_params * 4
    size_mb = size_bytes / (1024 ** 2)
    size_gb = size_bytes / (1024 ** 3)
    
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "size_bytes": size_bytes,
        "size_mb": size_mb,
        "size_gb": size_gb,
        "parameters_millions": total_params / 1e6,
        "parameters_billions": total_params / 1e9,
    }


def format_training_summary(summary: Dict[str, Any]) -> str:
    """
    Format training summary as a readable string.
    
    Args:
        summary: Training summary dictionary
        
    Returns:
        Formatted string
    """
    lines = []
    lines.append("=" * 80)
    lines.append("Training Summary")
    lines.append("=" * 80)
    
    # Model info
    if "model_info" in summary:
        model_info = summary["model_info"]
        lines.append(f"\nModel: {model_info.get('model_name', 'N/A')}")
        lines.append(f"  Type: {model_info.get('model_type', 'N/A')}")
        lines.append(f"  Parameters: {model_info.get('parameters_millions', 0):.2f}M")
        lines.append(f"  Size: {model_info.get('model_size_gb', 0):.2f} GB")
    
    # Device info
    if "device_info" in summary:
        device_info = summary["device_info"]
        lines.append(f"\nDevice: {device_info.get('type', 'N/A')}")
        if device_info.get('type') == 'cuda':
            lines.append(f"  GPU: {device_info.get('device_name', 'N/A')}")
            lines.append(f"  Memory: {device_info.get('memory_total', 0):.1f} GB")
    
    # Dataset info
    if "dataset_info" in summary:
        dataset_info = summary["dataset_info"]
        lines.append(f"\nDataset:")
        lines.append(f"  Train: {dataset_info.get('train_size', 0)} samples")
        lines.append(f"  Eval: {dataset_info.get('eval_size', 0)} samples")
    
    # Training config
    if "training_config" in summary:
        config = summary["training_config"]
        lines.append(f"\nTraining Configuration:")
        lines.append(f"  Learning Rate: {config.get('learning_rate', 'N/A')}")
        lines.append(f"  Epochs: {config.get('num_epochs', 'N/A')}")
        lines.append(f"  Batch Size: {config.get('batch_size', 'N/A')}")
        lines.append(f"  Effective Batch Size: {config.get('effective_batch_size', 'N/A')}")
        lines.append(f"  Max Length: {config.get('max_length', 'N/A')}")
        lines.append(f"  FP16: {config.get('fp16', False)}")
        lines.append(f"  BF16: {config.get('bf16', False)}")
    
    lines.append("=" * 80)
    
    return "\n".join(lines)

