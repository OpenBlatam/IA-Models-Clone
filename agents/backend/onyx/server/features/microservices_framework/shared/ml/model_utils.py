"""
Shared ML Model Utilities
Common utilities for model loading, inference, and optimization.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Union
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def get_device() -> str:
    """Get available device (cuda, mps, or cpu)."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def get_dtype(device: str, use_fp16: bool = True) -> torch.dtype:
    """Get appropriate dtype for device."""
    if device == "cuda" and use_fp16:
        return torch.float16
    return torch.float32


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def estimate_model_size_mb(model: nn.Module, dtype: torch.dtype = torch.float32) -> float:
    """Estimate model size in MB."""
    param_size = sum(p.numel() * dtype.itemsize for p in model.parameters())
    buffer_size = sum(b.numel() * dtype.itemsize for b in model.buffers())
    return (param_size + buffer_size) / (1024 * 1024)


def load_model_checkpoint(
    checkpoint_path: Union[str, Path],
    model: nn.Module,
    device: str = "cpu",
    strict: bool = True
) -> nn.Module:
    """Load model checkpoint."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"], strict=strict)
        elif "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
        else:
            model.load_state_dict(checkpoint, strict=strict)
    else:
        model.load_state_dict(checkpoint, strict=strict)
    
    logger.info(f"Loaded checkpoint from {checkpoint_path}")
    return model


def save_model_checkpoint(
    model: nn.Module,
    checkpoint_path: Union[str, Path],
    additional_info: Optional[Dict[str, Any]] = None
):
    """Save model checkpoint."""
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        "state_dict": model.state_dict(),
        "model_class": model.__class__.__name__,
    }
    
    if additional_info:
        checkpoint.update(additional_info)
    
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Saved checkpoint to {checkpoint_path}")


def enable_mixed_precision(model: nn.Module, device: str) -> nn.Module:
    """Enable mixed precision if CUDA is available."""
    if device == "cuda":
        try:
            from torch.cuda.amp import autocast
            return model.half()
        except Exception as e:
            logger.warning(f"Failed to enable mixed precision: {e}")
    return model


def freeze_model_parameters(model: nn.Module, freeze_all: bool = False):
    """Freeze model parameters."""
    if freeze_all:
        for param in model.parameters():
            param.requires_grad = False
    else:
        # Freeze only certain layers (e.g., embeddings)
        for name, param in model.named_parameters():
            if "embedding" in name.lower() or "encoder" in name.lower():
                param.requires_grad = False


def get_model_summary(model: nn.Module) -> Dict[str, Any]:
    """Get model summary information."""
    total_params = count_parameters(model)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "frozen_parameters": total_params - trainable_params,
        "model_size_mb": estimate_model_size_mb(model),
        "device": next(model.parameters()).device.type if list(model.parameters()) else "unknown",
    }


def clip_gradients(model: nn.Module, max_norm: float = 1.0):
    """Clip gradients to prevent exploding gradients."""
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)


def check_for_nan_inf(model: nn.Module) -> bool:
    """Check for NaN or Inf values in model parameters."""
    for name, param in model.named_parameters():
        if torch.isnan(param).any() or torch.isinf(param).any():
            logger.warning(f"NaN or Inf detected in {name}")
            return True
    return False



