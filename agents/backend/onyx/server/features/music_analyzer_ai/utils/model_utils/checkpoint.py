"""
Model Checkpoint Module

Checkpoint saving and loading utilities.
"""

from typing import Dict, Any, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def save_model_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    loss: float,
    filepath: str,
    metadata: Optional[Dict[str, Any]] = None
):
    """
    Save model checkpoint with metadata.
    
    Args:
        model: Model to save
        optimizer: Optimizer to save
        epoch: Current epoch
        loss: Current loss
        filepath: Path to save checkpoint
        metadata: Additional metadata
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required")
    
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "loss": loss,
        "metadata": metadata or {}
    }
    
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, filepath)
    logger.info(f"Saved checkpoint to {filepath}")


def load_model_checkpoint(
    model: nn.Module,
    filepath: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    Load model checkpoint.
    
    Args:
        model: Model to load checkpoint into
        filepath: Path to checkpoint file
        optimizer: Optimizer to load state into
        device: Device to load checkpoint on
    
    Returns:
        Checkpoint dictionary
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required")
    
    checkpoint = torch.load(filepath, map_location=device)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    logger.info(f"Loaded checkpoint from {filepath}")
    return checkpoint



