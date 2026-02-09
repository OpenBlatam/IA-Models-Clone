"""
Model Utilities for Deep Learning Best Practices

Provides shared utilities for:
- Model compilation
- Model analysis
- Reproducibility

Note: Other utilities are now in specialized modules:
- Device management: utils.device
- Gradient management: utils.gradients
- Validation: utils.validation
- Initialization: utils.initialization (re-exports from models.initialization)
"""

import logging
from typing import Optional
import torch
import torch.nn as nn

# Import from granular modules for backward compatibility
from .device import get_device, setup_gpu_optimizations, clear_gpu_cache
from .gradients import clip_gradients
from .validation import check_for_nan_inf
from .initialization import initialize_weights

logger = logging.getLogger(__name__)


def compile_model(
    model: nn.Module,
    mode: str = "reduce-overhead",
    fullgraph: bool = False
) -> nn.Module:
    """
    Compile model for faster inference (PyTorch 2.0+).
    
    Args:
        model: Model to compile
        mode: Compilation mode ('default', 'reduce-overhead', 'max-autotune')
        fullgraph: Whether to compile full graph
        
    Returns:
        Compiled model (or original if compilation fails)
    """
    if not hasattr(torch, 'compile'):
        logger.warning("torch.compile not available (requires PyTorch 2.0+)")
        return model
    
    try:
        compiled_model = torch.compile(
            model,
            mode=mode,
            fullgraph=fullgraph
        )
        logger.info(f"Model compiled with mode: {mode}")
        return compiled_model
    except Exception as e:
        logger.warning(f"Could not compile model: {e}")
        return model


def enable_gradient_checkpointing(model: nn.Module) -> None:
    """
    Enable gradient checkpointing to save memory.
    
    Args:
        model: Model to enable checkpointing for
    """
    if hasattr(model, 'gradient_checkpointing_enable'):
        try:
            model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")
        except Exception as e:
            logger.debug(f"Could not enable gradient checkpointing: {e}")
    else:
        # Try to enable for submodules
        for module in model.modules():
            if hasattr(module, 'gradient_checkpointing_enable'):
                try:
                    module.gradient_checkpointing_enable()
                    logger.info("Gradient checkpointing enabled for submodules")
                    break
                except Exception:
                    continue


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """
    Count model parameters.
    
    Args:
        model: Model to count parameters for
        trainable_only: Whether to count only trainable parameters
        
    Returns:
        Number of parameters
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def get_model_size_mb(model: nn.Module, trainable_only: bool = True) -> float:
    """
    Get model size in megabytes.
    
    Args:
        model: Model to get size for
        trainable_only: Whether to count only trainable parameters
        
    Returns:
        Model size in MB
    """
    num_params = count_parameters(model, trainable_only)
    # Assume float32 (4 bytes per parameter)
    size_bytes = num_params * 4
    size_mb = size_bytes / (1024 ** 2)
    return size_mb


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
        torch.cuda.manual_seed_all(seed)
    
    logger.info(f"Random seed set to {seed}")


# Re-export for backward compatibility
__all__ = [
    "initialize_weights",
    "clip_gradients",
    "check_for_nan_inf",
    "get_device",
    "setup_gpu_optimizations",
    "compile_model",
    "enable_gradient_checkpointing",
    "count_parameters",
    "get_model_size_mb",
    "clear_gpu_cache",
    "set_seed"
]
