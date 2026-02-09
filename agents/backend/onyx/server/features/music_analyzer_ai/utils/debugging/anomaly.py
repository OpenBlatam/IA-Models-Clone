"""
Anomaly Detection Module

Anomaly detection utilities.
"""

import logging

logger = logging.getLogger(__name__)

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def enable_anomaly_detection():
    """Enable anomaly detection for debugging"""
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required for anomaly detection")
    
    torch.autograd.set_detect_anomaly(True)
    logger.warning("Anomaly detection enabled - this will slow down training")


def disable_anomaly_detection():
    """Disable anomaly detection"""
    if TORCH_AVAILABLE:
        torch.autograd.set_detect_anomaly(False)


def debug_training_step(
    model: torch.nn.Module,
    loss: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    check_gradients: bool = True,
    check_weights: bool = False
) -> bool:
    """
    Debug a training step.
    
    Args:
        model: Model being trained
        loss: Loss tensor
        optimizer: Optimizer
        check_gradients: Whether to check gradients
        check_weights: Whether to check weights
    
    Returns:
        True if training step is valid, False otherwise
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required")
    
    from .training import TrainingDebugger
    
    # Check loss
    if not TrainingDebugger.check_loss(loss):
        return False
    
    # Check gradients
    if check_gradients:
        issues = TrainingDebugger.check_gradients(model)
        if issues:
            return False
    
    # Check weights
    if check_weights:
        issues = TrainingDebugger.check_weights(model)
        if issues:
            return False
    
    return True



