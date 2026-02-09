"""
Gradient Management Utilities

Handles gradient clipping, monitoring, and validation.
"""

import logging
from typing import Optional, List
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def clip_gradients(
    model: nn.Module,
    max_norm: float = 1.0,
    norm_type: float = 2.0
) -> float:
    """
    Clip gradients to prevent exploding gradients.
    
    Args:
        model: PyTorch model
        max_norm: Maximum gradient norm
        norm_type: Type of norm (2.0 for L2 norm)
        
    Returns:
        Total gradient norm before clipping
    """
    if not hasattr(model, 'parameters'):
        return 0.0
    
    total_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm,
        norm_type=norm_type
    )
    return total_norm.item()


def get_gradient_norm(
    model: nn.Module,
    norm_type: float = 2.0
) -> float:
    """
    Get gradient norm without clipping.
    
    Args:
        model: PyTorch model
        norm_type: Type of norm
        
    Returns:
        Gradient norm
    """
    if not hasattr(model, 'parameters'):
        return 0.0
    
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
    
    total_norm = total_norm ** (1.0 / norm_type)
    return total_norm


def check_gradients(
    model: nn.Module,
    check_nan: bool = True,
    check_inf: bool = True,
    check_zero: bool = False
) -> dict:
    """
    Check gradients for issues.
    
    Args:
        model: PyTorch model
        check_nan: Check for NaN values
        check_inf: Check for Inf values
        check_zero: Check for zero gradients
        
    Returns:
        Dictionary with check results
    """
    results = {
        "has_nan": False,
        "has_inf": False,
        "has_zero": False,
        "num_params": 0,
        "num_params_with_grad": 0
    }
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            results["num_params_with_grad"] += 1
            
            if check_nan and torch.isnan(param.grad).any():
                results["has_nan"] = True
                logger.warning(f"NaN gradient detected in {name}")
            
            if check_inf and torch.isinf(param.grad).any():
                results["has_inf"] = True
                logger.warning(f"Inf gradient detected in {name}")
            
            if check_zero and (param.grad == 0).all():
                results["has_zero"] = True
                logger.warning(f"Zero gradient detected in {name}")
        
        results["num_params"] += 1
    
    return results



