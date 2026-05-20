"""
Model Analysis Utilities
========================

Utilities for analyzing and understanding models:
- Model complexity analysis
- Layer-wise analysis
- Activation analysis
- Gradient flow analysis
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
import torch
import torch.nn as nn
import numpy as np

logger = logging.getLogger(__name__)


def analyze_model_complexity(model: nn.Module) -> Dict[str, Any]:
    """
    Analyze model complexity.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with complexity metrics
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Count layers by type
    layer_counts = {}
    for module in model.modules():
        module_type = type(module).__name__
        if module_type not in ['Module', 'Sequential']:
            layer_counts[module_type] = layer_counts.get(module_type, 0) + 1
    
    # Calculate memory estimate (rough)
    param_memory_mb = total_params * 4 / 1024**2  # Assuming float32
    
    analysis = {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': total_params - trainable_params,
        'estimated_memory_mb': param_memory_mb,
        'layer_counts': layer_counts,
        'num_layers': len(list(model.modules())) - 1  # Exclude root
    }
    
    logger.info(f"Model complexity analysis: {analysis}")
    return analysis


def analyze_gradient_flow(model: nn.Module) -> Dict[str, float]:
    """
    Analyze gradient flow through model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with gradient statistics per layer
    """
    grad_stats = {}
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_mean = param.grad.mean().item()
            grad_std = param.grad.std().item()
            
            grad_stats[name] = {
                'norm': grad_norm,
                'mean': grad_mean,
                'std': grad_std,
                'max': param.grad.max().item(),
                'min': param.grad.min().item()
            }
    
    return grad_stats


def get_layer_output_shapes(
    model: nn.Module,
    input_shape: tuple,
    device: Optional[torch.device] = None
) -> Dict[str, tuple]:
    """
    Get output shapes for each layer.
    
    Args:
        model: PyTorch model
        input_shape: Input tensor shape
        device: Device to run on
        
    Returns:
        Dictionary mapping layer names to output shapes
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = model.to(device)
    model.eval()
    
    shapes = {}
    hooks = []
    
    def hook_fn(name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                shapes[name] = tuple(output.shape)
            elif isinstance(output, (tuple, list)):
                shapes[name] = [tuple(o.shape) if isinstance(o, torch.Tensor) else None
                               for o in output]
        return hook
    
    # Register hooks
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules
            hooks.append(module.register_forward_hook(hook_fn(name)))
    
    # Forward pass
    dummy_input = torch.randn(input_shape).to(device)
    with torch.no_grad():
        _ = model(dummy_input)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    return shapes


def check_model_health(model: nn.Module) -> Tuple[bool, List[str]]:
    """
    Check model health (NaN, Inf, etc.).
    
    Args:
        model: PyTorch model
        
    Returns:
        Tuple of (is_healthy, list_of_issues)
    """
    issues = []
    
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            issues.append(f"NaN in parameter: {name}")
        if torch.isinf(param).any():
            issues.append(f"Inf in parameter: {name}")
        if param.grad is not None:
            if torch.isnan(param.grad).any():
                issues.append(f"NaN in gradient: {name}")
            if torch.isinf(param.grad).any():
                issues.append(f"Inf in gradient: {name}")
    
    is_healthy = len(issues) == 0
    
    if not is_healthy:
        logger.warning(f"Model health issues found: {issues}")
    
    return is_healthy, issues



