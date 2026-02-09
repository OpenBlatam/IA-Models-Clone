"""
Model Pruning - Model Compression
==================================

Provides pruning techniques for model compression:
- Magnitude-based pruning
- Structured pruning
- Unstructured pruning
"""

import logging
from typing import Optional, Dict, Any, Callable
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

logger = logging.getLogger(__name__)


def prune_model(
    model: nn.Module,
    pruning_method: str = 'magnitude',
    amount: float = 0.2,
    module_types: tuple = (nn.Linear, nn.Conv2d)
) -> nn.Module:
    """
    Prune model weights.
    
    Args:
        model: PyTorch model
        pruning_method: Pruning method ('magnitude', 'random', 'ln_structured')
        amount: Fraction of parameters to prune (0.0 to 1.0)
        module_types: Types of modules to prune
        
    Returns:
        Pruned model
    """
    for name, module in model.named_modules():
        if isinstance(module, module_types):
            if pruning_method == 'magnitude':
                prune.l1_unstructured(module, name='weight', amount=amount)
            elif pruning_method == 'random':
                prune.random_unstructured(module, name='weight', amount=amount)
            elif pruning_method == 'ln_structured':
                prune.ln_structured(module, name='weight', amount=amount, n=2, dim=0)
            else:
                raise ValueError(f"Unknown pruning method: {pruning_method}")
    
    # Remove pruning masks (make pruning permanent)
    for name, module in model.named_modules():
        if isinstance(module, module_types):
            prune.remove(module, 'weight')
    
    logger.info(f"Model pruned using {pruning_method} method ({amount*100:.1f}%)")
    return model


def get_pruning_sparsity(model: nn.Module) -> Dict[str, float]:
    """
    Calculate sparsity of pruned model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with sparsity per layer
    """
    sparsity = {}
    
    for name, module in model.named_modules():
        if hasattr(module, 'weight'):
            weight = module.weight
            if weight is not None:
                sparsity[name] = float(torch.sum(weight == 0) / weight.numel())
    
    return sparsity


def iterative_pruning(
    model: nn.Module,
    pruning_steps: int = 5,
    final_sparsity: float = 0.8,
    module_types: tuple = (nn.Linear, nn.Conv2d)
) -> nn.Module:
    """
    Iterative pruning (gradual pruning).
    
    Args:
        model: PyTorch model
        pruning_steps: Number of pruning steps
        final_sparsity: Final target sparsity
        module_types: Types of modules to prune
        
    Returns:
        Pruned model
    """
    amount_per_step = final_sparsity / pruning_steps
    
    for step in range(pruning_steps):
        prune_model(model, pruning_method='magnitude', amount=amount_per_step, module_types=module_types)
        current_sparsity = sum(get_pruning_sparsity(model).values()) / len(get_pruning_sparsity(model))
        logger.info(f"Pruning step {step+1}/{pruning_steps}, sparsity: {current_sparsity:.2%}")
    
    return model



