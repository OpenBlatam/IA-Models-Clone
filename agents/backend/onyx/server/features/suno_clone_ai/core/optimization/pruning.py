"""
Model Pruning

Utilities for model pruning.
"""

import logging
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from typing import Optional, List

logger = logging.getLogger(__name__)


class Pruner:
    """Prune models to reduce size."""
    
    @staticmethod
    def magnitude_pruning(
        model: nn.Module,
        amount: float = 0.2,
        module_types: Optional[List[type]] = None
    ) -> nn.Module:
        """
        Apply magnitude-based pruning.
        
        Args:
            model: Model to prune
            amount: Pruning amount (0.0 to 1.0)
            module_types: Types of modules to prune
            
        Returns:
            Pruned model
        """
        if module_types is None:
            module_types = [nn.Linear, nn.Conv1d, nn.Conv2d]
        
        for name, module in model.named_modules():
            if isinstance(module, tuple(module_types)):
                prune.l1_unstructured(module, name='weight', amount=amount)
                prune.remove(module, 'weight')
        
        logger.info(f"Applied magnitude pruning with amount={amount}")
        return model
    
    @staticmethod
    def structured_pruning(
        model: nn.Module,
        amount: float = 0.2,
        module_types: Optional[List[type]] = None
    ) -> nn.Module:
        """
        Apply structured pruning.
        
        Args:
            model: Model to prune
            amount: Pruning amount (0.0 to 1.0)
            module_types: Types of modules to prune
            
        Returns:
            Pruned model
        """
        if module_types is None:
            module_types = [nn.Linear, nn.Conv1d, nn.Conv2d]
        
        for name, module in model.named_modules():
            if isinstance(module, tuple(module_types)):
                prune.ln_structured(module, name='weight', amount=amount, n=2, dim=0)
                prune.remove(module, 'weight')
        
        logger.info(f"Applied structured pruning with amount={amount}")
        return model


def prune_model(
    model: nn.Module,
    method: str = "magnitude",
    **kwargs
) -> nn.Module:
    """
    Prune model.
    
    Args:
        model: Model to prune
        method: Pruning method ('magnitude', 'structured')
        **kwargs: Additional arguments
        
    Returns:
        Pruned model
    """
    pruner = Pruner()
    
    if method == "magnitude":
        return pruner.magnitude_pruning(model, **kwargs)
    elif method == "structured":
        return pruner.structured_pruning(model, **kwargs)
    else:
        raise ValueError(f"Unknown pruning method: {method}")


def magnitude_pruning(
    model: nn.Module,
    amount: float = 0.2
) -> nn.Module:
    """Convenience function for magnitude pruning."""
    return Pruner.magnitude_pruning(model, amount)


def structured_pruning(
    model: nn.Module,
    amount: float = 0.2
) -> nn.Module:
    """Convenience function for structured pruning."""
    return Pruner.structured_pruning(model, amount)



