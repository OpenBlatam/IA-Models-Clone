"""
Parameter Counter
=================

Counts and analyzes model parameters.
"""

import torch.nn as nn
from typing import Union, Dict, Any


class ParameterCounter:
    """Counts and analyzes model parameters."""
    
    @staticmethod
    def count_parameters(
        model: nn.Module,
        trainable_only: bool = False,
        by_layer: bool = False
    ) -> Union[int, Dict[str, int]]:
        """
        Count model parameters with optional layer breakdown.
        
        Args:
            model: PyTorch model
            trainable_only: If True, only count trainable parameters
            by_layer: If True, return breakdown by layer
            
        Returns:
            Parameter count or dictionary by layer
        """
        if by_layer:
            counts = {}
            for name, param in model.named_parameters():
                if not trainable_only or param.requires_grad:
                    counts[name] = param.numel()
            return counts
        
        if trainable_only:
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        return sum(p.numel() for p in model.parameters())
    
    @staticmethod
    def get_parameter_stats(model: nn.Module) -> Dict[str, Any]:
        """
        Get detailed parameter statistics.
        
        Args:
            model: PyTorch model
            
        Returns:
            Dictionary with parameter statistics
        """
        total = ParameterCounter.count_parameters(model, trainable_only=False)
        trainable = ParameterCounter.count_parameters(model, trainable_only=True)
        by_layer = ParameterCounter.count_parameters(model, by_layer=True)
        
        return {
            "total": total,
            "trainable": trainable,
            "non_trainable": total - trainable,
            "trainable_percentage": (trainable / total * 100) if total > 0 else 0,
            "by_layer": by_layer,
            "num_layers": len(by_layer),
        }


