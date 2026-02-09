"""
Model Inspection

Utilities for inspecting model architecture and parameters.
"""

import logging
import torch
import torch.nn as nn
from typing import Dict, Optional
from collections import OrderedDict

logger = logging.getLogger(__name__)


class ModelInspector:
    """Inspect model architecture and parameters."""
    
    @staticmethod
    def count_parameters(model: nn.Module) -> Dict[str, int]:
        """
        Count model parameters.
        
        Args:
            model: Model to count
            
        Returns:
            Dictionary with parameter counts
        """
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        non_trainable_params = total_params - trainable_params
        
        return {
            'total': total_params,
            'trainable': trainable_params,
            'non_trainable': non_trainable_params
        }
    
    @staticmethod
    def get_model_summary(
        model: nn.Module,
        input_size: Optional[tuple] = None
    ) -> Dict:
        """
        Get model summary.
        
        Args:
            model: Model to summarize
            input_size: Input size for forward pass
            
        Returns:
            Model summary dictionary
        """
        summary = {
            'name': model.__class__.__name__,
            'parameters': ModelInspector.count_parameters(model),
            'layers': []
        }
        
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf node
                layer_info = {
                    'name': name,
                    'type': module.__class__.__name__,
                    'parameters': sum(p.numel() for p in module.parameters())
                }
                summary['layers'].append(layer_info)
        
        return summary
    
    @staticmethod
    def inspect_layer(
        model: nn.Module,
        layer_name: str
    ) -> Optional[Dict]:
        """
        Inspect specific layer.
        
        Args:
            model: Model to inspect
            layer_name: Name of layer
            
        Returns:
            Layer information or None
        """
        for name, module in model.named_modules():
            if name == layer_name:
                return {
                    'name': name,
                    'type': module.__class__.__name__,
                    'parameters': dict(module.named_parameters()),
                    'buffers': dict(module.named_buffers())
                }
        
        return None


def inspect_model(
    model: nn.Module,
    input_size: Optional[tuple] = None
) -> Dict:
    """
    Convenience function to inspect model.
    
    Args:
        model: Model to inspect
        input_size: Input size
        
    Returns:
        Model summary
    """
    return ModelInspector.get_model_summary(model, input_size)


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Convenience function to count parameters.
    
    Args:
        model: Model to count
        
    Returns:
        Parameter counts
    """
    return ModelInspector.count_parameters(model)


def get_model_summary(
    model: nn.Module,
    input_size: Optional[tuple] = None
) -> Dict:
    """
    Convenience function to get model summary.
    
    Args:
        model: Model to summarize
        input_size: Input size
        
    Returns:
        Model summary
    """
    return ModelInspector.get_model_summary(model, input_size)



