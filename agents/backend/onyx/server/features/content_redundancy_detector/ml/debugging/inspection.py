"""
Model Inspection
Advanced model inspection utilities
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


class ModelInspector:
    """
    Advanced model inspection utilities
    """
    
    def __init__(self, model: nn.Module):
        """
        Initialize model inspector
        
        Args:
            model: Model to inspect
        """
        self.model = model
        self.hooks = []
        self.activations = {}
        self.gradients = {}
    
    def register_hooks(self) -> None:
        """Register forward and backward hooks"""
        def forward_hook(name):
            def hook(module, input, output):
                self.activations[name] = output.detach()
            return hook
        
        def backward_hook(name):
            def hook(module, grad_input, grad_output):
                if grad_output[0] is not None:
                    self.gradients[name] = grad_output[0].detach()
            return hook
        
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # Leaf node
                hook_f = module.register_forward_hook(forward_hook(name))
                hook_b = module.register_backward_hook(backward_hook(name))
                self.hooks.extend([hook_f, hook_b])
    
    def remove_hooks(self) -> None:
        """Remove all hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
    
    def inspect_forward(self, inputs: torch.Tensor) -> Dict[str, Any]:
        """
        Inspect forward pass
        
        Args:
            inputs: Input tensor
            
        Returns:
            Dictionary with activation information
        """
        self.register_hooks()
        self.activations.clear()
        
        try:
            with torch.no_grad():
                _ = self.model(inputs)
        finally:
            self.remove_hooks()
        
        return {
            'activations': {
                name: {
                    'shape': list(act.shape),
                    'mean': float(act.mean().item()),
                    'std': float(act.std().item()),
                    'min': float(act.min().item()),
                    'max': float(act.max().item()),
                }
                for name, act in self.activations.items()
            }
        }
    
    def inspect_backward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        criterion: nn.Module,
    ) -> Dict[str, Any]:
        """
        Inspect backward pass
        
        Args:
            inputs: Input tensor
            targets: Target tensor
            criterion: Loss function
            
        Returns:
            Dictionary with gradient information
        """
        self.register_hooks()
        self.gradients.clear()
        
        try:
            outputs = self.model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
        finally:
            self.remove_hooks()
        
        return {
            'gradients': {
                name: {
                    'shape': list(grad.shape),
                    'mean': float(grad.mean().item()),
                    'std': float(grad.std().item()),
                    'norm': float(grad.norm().item()),
                }
                for name, grad in self.gradients.items()
            }
        }
    
    def get_layer_info(self) -> List[Dict[str, Any]]:
        """
        Get information about all layers
        
        Returns:
            List of layer information dictionaries
        """
        layers = []
        
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # Leaf node
                params = sum(p.numel() for p in module.parameters())
                layers.append({
                    'name': name,
                    'type': module.__class__.__name__,
                    'parameters': params,
                })
        
        return layers



