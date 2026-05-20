"""
Model Analyzer
Analyze model architecture and properties
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class ModelAnalyzer:
    """
    Analyze model architecture and properties
    """
    
    def __init__(self, model: nn.Module):
        """
        Initialize analyzer
        
        Args:
            model: Model to analyze
        """
        self.model = model
    
    def get_architecture_info(self) -> Dict[str, Any]:
        """
        Get architecture information
        
        Returns:
            Dictionary with architecture info
        """
        info = {
            'model_name': self.model.__class__.__name__,
            'total_parameters': sum(p.numel() for p in self.model.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            'layers': [],
        }
        
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # Leaf node
                layer_info = {
                    'name': name,
                    'type': module.__class__.__name__,
                    'parameters': sum(p.numel() for p in module.parameters()),
                }
                info['layers'].append(layer_info)
        
        return info
    
    def get_parameter_distribution(self) -> Dict[str, Any]:
        """
        Get parameter distribution statistics
        
        Returns:
            Dictionary with parameter statistics
        """
        all_params = torch.cat([p.flatten() for p in self.model.parameters()])
        
        return {
            'mean': float(all_params.mean().item()),
            'std': float(all_params.std().item()),
            'min': float(all_params.min().item()),
            'max': float(all_params.max().item()),
            'median': float(all_params.median().item()),
        }
    
    def get_layer_sizes(self) -> List[Dict[str, Any]]:
        """
        Get sizes of all layers
        
        Returns:
            List of layer size information
        """
        layer_sizes = []
        
        for name, param in self.model.named_parameters():
            layer_sizes.append({
                'name': name,
                'shape': list(param.shape),
                'size': param.numel(),
                'dtype': str(param.dtype),
            })
        
        return layer_sizes
    
    def analyze_complexity(self) -> Dict[str, Any]:
        """
        Analyze model complexity
        
        Returns:
            Dictionary with complexity metrics
        """
        info = self.get_architecture_info()
        
        # Count layer types
        layer_types = {}
        for layer in info['layers']:
            layer_type = layer['type']
            layer_types[layer_type] = layer_types.get(layer_type, 0) + 1
        
        return {
            'total_parameters': info['total_parameters'],
            'trainable_parameters': info['trainable_parameters'],
            'num_layers': len(info['layers']),
            'layer_types': layer_types,
            'parameters_per_layer': info['total_parameters'] / len(info['layers']) if info['layers'] else 0,
        }
    
    def print_summary(self) -> None:
        """Print model summary"""
        info = self.get_architecture_info()
        complexity = self.analyze_complexity()
        
        print(f"\n{'='*60}")
        print(f"Model: {info['model_name']}")
        print(f"{'='*60}")
        print(f"Total Parameters: {info['total_parameters']:,}")
        print(f"Trainable Parameters: {info['trainable_parameters']:,}")
        print(f"Non-trainable Parameters: {info['total_parameters'] - info['trainable_parameters']:,}")
        print(f"Number of Layers: {complexity['num_layers']}")
        print(f"\nLayer Types:")
        for layer_type, count in complexity['layer_types'].items():
            print(f"  {layer_type}: {count}")
        print(f"{'='*60}\n")



