"""
Model Analyzer
==============

Analyzes model architecture and properties.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional
from .parameter_counter import ParameterCounter


class ModelAnalyzer:
    """Analyzes model architecture and properties."""
    
    @staticmethod
    def get_model_info(
        model: nn.Module,
        include_architecture: bool = True
    ) -> Dict[str, Any]:
        """
        Get comprehensive model information.
        
        Args:
            model: PyTorch model
            include_architecture: Include architecture details
            
        Returns:
            Dictionary with model information
        """
        stats = ParameterCounter.get_parameter_stats(model)
        
        info = {
            "total_parameters": stats["total"],
            "trainable_parameters": stats["trainable"],
            "non_trainable_parameters": stats["non_trainable"],
            "trainable_percentage": stats["trainable_percentage"],
            "parameter_mb": stats["total"] * 4 / (1024 * 1024),  # Assuming float32
        }
        
        if include_architecture:
            layers = ModelAnalyzer.get_layer_info(model)
            info["architecture"] = {
                "num_layers": len(layers),
                "layers": layers,
                "parameters_by_layer": stats["by_layer"],
            }
        
        return info
    
    @staticmethod
    def get_layer_info(model: nn.Module) -> List[Dict[str, Any]]:
        """
        Get information about each layer.
        
        Args:
            model: PyTorch model
            
        Returns:
            List of layer information dictionaries
        """
        layers_info = []
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf module
                params = sum(p.numel() for p in module.parameters())
                trainable = any(p.requires_grad for p in module.parameters())
                
                layer_info = {
                    "name": name,
                    "type": type(module).__name__,
                    "parameters": params,
                    "trainable": trainable,
                }
                layers_info.append(layer_info)
        return layers_info
    
    @staticmethod
    def export_model_summary(
        model: nn.Module,
        output_path: Optional[str] = None
    ) -> str:
        """
        Export model summary to string or file.
        
        Args:
            model: PyTorch model
            output_path: Optional file path to save summary
            
        Returns:
            Model summary string
        """
        info = ModelAnalyzer.get_model_info(model, include_architecture=True)
        layers = info["architecture"]["layers"]
        
        summary_lines = [
            "=" * 60,
            "Model Summary",
            "=" * 60,
            f"Total Parameters: {info['total_parameters']:,}",
            f"Trainable Parameters: {info['trainable_parameters']:,}",
            f"Non-trainable Parameters: {info['non_trainable_parameters']:,}",
            f"Trainable Percentage: {info['trainable_percentage']:.2f}%",
            f"Model Size: {info['parameter_mb']:.2f} MB",
            "",
            "Layers:",
            "-" * 60,
        ]
        
        for layer in layers:
            summary_lines.append(
                f"{layer['name']:40s} | {layer['type']:20s} | "
                f"Params: {layer['parameters']:>10,} | Trainable: {layer['trainable']}"
            )
        
        summary = "\n".join(summary_lines)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(summary)
        
        return summary
    
    @staticmethod
    def compare_models(
        model1: nn.Module,
        model2: nn.Module
    ) -> Dict[str, Any]:
        """
        Compare two models.
        
        Args:
            model1: First model
            model2: Second model
            
        Returns:
            Dictionary with comparison results
        """
        info1 = ModelAnalyzer.get_model_info(model1)
        info2 = ModelAnalyzer.get_model_info(model2)
        
        # Compare parameters
        param_diff = abs(info1["total_parameters"] - info2["total_parameters"])
        
        # Compare state dicts
        state1 = model1.state_dict()
        state2 = model2.state_dict()
        
        keys_match = set(state1.keys()) == set(state2.keys())
        values_match = True
        if keys_match:
            for key in state1.keys():
                if not torch.equal(state1[key], state2[key]):
                    values_match = False
                    break
        
        return {
            "model1_info": info1,
            "model2_info": info2,
            "parameter_difference": param_diff,
            "keys_match": keys_match,
            "values_match": values_match,
            "are_identical": keys_match and values_match,
        }

