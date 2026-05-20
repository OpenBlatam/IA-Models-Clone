"""
Model Comparison
Compare models and results
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


class ModelComparator:
    """
    Compare models and their performance
    """
    
    @staticmethod
    def compare_models(
        models: Dict[str, nn.Module],
        input_shape: tuple,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compare multiple models
        
        Args:
            models: Dictionary of model_name -> model
            input_shape: Input shape for testing
            
        Returns:
            Dictionary with comparison results
        """
        results = {}
        
        for name, model in models.items():
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            # Test forward pass
            try:
                model.eval()
                dummy_input = torch.randn(input_shape)
                with torch.no_grad():
                    output = model(dummy_input)
                
                forward_success = True
                output_shape = list(output.shape)
            except Exception as e:
                forward_success = False
                output_shape = None
                logger.warning(f"Forward pass failed for {name}: {e}")
            
            results[name] = {
                'total_parameters': total_params,
                'trainable_parameters': trainable_params,
                'forward_success': forward_success,
                'output_shape': output_shape,
            }
        
        return results
    
    @staticmethod
    def compare_results(
        results: Dict[str, Dict[str, float]],
    ) -> Dict[str, Any]:
        """
        Compare results from different runs
        
        Args:
            results: Dictionary of run_name -> metrics
            
        Returns:
            Dictionary with comparison
        """
        if not results:
            return {}
        
        # Get all metric names
        all_metrics = set()
        for run_results in results.values():
            all_metrics.update(run_results.keys())
        
        comparison = {}
        for metric in all_metrics:
            values = [run_results.get(metric, None) for run_results in results.values()]
            values = [v for v in values if v is not None]
            
            if values:
                comparison[metric] = {
                    'values': values,
                    'mean': sum(values) / len(values),
                    'min': min(values),
                    'max': max(values),
                    'std': (sum((x - sum(values)/len(values))**2 for x in values) / len(values))**0.5 if len(values) > 1 else 0.0,
                }
        
        return comparison



