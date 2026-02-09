"""
Model Comparison

Utilities for comparing models.
"""

import logging
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)


class ModelComparator:
    """Compare models."""
    
    def __init__(self):
        """Initialize model comparator."""
        pass
    
    def compare_performance(
        self,
        models: Dict[str, nn.Module],
        test_loader: torch.utils.data.DataLoader,
        metrics: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare model performance.
        
        Args:
            models: Dictionary of model names and models
            test_loader: Test data loader
            metrics: List of metrics to compute
            
        Returns:
            Performance comparison
        """
        if metrics is None:
            metrics = ['loss', 'accuracy']
        
        results = {}
        
        for name, model in models.items():
            model.eval()
            model_metrics = defaultdict(list)
            
            with torch.no_grad():
                for batch in test_loader:
                    inputs = batch['input'] if isinstance(batch, dict) else batch[0]
                    targets = batch.get('target') if isinstance(batch, dict) else (batch[1] if len(batch) > 1 else None)
                    
                    outputs = model(inputs)
                    
                    # Compute metrics
                    if 'loss' in metrics:
                        loss = nn.functional.mse_loss(outputs, targets) if targets is not None else 0.0
                        model_metrics['loss'].append(loss.item())
                    
                    if 'accuracy' in metrics and targets is not None:
                        preds = outputs.argmax(dim=-1) if outputs.dim() > 1 else outputs
                        acc = (preds == targets).float().mean()
                        model_metrics['accuracy'].append(acc.item())
            
            # Average metrics
            results[name] = {
                metric: sum(model_metrics[metric]) / len(model_metrics[metric])
                for metric in metrics
                if model_metrics[metric]
            }
        
        return results
    
    def compare_architectures(
        self,
        models: Dict[str, nn.Module]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compare model architectures.
        
        Args:
            models: Dictionary of model names and models
            
        Returns:
            Architecture comparison
        """
        from core.debugging import count_parameters
        
        results = {}
        
        for name, model in models.items():
            param_counts = count_parameters(model)
            
            results[name] = {
                'total_params': param_counts['total'],
                'trainable_params': param_counts['trainable'],
                'num_layers': len(list(model.modules())),
                'model_size_mb': param_counts['total'] * 4 / (1024 ** 2)  # Assuming float32
            }
        
        return results
    
    def compare_inference_time(
        self,
        models: Dict[str, nn.Module],
        input_shape: tuple,
        num_iterations: int = 10
    ) -> Dict[str, float]:
        """
        Compare inference time.
        
        Args:
            models: Dictionary of model names and models
            input_shape: Input tensor shape
            num_iterations: Number of iterations
            
        Returns:
            Inference time comparison
        """
        from core.profiling import MemoryProfiler
        
        results = {}
        
        for name, model in models.items():
            perf_metrics = MemoryProfiler.profile_model_memory(model, input_shape)
            results[name] = perf_metrics.get('avg_inference_time_s', 0.0)
        
        return results


def compare_models(
    models: Dict[str, nn.Module],
    test_loader: torch.utils.data.DataLoader,
    **kwargs
) -> Dict[str, Dict[str, float]]:
    """Compare models."""
    comparator = ModelComparator()
    return comparator.compare_performance(models, test_loader, **kwargs)


def compare_performance(
    models: Dict[str, nn.Module],
    test_loader: torch.utils.data.DataLoader,
    **kwargs
) -> Dict[str, Dict[str, float]]:
    """Compare model performance."""
    comparator = ModelComparator()
    return comparator.compare_performance(models, test_loader, **kwargs)


def compare_architectures(
    models: Dict[str, nn.Module]
) -> Dict[str, Dict[str, Any]]:
    """Compare model architectures."""
    comparator = ModelComparator()
    return comparator.compare_architectures(models)



