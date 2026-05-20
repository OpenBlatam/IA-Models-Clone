"""
Feature Importance Analysis
Analyze feature importance in models
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


class FeatureImportanceAnalyzer:
    """
    Analyze feature importance in models
    """
    
    def __init__(self, model: nn.Module):
        """
        Initialize analyzer
        
        Args:
            model: Model to analyze
        """
        self.model = model
    
    def compute_gradient_importance(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        criterion: nn.Module,
    ) -> Dict[str, np.ndarray]:
        """
        Compute feature importance using gradients
        
        Args:
            inputs: Input tensor
            targets: Target labels
            criterion: Loss function
            
        Returns:
            Dictionary with importance scores
        """
        self.model.train()
        inputs.requires_grad = True
        
        # Forward and backward
        outputs = self.model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        
        # Compute importance as gradient magnitude
        importance = torch.abs(inputs.grad)
        importance = importance.mean(dim=0)  # Average over batch
        
        return {
            'importance': importance.detach().cpu().numpy(),
            'gradient_magnitude': importance.detach().cpu().numpy(),
        }
    
    def compute_permutation_importance(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        criterion: nn.Module,
        n_permutations: int = 10,
    ) -> Dict[str, np.ndarray]:
        """
        Compute permutation importance
        
        Args:
            inputs: Input tensor
            targets: Target labels
            criterion: Loss function
            n_permutations: Number of permutations
            
        Returns:
            Dictionary with importance scores
        """
        self.model.eval()
        
        # Baseline loss
        with torch.no_grad():
            baseline_outputs = self.model(inputs)
            baseline_loss = criterion(baseline_outputs, targets).item()
        
        # Permute features and compute importance
        importance_scores = []
        
        for feature_idx in range(inputs.shape[1]):
            feature_importance = []
            
            for _ in range(n_permutations):
                # Permute feature
                permuted_inputs = inputs.clone()
                permuted_inputs[:, feature_idx] = permuted_inputs[:, feature_idx][torch.randperm(inputs.size(0))]
                
                # Compute loss
                with torch.no_grad():
                    permuted_outputs = self.model(permuted_inputs)
                    permuted_loss = criterion(permuted_outputs, targets).item()
                
                # Importance = increase in loss
                feature_importance.append(permuted_loss - baseline_loss)
            
            importance_scores.append(np.mean(feature_importance))
        
        return {
            'importance': np.array(importance_scores),
            'baseline_loss': baseline_loss,
        }
    
    def analyze_layer_importance(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        criterion: nn.Module,
    ) -> Dict[str, Any]:
        """
        Analyze importance of different layers
        
        Args:
            inputs: Input tensor
            targets: Target labels
            criterion: Loss function
            
        Returns:
            Dictionary with layer importance
        """
        self.model.train()
        layer_importance = {}
        
        # Register hooks for all layers
        activations = {}
        gradients = {}
        
        def forward_hook(name):
            def hook(module, input, output):
                activations[name] = output
            return hook
        
        def backward_hook(name):
            def hook(module, grad_input, grad_output):
                if grad_output[0] is not None:
                    gradients[name] = grad_output[0]
            return hook
        
        hooks = []
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # Leaf node
                hooks.append(module.register_forward_hook(forward_hook(name)))
                hooks.append(module.register_backward_hook(backward_hook(name)))
        
        # Forward and backward
        inputs.requires_grad = True
        outputs = self.model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        
        # Compute importance for each layer
        for name in activations:
            if name in gradients:
                grad = gradients[name]
                act = activations[name]
                
                if grad is not None and act is not None:
                    importance = torch.abs(grad * act).mean().item()
                    layer_importance[name] = importance
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        return layer_importance



