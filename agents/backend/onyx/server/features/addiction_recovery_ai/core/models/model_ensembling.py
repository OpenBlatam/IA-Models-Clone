"""
Model Ensembling for Improved Accuracy
"""

import torch
import torch.nn as nn
from typing import List, Optional, Dict, Callable
import numpy as np
import logging

logger = logging.getLogger(__name__)


class ModelEnsemble(nn.Module):
    """Ensemble of multiple models"""
    
    def __init__(
        self,
        models: List[torch.nn.Module],
        weights: Optional[List[float]] = None,
        method: str = "mean"
    ):
        """
        Initialize model ensemble
        
        Args:
            models: List of models to ensemble
            weights: Optional weights for each model
            method: Ensemble method (mean, weighted_mean, max, voting)
        """
        super().__init__()
        self.models = nn.ModuleList(models)
        self.method = method
        
        if weights is None:
            self.weights = [1.0 / len(models)] * len(models)
        else:
            assert len(weights) == len(models), "Weights must match models"
            # Normalize weights
            total = sum(weights)
            self.weights = [w / total for w in weights]
        
        logger.info(f"ModelEnsemble initialized with {len(models)} models, method={method}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ensemble"""
        outputs = []
        for model in self.models:
            with torch.no_grad():
                output = model(x)
            outputs.append(output)
        
        # Stack outputs
        stacked = torch.stack(outputs, dim=0)
        
        if self.method == "mean":
            return stacked.mean(dim=0)
        elif self.method == "weighted_mean":
            weights_tensor = torch.tensor(self.weights, device=stacked.device)
            weights_tensor = weights_tensor.view(-1, *([1] * (stacked.dim() - 1)))
            return (stacked * weights_tensor).sum(dim=0)
        elif self.method == "max":
            return stacked.max(dim=0)[0]
        elif self.method == "voting":
            # For classification
            if stacked.dim() > 2:
                # Multi-class: use argmax then mode
                predictions = stacked.argmax(dim=-1)
                return predictions.mode(dim=0)[0]
            else:
                # Binary: majority vote
                predictions = (stacked > 0.5).long()
                return predictions.mode(dim=0)[0].float()
        else:
            raise ValueError(f"Unknown ensemble method: {method}")
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Predict with ensemble"""
        self.eval()
        with torch.no_grad():
            return self.forward(x)


class StackingEnsemble(nn.Module):
    """Stacking ensemble with meta-learner"""
    
    def __init__(
        self,
        base_models: List[torch.nn.Module],
        meta_model: Optional[torch.nn.Module] = None,
        num_base_models: Optional[int] = None
    ):
        """
        Initialize stacking ensemble
        
        Args:
            base_models: List of base models
            meta_model: Meta-learner model (if None, creates simple MLP)
            num_base_models: Number of base models (for meta-model input size)
        """
        super().__init__()
        self.base_models = nn.ModuleList(base_models)
        
        if meta_model is None:
            num_base = num_base_models or len(base_models)
            # Simple meta-learner
            self.meta_model = nn.Sequential(
                nn.Linear(num_base, 32),
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
                nn.Sigmoid()
            )
        else:
            self.meta_model = meta_model
        
        logger.info(f"StackingEnsemble initialized with {len(base_models)} base models")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        # Get predictions from base models
        base_predictions = []
        for model in self.base_models:
            with torch.no_grad():
                pred = model(x)
                if pred.dim() > 1:
                    pred = pred.squeeze()
            base_predictions.append(pred)
        
        # Stack base predictions
        base_stacked = torch.stack(base_predictions, dim=1)
        
        # Meta-model prediction
        meta_output = self.meta_model(base_stacked)
        
        return meta_output


def create_ensemble(
    models: List[torch.nn.Module],
    method: str = "mean",
    weights: Optional[List[float]] = None
) -> ModelEnsemble:
    """
    Create model ensemble
    
    Args:
        models: List of models
        method: Ensemble method
        weights: Optional weights
    
    Returns:
        Model ensemble
    """
    return ModelEnsemble(models, weights=weights, method=method)


def create_stacking_ensemble(
    base_models: List[torch.nn.Module],
    meta_model: Optional[torch.nn.Module] = None
) -> StackingEnsemble:
    """
    Create stacking ensemble
    
    Args:
        base_models: List of base models
        meta_model: Optional meta-learner
    
    Returns:
        Stacking ensemble
    """
    return StackingEnsemble(base_models, meta_model=meta_model)

