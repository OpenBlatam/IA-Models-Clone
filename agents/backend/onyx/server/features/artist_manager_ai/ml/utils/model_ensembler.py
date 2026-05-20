"""
Model Ensembler
===============

Utilities for model ensembling.
"""

import torch
import torch.nn as nn
import logging
from typing import List, Dict, Any, Optional, Callable
import numpy as np

logger = logging.getLogger(__name__)


class ModelEnsembler:
    """
    Model ensembler for combining multiple models.
    
    Features:
    - Weighted averaging
    - Voting
    - Stacking
    """
    
    def __init__(self, models: List[nn.Module], weights: Optional[List[float]] = None):
        """
        Initialize ensembler.
        
        Args:
            models: List of models
            weights: Optional weights for each model
        """
        self.models = models
        self.weights = weights or [1.0 / len(models)] * len(models)
        
        if len(self.weights) != len(models):
            raise ValueError("Number of weights must match number of models")
        
        self._logger = logger
    
    def predict_average(
        self,
        inputs: torch.Tensor,
        device: torch.device
    ) -> torch.Tensor:
        """
        Average predictions from all models.
        
        Args:
            inputs: Input tensor
            device: Device
        
        Returns:
            Averaged predictions
        """
        predictions = []
        
        for model, weight in zip(self.models, self.weights):
            model = model.to(device)
            model.eval()
            
            with torch.no_grad():
                pred = model(inputs.to(device))
                predictions.append(pred.cpu() * weight)
        
        # Weighted average
        ensemble_pred = torch.stack(predictions).sum(dim=0) / sum(self.weights)
        
        return ensemble_pred
    
    def predict_vote(
        self,
        inputs: torch.Tensor,
        device: torch.device,
        threshold: float = 0.5
    ) -> torch.Tensor:
        """
        Voting ensemble (for classification).
        
        Args:
            inputs: Input tensor
            device: Device
            threshold: Classification threshold
        
        Returns:
            Voted predictions
        """
        predictions = []
        
        for model in self.models:
            model = model.to(device)
            model.eval()
            
            with torch.no_grad():
                pred = model(inputs.to(device))
                pred_binary = (pred >= threshold).float()
                predictions.append(pred_binary.cpu())
        
        # Majority vote
        votes = torch.stack(predictions)
        ensemble_pred = (votes.sum(dim=0) > len(self.models) / 2).float()
        
        return ensemble_pred
    
    def predict_stack(
        self,
        inputs: torch.Tensor,
        meta_model: Optional[nn.Module],
        device: torch.device
    ) -> torch.Tensor:
        """
        Stacking ensemble.
        
        Args:
            inputs: Input tensor
            meta_model: Meta-learner model
            device: Device
        
        Returns:
            Stacked predictions
        """
        # Get base predictions
        base_predictions = []
        
        for model in self.models:
            model = model.to(device)
            model.eval()
            
            with torch.no_grad():
                pred = model(inputs.to(device))
                base_predictions.append(pred.cpu())
        
        # Stack predictions
        stacked = torch.cat(base_predictions, dim=1)
        
        # Meta-model prediction
        if meta_model:
            meta_model = meta_model.to(device)
            meta_model.eval()
            
            with torch.no_grad():
                final_pred = meta_model(stacked.to(device))
            return final_pred.cpu()
        
        # Simple average if no meta-model
        return stacked.mean(dim=1, keepdim=True)




