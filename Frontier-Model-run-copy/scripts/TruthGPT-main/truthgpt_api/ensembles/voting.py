"""
Voting Ensemble for TruthGPT API
================================

TensorFlow-like voting ensemble implementation.
"""

import torch
import torch.nn as nn
from typing import List, Optional, Dict, Any, Tuple
import numpy as np


class VotingEnsemble:
    """
    Voting ensemble model.
    
    Similar to tf.keras.ensemble.VotingClassifier, this class
    implements voting ensemble for multiple models.
    """
    
    def __init__(self, 
                 models: List[Any],
                 weights: Optional[List[float]] = None,
                 voting: str = 'hard',
                 name: Optional[str] = None):
        """
        Initialize VotingEnsemble.
        
        Args:
            models: List of models to ensemble
            weights: Weights for each model
            voting: Type of voting ('hard' or 'soft')
            name: Optional name for the ensemble
        """
        self.models = models
        self.weights = weights or [1.0] * len(models)
        self.voting = voting
        self.name = name or "voting_ensemble"
        
        if len(self.models) != len(self.weights):
            raise ValueError("Number of models and weights must match")
        
        if len(self.models) == 0:
            raise ValueError("At least one model is required")
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Make predictions.
        
        Args:
            x: Input data
            
        Returns:
            Ensemble predictions
        """
        predictions = []
        
        for model, weight in zip(self.models, self.weights):
            model.eval()
            with torch.no_grad():
                pred = model(x)
                
                if self.voting == 'hard':
                    pred = torch.argmax(pred, dim=1)
                
                predictions.append(pred * weight)
        
        # Combine predictions
        if self.voting == 'hard':
            # Hard voting: majority vote
            stacked = torch.stack(predictions, dim=0)
            ensemble_pred = torch.mode(stacked, dim=0)[0]
        else:
            # Soft voting: weighted average
            stacked = torch.stack(predictions, dim=0)
            ensemble_pred = torch.mean(stacked, dim=0)
        
        return ensemble_pred
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Make probability predictions.
        
        Args:
            x: Input data
            
        Returns:
            Ensemble probability predictions
        """
        predictions = []
        
        for model, weight in zip(self.models, self.weights):
            model.eval()
            with torch.no_grad():
                pred = model(x)
                predictions.append(pred * weight)
        
        # Weighted average
        stacked = torch.stack(predictions, dim=0)
        ensemble_pred = torch.mean(stacked, dim=0)
        
        return ensemble_pred
    
    def evaluate(self, 
                 x: torch.Tensor, 
                 y: torch.Tensor,
                 metric: str = 'accuracy') -> float:
        """
        Evaluate ensemble.
        
        Args:
            x: Input data
            y: True labels
            metric: Metric to compute
            
        Returns:
            Metric value
        """
        predictions = self.predict(x)
        
        if metric == 'accuracy':
            correct = (predictions == y).float().mean()
            return correct.item()
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    def __repr__(self):
        return f"VotingEnsemble(models={len(self.models)}, voting={self.voting})"
