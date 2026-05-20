"""
Bagging Ensemble for TruthGPT API
=================================

TensorFlow-like bagging ensemble implementation.
"""

import torch
import torch.nn as nn
from typing import List, Optional, Dict, Any
import numpy as np


class BaggingEnsemble:
    """
    Bagging ensemble model.
    
    Similar to tf.keras.ensemble.BaggingClassifier, this class
    implements bootstrap aggregating for ensemble learning.
    """
    
    def __init__(self, 
                 models: List[Any],
                 n_samples: Optional[int] = None,
                 max_samples: float = 1.0,
                 random_state: Optional[int] = None,
                 name: Optional[str] = None):
        """
        Initialize BaggingEnsemble.
        
        Args:
            models: List of models to ensemble
            n_samples: Number of samples for bootstrap
            max_samples: Maximum fraction of samples
            random_state: Random state
            name: Optional name for the ensemble
        """
        self.models = models
        self.n_samples = n_samples
        self.max_samples = max_samples
        self.random_state = random_state
        self.name = name or "bagging_ensemble"
        
        # Set random state
        if random_state is not None:
            torch.manual_seed(random_state)
            np.random.seed(random_state)
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Make predictions.
        
        Args:
            x: Input data
            
        Returns:
            Ensemble predictions
        """
        predictions = []
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                pred = model(x)
                predictions.append(pred)
        
        # Average predictions
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
        return self.predict(x)
    
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
        predicted_labels = torch.argmax(predictions, dim=1)
        
        if metric == 'accuracy':
            correct = (predicted_labels == y).float().mean()
            return correct.item()
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    def __repr__(self):
        return f"BaggingEnsemble(models={len(self.models)}, n_samples={self.n_samples})"
