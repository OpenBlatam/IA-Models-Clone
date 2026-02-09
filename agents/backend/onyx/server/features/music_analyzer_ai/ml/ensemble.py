"""
Ensemble Models
Combine multiple models for better predictions
"""

from typing import List, Dict, Any, Optional, Callable
import logging
import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class EnsembleModel:
    """
    Ensemble of multiple models with different aggregation strategies
    """
    
    def __init__(
        self,
        models: List[nn.Module],
        weights: Optional[List[float]] = None,
        aggregation: str = "weighted_average"  # "average", "weighted_average", "voting", "stacking"
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for ensemble models")
        
        self.models = models
        self.aggregation = aggregation
        
        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        
        if len(weights) != len(models):
            raise ValueError("Number of weights must match number of models")
        
        self.weights = weights
        
        # Set all models to eval mode
        for model in self.models:
            model.eval()
    
    def predict(self, input_data: Any) -> Any:
        """Make ensemble prediction"""
        predictions = []
        
        for model in self.models:
            with torch.no_grad():
                if isinstance(input_data, torch.Tensor):
                    pred = model(input_data)
                else:
                    input_tensor = torch.tensor(input_data)
                    pred = model(input_tensor)
                predictions.append(pred)
        
        # Aggregate predictions
        if self.aggregation == "average":
            return self._average(predictions)
        elif self.aggregation == "weighted_average":
            return self._weighted_average(predictions)
        elif self.aggregation == "voting":
            return self._voting(predictions)
        elif self.aggregation == "stacking":
            return self._stacking(predictions)
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation}")
    
    def _average(self, predictions: List[torch.Tensor]) -> torch.Tensor:
        """Average predictions"""
        stacked = torch.stack(predictions)
        return stacked.mean(dim=0)
    
    def _weighted_average(self, predictions: List[torch.Tensor]) -> torch.Tensor:
        """Weighted average predictions"""
        weighted_sum = sum(w * p for w, p in zip(self.weights, predictions))
        total_weight = sum(self.weights)
        return weighted_sum / total_weight
    
    def _voting(self, predictions: List[torch.Tensor]) -> torch.Tensor:
        """Voting aggregation (for classification)"""
        # Get class predictions
        class_preds = [torch.argmax(p, dim=-1) for p in predictions]
        stacked = torch.stack(class_preds)
        
        # Majority vote
        mode, _ = torch.mode(stacked, dim=0)
        return mode
    
    def _stacking(self, predictions: List[torch.Tensor]) -> torch.Tensor:
        """Stacking aggregation (requires meta-learner)"""
        # Simple stacking: average for now
        # In practice, would use a meta-learner
        return self._average(predictions)


class ModelEnsembleBuilder:
    """
    Builder for creating ensemble models
    """
    
    @staticmethod
    def create_diverse_ensemble(
        model_factories: List[Callable],
        weights: Optional[List[float]] = None
    ) -> EnsembleModel:
        """Create ensemble from diverse model architectures"""
        models = [factory() for factory in model_factories]
        return EnsembleModel(models, weights=weights)
    
    @staticmethod
    def create_bagging_ensemble(
        model_factory: Callable,
        n_models: int = 5,
        bootstrap: bool = True
    ) -> EnsembleModel:
        """Create bagging ensemble"""
        models = []
        for _ in range(n_models):
            model = model_factory()
            models.append(model)
        
        return EnsembleModel(models, weights=None, aggregation="average")

