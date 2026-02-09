"""
Model Ensemble
Ensemble model implementations
"""

import torch
import torch.nn as nn
from typing import List, Optional, Dict, Any
import logging
import numpy as np

logger = logging.getLogger(__name__)


class EnsembleModel(nn.Module):
    """
    Ensemble of multiple models
    """
    
    def __init__(
        self,
        models: List[nn.Module],
        voting: str = 'soft',
        weights: Optional[List[float]] = None,
    ):
        """
        Initialize ensemble
        
        Args:
            models: List of models
            voting: Voting strategy ('soft' or 'hard')
            weights: Weights for each model (optional)
        """
        super(EnsembleModel, self).__init__()
        self.models = nn.ModuleList(models)
        self.voting = voting
        self.weights = weights or [1.0 / len(models)] * len(models)
        
        if len(self.weights) != len(models):
            raise ValueError("Number of weights must match number of models")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through ensemble
        
        Args:
            x: Input tensor
            
        Returns:
            Ensemble prediction
        """
        outputs = []
        for model in self.models:
            outputs.append(model(x))
        
        # Stack outputs
        stacked = torch.stack(outputs, dim=0)  # [num_models, batch_size, num_classes]
        
        if self.voting == 'soft':
            # Weighted average
            weights = torch.tensor(self.weights, device=stacked.device).view(-1, 1, 1)
            weighted = stacked * weights
            ensemble_output = weighted.sum(dim=0)
        else:  # hard voting
            # Majority vote
            predictions = torch.argmax(stacked, dim=2)  # [num_models, batch_size]
            ensemble_output = torch.mode(predictions, dim=0)[0]
        
        return ensemble_output
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict probabilities
        
        Args:
            x: Input tensor
            
        Returns:
            Probability predictions
        """
        self.eval()
        with torch.no_grad():
            outputs = []
            for model in self.models:
                output = model(x)
                probs = torch.softmax(output, dim=1)
                outputs.append(probs)
            
            stacked = torch.stack(outputs, dim=0)
            weights = torch.tensor(self.weights, device=stacked.device).view(-1, 1, 1)
            weighted = stacked * weights
            ensemble_probs = weighted.sum(dim=0)
        
        return ensemble_probs


class EnsembleBuilder:
    """
    Builder for creating ensembles
    """
    
    @staticmethod
    def create_ensemble(
        model_configs: List[Dict[str, Any]],
        voting: str = 'soft',
        weights: Optional[List[float]] = None,
    ) -> EnsembleModel:
        """
        Create ensemble from configurations
        
        Args:
            model_configs: List of model configurations
            voting: Voting strategy
            weights: Model weights
            
        Returns:
            Ensemble model
        """
        from ..models.mobilenet.factory import MobileNetFactory
        from ..models.mobilenet.config import MobileNetConfig
        
        models = []
        for config in model_configs:
            model_config = MobileNetConfig.from_dict(config)
            model = MobileNetFactory.create_model(model_config)
            models.append(model)
        
        return EnsembleModel(models, voting=voting, weights=weights)
    
    @staticmethod
    def create_diverse_ensemble(
        base_config: Dict[str, Any],
        num_models: int = 5,
        diversity_params: Optional[Dict[str, Any]] = None,
    ) -> EnsembleModel:
        """
        Create diverse ensemble with variations
        
        Args:
            base_config: Base model configuration
            num_models: Number of models in ensemble
            diversity_params: Parameters for diversity
            
        Returns:
            Ensemble model
        """
        import random
        
        models = []
        diversity_params = diversity_params or {}
        
        for i in range(num_models):
            config = base_config.copy()
            
            # Add diversity
            if 'dropout' in diversity_params:
                config['dropout'] = random.uniform(
                    diversity_params['dropout']['min'],
                    diversity_params['dropout']['max']
                )
            
            if 'weight_decay' in diversity_params:
                config['weight_decay'] = random.uniform(
                    diversity_params['weight_decay']['min'],
                    diversity_params['weight_decay']['max']
                )
            
            from ..models.mobilenet.factory import MobileNetFactory
            from ..models.mobilenet.config import MobileNetConfig
            
            model_config = MobileNetConfig.from_dict(config)
            model = MobileNetFactory.create_model(model_config)
            models.append(model)
        
        return EnsembleModel(models, voting='soft')



