"""
Model ensembling para mejor rendimiento
"""

import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)


class ModelEnsemble:
    """Ensemble de modelos"""
    
    def __init__(
        self,
        models: List[nn.Module],
        weights: Optional[List[float]] = None,
        voting: str = "soft"  # "soft" or "hard"
    ):
        self.models = models
        self.weights = weights or [1.0 / len(models)] * len(models)
        self.voting = voting
        
        # Normalizar pesos
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]
    
    def predict(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Predicción del ensemble
        
        Args:
            inputs: Inputs del modelo
            
        Returns:
            Predicciones del ensemble
        """
        predictions = []
        
        for model, weight in zip(self.models, self.weights):
            model.eval()
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                predictions.append(logits * weight)
        
        # Promedio ponderado
        ensemble_pred = torch.stack(predictions).sum(dim=0)
        
        return ensemble_pred
    
    def predict_proba(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Predicciones como probabilidades"""
        logits = self.predict(inputs)
        return torch.softmax(logits, dim=-1)


class WeightedEnsemble:
    """Ensemble con pesos aprendidos"""
    
    def __init__(self, models: List[nn.Module]):
        self.models = models
        # Pesos aprendibles
        self.ensemble_weights = nn.Parameter(torch.ones(len(models)) / len(models))
    
    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass del ensemble"""
        predictions = []
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                predictions.append(logits)
        
        # Normalizar pesos
        weights = torch.softmax(self.ensemble_weights, dim=0)
        
        # Promedio ponderado
        ensemble_pred = sum(w * pred for w, pred in zip(weights, predictions))
        
        return ensemble_pred




