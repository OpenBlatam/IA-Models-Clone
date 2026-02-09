"""
Ensembling avanzado con múltiples estrategias
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)


class StackingEnsemble:
    """Stacking ensemble con meta-learner"""
    
    def __init__(
        self,
        base_models: List[nn.Module],
        meta_learner: Optional[nn.Module] = None
    ):
        self.base_models = base_models
        self.meta_learner = meta_learner
        
        if meta_learner is None:
            # Meta-learner por defecto
            self.meta_learner = nn.Sequential(
                nn.Linear(len(base_models), 64),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(64, 1)
            )
    
    def predict(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Predicción con stacking"""
        # Predicciones de base models
        base_predictions = []
        
        for model in self.base_models:
            model.eval()
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                base_predictions.append(logits)
        
        # Stack predictions
        stacked = torch.stack(base_predictions, dim=1)
        
        # Meta-learner
        if self.meta_learner:
            meta_input = stacked.view(stacked.size(0), -1)
            final_prediction = self.meta_learner(meta_input)
        else:
            final_prediction = stacked.mean(dim=1)
        
        return final_prediction


class BlendingEnsemble:
    """Blending ensemble con pesos aprendidos"""
    
    def __init__(
        self,
        models: List[nn.Module],
        learn_weights: bool = True
    ):
        self.models = models
        self.learn_weights = learn_weights
        
        if learn_weights:
            # Pesos aprendibles
            self.weights = nn.Parameter(torch.ones(len(models)) / len(models))
        else:
            # Pesos uniformes
            self.weights = torch.ones(len(models)) / len(models)
    
    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass"""
        predictions = []
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                predictions.append(logits)
        
        # Normalizar pesos
        if self.learn_weights:
            weights = F.softmax(self.weights, dim=0)
        else:
            weights = self.weights
        
        # Promedio ponderado
        ensemble_pred = sum(w * pred for w, pred in zip(weights, predictions))
        
        return ensemble_pred


class DynamicEnsemble:
    """Ensemble dinámico que selecciona modelos según input"""
    
    def __init__(
        self,
        models: List[nn.Module],
        selector: Optional[nn.Module] = None
    ):
        self.models = models
        self.selector = selector
        
        if selector is None:
            # Selector por defecto
            input_dim = 768  # Ajustar según necesidad
            self.selector = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Linear(128, len(models)),
                nn.Softmax(dim=-1)
            )
    
    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward con selección dinámica"""
        # Obtener features para selector
        if "input_ids" in inputs:
            # Usar embeddings
            features = inputs["input_ids"].float().mean(dim=1)
        else:
            features = list(inputs.values())[0].mean(dim=1)
        
        # Seleccionar modelos
        model_weights = self.selector(features)
        
        # Predicciones
        predictions = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                predictions.append(logits)
        
        # Promedio ponderado dinámico
        predictions_tensor = torch.stack(predictions, dim=0)
        ensemble_pred = torch.sum(
            model_weights.unsqueeze(-1) * predictions_tensor,
            dim=0
        )
        
        return ensemble_pred




