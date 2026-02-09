"""
Model Ensembling - Ensamblaje de modelos
=========================================
Voting, stacking, y blending de modelos
"""

import logging
import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Any, Optional, Callable
from collections import defaultdict

logger = logging.getLogger(__name__)


class ModelEnsemble:
    """Ensamblaje de modelos"""
    
    def __init__(self, models: List[nn.Module], weights: Optional[List[float]] = None):
        self.models = models
        self.weights = weights or [1.0 / len(models)] * len(models)
        
        if len(self.weights) != len(self.models):
            raise ValueError("Number of weights must match number of models")
        
        # Normalizar pesos
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Predicción del ensemble"""
        predictions = []
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                pred = model(x)
                predictions.append(pred)
        
        # Weighted average
        ensemble_pred = sum(w * pred for w, pred in zip(self.weights, predictions))
        return ensemble_pred
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Probabilidades del ensemble"""
        predictions = self.predict(x)
        return torch.softmax(predictions, dim=-1)


class VotingEnsemble:
    """Ensemble por votación"""
    
    def __init__(self, models: List[nn.Module], voting: str = "hard"):
        self.models = models
        self.voting = voting  # "hard" or "soft"
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Predicción por votación"""
        all_predictions = []
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                pred = model(x)
                if self.voting == "hard":
                    pred = torch.argmax(pred, dim=-1)
                all_predictions.append(pred)
        
        if self.voting == "hard":
            # Majority voting
            stacked = torch.stack(all_predictions, dim=0)
            ensemble_pred, _ = torch.mode(stacked, dim=0)
            return ensemble_pred
        else:
            # Soft voting - average probabilities
            stacked = torch.stack(all_predictions, dim=0)
            ensemble_pred = torch.mean(stacked, dim=0)
            return ensemble_pred


class StackingEnsemble:
    """Stacking ensemble con meta-learner"""
    
    def __init__(
        self,
        base_models: List[nn.Module],
        meta_model: Optional[nn.Module] = None,
        meta_input_dim: Optional[int] = None
    ):
        self.base_models = base_models
        self.meta_model = meta_model
        
        if meta_model is None and meta_input_dim:
            # Crear meta-modelo por defecto
            self.meta_model = nn.Sequential(
                nn.Linear(meta_input_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 1)
            )
    
    def fit_meta_model(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_val: torch.Tensor,
        y_val: torch.Tensor,
        epochs: int = 10
    ):
        """Entrena meta-modelo"""
        # Generar predicciones de base models
        base_predictions_train = self._get_base_predictions(X_train)
        base_predictions_val = self._get_base_predictions(X_val)
        
        # Entrenar meta-modelo
        optimizer = torch.optim.Adam(self.meta_model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        
        for epoch in range(epochs):
            self.meta_model.train()
            optimizer.zero_grad()
            
            meta_pred = self.meta_model(base_predictions_train)
            loss = criterion(meta_pred.squeeze(), y_train)
            
            loss.backward()
            optimizer.step()
            
            # Validación
            self.meta_model.eval()
            with torch.no_grad():
                val_pred = self.meta_model(base_predictions_val)
                val_loss = criterion(val_pred.squeeze(), y_val)
            
            logger.info(f"Meta-model epoch {epoch+1}/{epochs}: train_loss={loss.item():.4f}, val_loss={val_loss.item():.4f}")
    
    def _get_base_predictions(self, x: torch.Tensor) -> torch.Tensor:
        """Obtiene predicciones de base models"""
        predictions = []
        
        for model in self.base_models:
            model.eval()
            with torch.no_grad():
                pred = model(x)
                predictions.append(pred)
        
        return torch.cat(predictions, dim=1)
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Predicción del stacking ensemble"""
        base_predictions = self._get_base_predictions(x)
        
        self.meta_model.eval()
        with torch.no_grad():
            meta_pred = self.meta_model(base_predictions)
        
        return meta_pred.squeeze()


class BlendingEnsemble:
    """Blending ensemble con split de datos"""
    
    def __init__(self, models: List[nn.Module], blend_ratio: float = 0.2):
        self.models = models
        self.blend_ratio = blend_ratio
        self.blend_weights: Optional[List[float]] = None
    
    def fit_blend_weights(
        self,
        X_blend: torch.Tensor,
        y_blend: torch.Tensor
    ):
        """Ajusta pesos de blending"""
        predictions = []
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                pred = model(X_blend)
                predictions.append(pred)
        
        # Optimizar pesos
        predictions_tensor = torch.stack(predictions, dim=0)  # [n_models, batch, n_classes]
        
        # Resolver sistema lineal para encontrar pesos óptimos
        # Simplificado: usar promedio ponderado
        self.blend_weights = [1.0 / len(self.models)] * len(self.models)
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Predicción del blending ensemble"""
        if self.blend_weights is None:
            self.blend_weights = [1.0 / len(self.models)] * len(self.models)
        
        predictions = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                pred = model(x)
                predictions.append(pred)
        
        # Weighted blend
        ensemble_pred = sum(w * pred for w, pred in zip(self.blend_weights, predictions))
        return ensemble_pred




