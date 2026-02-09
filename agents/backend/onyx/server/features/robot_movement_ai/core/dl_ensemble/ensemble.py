"""
Model Ensemble - Modular Ensemble Methods
==========================================

Métodos de ensemble modulares para modelos.
"""

import logging
from typing import Dict, Any, Optional, List, Callable
import torch
import torch.nn as nn
import numpy as np

logger = logging.getLogger(__name__)


class EnsembleModel(nn.Module):
    """
    Modelo ensemble modular.
    
    Combina múltiples modelos usando diferentes estrategias.
    """
    
    def __init__(
        self,
        models: List[nn.Module],
        strategy: str = 'average',
        weights: Optional[List[float]] = None
    ):
        """
        Inicializar ensemble.
        
        Args:
            models: Lista de modelos
            strategy: Estrategia ('average', 'weighted', 'voting', 'stacking')
            weights: Pesos para weighted average (opcional)
        """
        super().__init__()
        self.models = nn.ModuleList(models)
        self.strategy = strategy
        
        if weights is None:
            self.weights = [1.0 / len(models)] * len(models)
        else:
            if len(weights) != len(models):
                raise ValueError("Number of weights must match number of models")
            # Normalizar pesos
            total = sum(weights)
            self.weights = [w / total for w in weights]
        
        logger.info(f"Ensemble created with {len(models)} models, strategy: {strategy}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass del ensemble.
        
        Args:
            x: Tensor de entrada
            
        Returns:
            Predicción del ensemble
        """
        predictions = []
        
        for model in self.models:
            with torch.no_grad() if self.strategy != 'stacking' else torch.enable_grad():
                pred = model(x)
                predictions.append(pred)
        
        if self.strategy == 'average':
            return torch.stack(predictions).mean(dim=0)
        
        elif self.strategy == 'weighted':
            weighted_preds = [pred * w for pred, w in zip(predictions, self.weights)]
            return torch.stack(weighted_preds).sum(dim=0)
        
        elif self.strategy == 'voting':
            # Para clasificación: majority voting
            # Para regresión: promedio
            return torch.stack(predictions).mean(dim=0)
        
        elif self.strategy == 'stacking':
            # Stacking requiere meta-modelo (implementación simplificada)
            return torch.stack(predictions).mean(dim=0)
        
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
    
    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        num_samples: int = 10
    ) -> Dict[str, torch.Tensor]:
        """
        Predecir con estimación de incertidumbre.
        
        Args:
            x: Tensor de entrada
            num_samples: Número de muestras para Monte Carlo
            
        Returns:
            Diccionario con predicción y incertidumbre
        """
        all_predictions = []
        
        for model in self.models:
            model.eval()
            model_preds = []
            
            for _ in range(num_samples):
                with torch.no_grad():
                    pred = model(x)
                    model_preds.append(pred)
            
            all_predictions.append(torch.stack(model_preds))
        
        # Promedio sobre modelos
        ensemble_preds = torch.stack([p.mean(dim=0) for p in all_predictions])
        mean_pred = ensemble_preds.mean(dim=0)
        
        # Calcular incertidumbre (varianza)
        uncertainty = ensemble_preds.var(dim=0)
        
        return {
            'prediction': mean_pred,
            'uncertainty': uncertainty,
            'std': torch.sqrt(uncertainty)
        }


class EnsembleBuilder:
    """Builder para ensembles."""
    
    def __init__(self):
        """Inicializar builder."""
        self.models = []
        self.strategy = 'average'
        self.weights = None
    
    def add_model(self, model: nn.Module, weight: Optional[float] = None) -> 'EnsembleBuilder':
        """
        Agregar modelo al ensemble.
        
        Args:
            model: Modelo a agregar
            weight: Peso del modelo (opcional)
            
        Returns:
            Builder
        """
        self.models.append(model)
        if weight is not None:
            if self.weights is None:
                self.weights = []
            self.weights.append(weight)
        return self
    
    def with_strategy(self, strategy: str) -> 'EnsembleBuilder':
        """
        Configurar estrategia.
        
        Args:
            strategy: Estrategia del ensemble
            
        Returns:
            Builder
        """
        self.strategy = strategy
        return self
    
    def with_weights(self, weights: List[float]) -> 'EnsembleBuilder':
        """
        Configurar pesos.
        
        Args:
            weights: Lista de pesos
            
        Returns:
            Builder
        """
        self.weights = weights
        return self
    
    def build(self) -> EnsembleModel:
        """
        Construir ensemble.
        
        Returns:
            Ensemble model
        """
        if not self.models:
            raise ValueError("At least one model is required")
        
        return EnsembleModel(
            models=self.models,
            strategy=self.strategy,
            weights=self.weights
        )








