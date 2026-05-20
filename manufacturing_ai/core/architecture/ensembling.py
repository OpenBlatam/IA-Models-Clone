"""
Model Ensembling
================

Sistema de ensembling para combinar múltiples modelos.
"""

import logging
from typing import List, Dict, Any, Optional, Callable
import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None

logger = logging.getLogger(__name__)


class ModelEnsemble(nn.Module):
    """
    Ensemble de modelos.
    
    Combina predicciones de múltiples modelos.
    """
    
    def __init__(
        self,
        models: List[nn.Module],
        weights: Optional[List[float]] = None,
        method: str = "average"
    ):
        """
        Inicializar ensemble.
        
        Args:
            models: Lista de modelos
            weights: Pesos para cada modelo (None para iguales)
            method: Método de combinación (average, weighted, voting)
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required")
        
        super().__init__()
        
        self.models = nn.ModuleList(models)
        self.method = method
        
        if weights is None:
            self.weights = [1.0 / len(models)] * len(models)
        else:
            assert len(weights) == len(models), "Weights must match models"
            # Normalizar pesos
            total = sum(weights)
            self.weights = [w / total for w in weights]
        
        logger.info(f"Created ensemble with {len(models)} models using {method}")
    
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            *args: Argumentos para modelos
            **kwargs: Keyword arguments para modelos
            
        Returns:
            Predicción combinada
        """
        predictions = []
        
        for model in self.models:
            pred = model(*args, **kwargs)
            predictions.append(pred)
        
        # Combinar predicciones
        if self.method == "average":
            return self._average(predictions)
        elif self.method == "weighted":
            return self._weighted_average(predictions)
        elif self.method == "voting":
            return self._voting(predictions)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _average(self, predictions: List[torch.Tensor]) -> torch.Tensor:
        """Promedio simple."""
        stacked = torch.stack(predictions, dim=0)
        return torch.mean(stacked, dim=0)
    
    def _weighted_average(self, predictions: List[torch.Tensor]) -> torch.Tensor:
        """Promedio ponderado."""
        weighted_sum = sum(w * p for w, p in zip(self.weights, predictions))
        return weighted_sum
    
    def _voting(self, predictions: List[torch.Tensor]) -> torch.Tensor:
        """Voting (para clasificación)."""
        # Convertir a clases
        classes = [torch.argmax(p, dim=-1) for p in predictions]
        stacked = torch.stack(classes, dim=0)
        
        # Votación por mayoría
        mode, _ = torch.mode(stacked, dim=0)
        
        # Convertir de vuelta a one-hot
        num_classes = predictions[0].shape[-1]
        result = torch.zeros_like(predictions[0])
        result.scatter_(1, mode.unsqueeze(1), 1.0)
        
        return result


class StackingEnsemble(nn.Module):
    """
    Stacking ensemble.
    
    Usa un meta-modelo para combinar predicciones.
    """
    
    def __init__(
        self,
        base_models: List[nn.Module],
        meta_model: Optional[nn.Module] = None,
        num_classes: int = 3
    ):
        """
        Inicializar stacking ensemble.
        
        Args:
            base_models: Modelos base
            meta_model: Meta-modelo (None para crear uno por defecto)
            num_classes: Número de clases
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required")
        
        super().__init__()
        
        self.base_models = nn.ModuleList(base_models)
        
        if meta_model is None:
            # Crear meta-modelo por defecto
            input_dim = len(base_models) * num_classes
            self.meta_model = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, num_classes)
            )
        else:
            self.meta_model = meta_model
    
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            *args: Argumentos para modelos base
            **kwargs: Keyword arguments
            
        Returns:
            Predicción del meta-modelo
        """
        # Obtener predicciones de modelos base
        base_predictions = []
        for model in self.base_models:
            pred = model(*args, **kwargs)
            base_predictions.append(pred)
        
        # Concatenar predicciones
        stacked = torch.cat(base_predictions, dim=-1)
        
        # Meta-modelo
        output = self.meta_model(stacked)
        
        return output


class EnsembleManager:
    """
    Gestor de ensembles.
    
    Crea y gestiona ensembles de modelos.
    """
    
    def __init__(self):
        """Inicializar gestor."""
        self.ensembles: Dict[str, ModelEnsemble] = {}
    
    def create_ensemble(
        self,
        ensemble_id: str,
        models: List[nn.Module],
        weights: Optional[List[float]] = None,
        method: str = "average"
    ) -> str:
        """
        Crear ensemble.
        
        Args:
            ensemble_id: ID del ensemble
            models: Lista de modelos
            weights: Pesos (opcional)
            method: Método de combinación
            
        Returns:
            ID del ensemble
        """
        ensemble = ModelEnsemble(models, weights, method)
        self.ensembles[ensemble_id] = ensemble
        
        logger.info(f"Created ensemble: {ensemble_id}")
        return ensemble_id
    
    def predict(
        self,
        ensemble_id: str,
        *args,
        **kwargs
    ) -> torch.Tensor:
        """
        Predecir con ensemble.
        
        Args:
            ensemble_id: ID del ensemble
            *args: Argumentos
            **kwargs: Keyword arguments
            
        Returns:
            Predicción
        """
        if ensemble_id not in self.ensembles:
            raise ValueError(f"Ensemble not found: {ensemble_id}")
        
        ensemble = self.ensembles[ensemble_id]
        ensemble.eval()
        
        with torch.no_grad():
            prediction = ensemble(*args, **kwargs)
        
        return prediction


# Instancia global
_ensemble_manager = None


def get_ensemble_manager() -> EnsembleManager:
    """Obtener instancia global."""
    global _ensemble_manager
    if _ensemble_manager is None:
        _ensemble_manager = EnsembleManager()
    return _ensemble_manager

