"""
Model Ensembling Module
========================

Sistema profesional para ensamblaje de modelos.
Incluye voting, averaging, stacking y weighted ensembling.
"""

import logging
from typing import Dict, Any, Optional, List, Union, Callable
import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    logging.warning("PyTorch not available. Ensembling disabled.")

logger = logging.getLogger(__name__)


class ModelEnsemble:
    """
    Sistema de ensamblaje de modelos.
    
    Soporta:
    - Voting (hard/soft)
    - Averaging
    - Weighted averaging
    - Stacking
    """
    
    def __init__(
        self,
        models: List[nn.Module],
        ensemble_method: str = "average",
        weights: Optional[List[float]] = None
    ):
        """
        Inicializar ensemble.
        
        Args:
            models: Lista de modelos
            ensemble_method: Método ("average", "weighted_average", "voting", "stacking")
            weights: Pesos para weighted averaging
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for ModelEnsemble")
        
        self.models = models
        self.ensemble_method = ensemble_method
        self.weights = weights
        
        if weights and len(weights) != len(models):
            raise ValueError(f"Number of weights ({len(weights)}) must match number of models ({len(models)})")
        
        if weights:
            # Normalizar pesos
            total = sum(weights)
            self.weights = [w / total for w in weights]
        
        logger.info(f"ModelEnsemble initialized with {len(models)} models using {ensemble_method}")
    
    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Predecir con ensemble.
        
        Args:
            inputs: Inputs para los modelos
            
        Returns:
            Predicción del ensemble
        """
        predictions = []
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                pred = model(inputs)
                predictions.append(pred)
        
        if self.ensemble_method == "average":
            return torch.stack(predictions).mean(dim=0)
        
        elif self.ensemble_method == "weighted_average":
            if not self.weights:
                raise ValueError("Weights required for weighted_average")
            weighted_preds = [pred * w for pred, w in zip(predictions, self.weights)]
            return torch.stack(weighted_preds).sum(dim=0)
        
        elif self.ensemble_method == "voting":
            # Hard voting para clasificación
            stacked = torch.stack(predictions)
            if stacked.ndim > 2:
                # Probabilidades -> clases
                classes = stacked.argmax(dim=-1)
                # Votación por mayoría
                return torch.mode(classes, dim=0)[0]
            else:
                # Regresión: promedio
                return stacked.mean(dim=0)
        
        elif self.ensemble_method == "max":
            return torch.stack(predictions).max(dim=0)[0]
        
        elif self.ensemble_method == "min":
            return torch.stack(predictions).min(dim=0)[0]
        
        else:
            raise ValueError(f"Unknown ensemble method: {self.ensemble_method}")
    
    def predict_with_uncertainty(
        self,
        inputs: torch.Tensor,
        num_samples: int = 10
    ) -> Dict[str, torch.Tensor]:
        """
        Predecir con estimación de incertidumbre.
        
        Args:
            inputs: Inputs para los modelos
            num_samples: Número de muestras por modelo
            
        Returns:
            Dict con predicción media y std
        """
        all_predictions = []
        
        for model in self.models:
            model.train()  # Activar dropout si existe
            model_predictions = []
            
            for _ in range(num_samples):
                with torch.no_grad():
                    pred = model(inputs)
                    model_predictions.append(pred)
            
            model.eval()
            all_predictions.extend(model_predictions)
        
        stacked = torch.stack(all_predictions)
        mean_pred = stacked.mean(dim=0)
        std_pred = stacked.std(dim=0)
        
        return {
            "mean": mean_pred,
            "std": std_pred,
            "predictions": stacked
        }


class StackingEnsemble:
    """
    Ensemble con stacking (meta-learner).
    
    Usa un modelo meta para combinar predicciones de modelos base.
    """
    
    def __init__(
        self,
        base_models: List[nn.Module],
        meta_model: Optional[nn.Module] = None,
        meta_input_size: Optional[int] = None
    ):
        """
        Inicializar stacking ensemble.
        
        Args:
            base_models: Modelos base
            meta_model: Modelo meta (opcional, se crea si no se proporciona)
            meta_input_size: Tamaño de entrada del meta modelo
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for StackingEnsemble")
        
        self.base_models = base_models
        
        if meta_model is None:
            if meta_input_size is None:
                meta_input_size = len(base_models)
            self.meta_model = self._create_default_meta_model(meta_input_size)
        else:
            self.meta_model = meta_model
        
        logger.info(f"StackingEnsemble initialized with {len(base_models)} base models")
    
    def _create_default_meta_model(self, input_size: int) -> nn.Module:
        """Crear modelo meta por defecto."""
        return nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def fit_meta_model(
        self,
        base_predictions: torch.Tensor,
        targets: torch.Tensor,
        epochs: int = 100
    ):
        """
        Entrenar modelo meta.
        
        Args:
            base_predictions: Predicciones de modelos base (N, num_models, output_dim)
            targets: Targets reales
            epochs: Número de épocas
        """
        optimizer = torch.optim.Adam(self.meta_model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        
        self.meta_model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # Flatten predictions si es necesario
            if base_predictions.ndim > 2:
                meta_input = base_predictions.view(base_predictions.size(0), -1)
            else:
                meta_input = base_predictions
            
            output = self.meta_model(meta_input)
            loss = criterion(output, targets)
            
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 20 == 0:
                logger.info(f"Meta model epoch {epoch + 1}/{epochs}: loss = {loss.item():.4f}")
    
    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Predecir con stacking ensemble.
        
        Args:
            inputs: Inputs para los modelos
            
        Returns:
            Predicción del ensemble
        """
        # Obtener predicciones de modelos base
        base_predictions = []
        for model in self.base_models:
            model.eval()
            with torch.no_grad():
                pred = model(inputs)
                base_predictions.append(pred)
        
        # Stack predicciones
        stacked = torch.stack(base_predictions, dim=1)  # (batch, num_models, output_dim)
        
        # Flatten para meta modelo
        if stacked.ndim > 2:
            meta_input = stacked.view(stacked.size(0), -1)
        else:
            meta_input = stacked
        
        # Predicción del meta modelo
        self.meta_model.eval()
        with torch.no_grad():
            final_pred = self.meta_model(meta_input)
        
        return final_pred

