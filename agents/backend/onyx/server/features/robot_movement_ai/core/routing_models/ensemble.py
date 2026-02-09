"""
Model Ensemble
==============

Ensamblaje de múltiples modelos para mejor rendimiento.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional
import numpy as np
from collections import OrderedDict

from .base_model import BaseRouteModel


class ModelEnsemble(nn.Module):
    """
    Ensamblaje de múltiples modelos.
    """
    
    def __init__(
        self,
        models: List[BaseRouteModel],
        weights: Optional[List[float]] = None,
        voting_method: str = "average"  # average, weighted_average, voting
    ):
        """
        Inicializar ensamblaje.
        
        Args:
            models: Lista de modelos
            weights: Pesos para cada modelo (opcional)
            voting_method: Método de votación
        """
        super(ModelEnsemble, self).__init__()
        
        self.models = nn.ModuleList(models)
        self.voting_method = voting_method
        
        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        
        # Normalizar pesos
        total_weight = sum(weights)
        self.weights = [w / total_weight for w in weights]
        
        # Validar
        assert len(models) == len(weights), "Número de modelos y pesos debe coincidir"
        assert len(models) > 0, "Debe haber al menos un modelo"
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass del ensamblaje.
        
        Args:
            x: Tensor de entrada
            
        Returns:
            Predicción ensamblada
        """
        predictions = []
        
        for model in self.models:
            pred = model(x)
            predictions.append(pred)
        
        # Stack predictions
        stacked = torch.stack(predictions, dim=0)  # [n_models, batch_size, output_dim]
        
        if self.voting_method == "average":
            return torch.mean(stacked, dim=0)
        
        elif self.voting_method == "weighted_average":
            weights_tensor = torch.tensor(
                self.weights,
                device=stacked.device,
                dtype=stacked.dtype
            ).view(-1, 1, 1)
            return torch.sum(stacked * weights_tensor, dim=0)
        
        elif self.voting_method == "voting":
            # Para clasificación (no aplicable aquí, pero incluido)
            return torch.mode(stacked, dim=0)[0]
        
        else:
            raise ValueError(f"Método de votación no soportado: {self.voting_method}")
    
    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        return_std: bool = True
    ) -> tuple:
        """
        Predecir con estimación de incertidumbre.
        
        Args:
            x: Tensor de entrada
            return_std: Retornar desviación estándar
            
        Returns:
            (mean, std) o solo mean
        """
        self.eval()
        
        with torch.no_grad():
            predictions = []
            for model in self.models:
                pred = model(x)
                predictions.append(pred)
            
            stacked = torch.stack(predictions, dim=0)  # [n_models, batch_size, output_dim]
            mean = torch.mean(stacked, dim=0)
            
            if return_std:
                std = torch.std(stacked, dim=0)
                return mean, std
            
            return mean
    
    def get_model_diversity(self) -> float:
        """
        Calcular diversidad entre modelos (usando parámetros).
        
        Returns:
            Score de diversidad
        """
        # Calcular distancia promedio entre parámetros de modelos
        distances = []
        
        for i in range(len(self.models)):
            for j in range(i + 1, len(self.models)):
                model1_params = torch.cat([p.flatten() for p in self.models[i].parameters()])
                model2_params = torch.cat([p.flatten() for p in self.models[j].parameters()])
                
                distance = torch.norm(model1_params - model2_params).item()
                distances.append(distance)
        
        return np.mean(distances) if distances else 0.0


class EnsembleBuilder:
    """
    Constructor de ensamblajes.
    """
    
    @staticmethod
    def create_bagging_ensemble(
        models: List[BaseRouteModel],
        n_samples: int = 5
    ) -> ModelEnsemble:
        """
        Crear ensamblaje usando bagging.
        
        Args:
            models: Lista de modelos
            n_samples: Número de muestras para bagging
            
        Returns:
            Ensamblaje
        """
        return ModelEnsemble(models, voting_method="average")
    
    @staticmethod
    def create_stacking_ensemble(
        base_models: List[BaseRouteModel],
        meta_model: Optional[BaseRouteModel] = None
    ) -> ModelEnsemble:
        """
        Crear ensamblaje usando stacking (requiere meta-modelo).
        
        Args:
            base_models: Modelos base
            meta_model: Meta-modelo (opcional)
            
        Returns:
            Ensamblaje
        """
        # Por simplicidad, retornamos ensemble promedio
        # En implementación completa, se entrenaría meta-modelo
        return ModelEnsemble(base_models, voting_method="weighted_average")
    
    @staticmethod
    def create_diverse_ensemble(
        models: List[BaseRouteModel],
        diversity_threshold: float = 0.1
    ) -> ModelEnsemble:
        """
        Crear ensamblaje con modelos diversos.
        
        Args:
            models: Lista de modelos
            diversity_threshold: Umbral de diversidad
            
        Returns:
            Ensamblaje
        """
        # Filtrar modelos similares
        diverse_models = [models[0]]
        
        for model in models[1:]:
            ensemble = ModelEnsemble(diverse_models + [model])
            diversity = ensemble.get_model_diversity()
            
            if diversity > diversity_threshold:
                diverse_models.append(model)
        
        return ModelEnsemble(diverse_models, voting_method="average")


