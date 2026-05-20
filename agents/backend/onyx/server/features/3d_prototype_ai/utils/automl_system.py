"""
AutoML System - Sistema de AutoML
==================================
Automated Machine Learning con búsqueda de arquitectura y optimización
"""

import logging
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
import random

try:
    from auto_ml import Predictor
    AUTOML_AVAILABLE = True
except ImportError:
    AUTOML_AVAILABLE = False
    logging.warning("auto_ml not available")

logger = logging.getLogger(__name__)


@dataclass
class ArchitectureSearchSpace:
    """Espacio de búsqueda de arquitectura"""
    num_layers: tuple = (2, 6)
    hidden_dims: tuple = (128, 512)
    dropout_rates: tuple = (0.1, 0.5)
    activation_functions: List[str] = None
    
    def __post_init__(self):
        if self.activation_functions is None:
            self.activation_functions = ["relu", "tanh", "gelu", "swish"]


class AutoMLSystem:
    """Sistema de AutoML"""
    
    def __init__(self):
        self.search_space = ArchitectureSearchSpace()
        self.trained_models: Dict[str, nn.Module] = {}
    
    def search_architecture(
        self,
        input_dim: int,
        output_dim: int,
        n_trials: int = 10,
        objective_fn: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """Busca mejor arquitectura"""
        best_model = None
        best_score = float('-inf')
        best_config = None
        
        for trial in range(n_trials):
            # Generar configuración aleatoria
            config = self._generate_random_config(input_dim, output_dim)
            
            # Crear modelo
            model = self._create_model_from_config(config)
            
            # Evaluar (simplificado - en producción usar objective_fn real)
            if objective_fn:
                score = objective_fn(model)
            else:
                score = random.uniform(0.5, 1.0)  # Simulado
            
            if score > best_score:
                best_score = score
                best_model = model
                best_config = config
            
            logger.info(f"Trial {trial+1}/{n_trials}: score={score:.4f}")
        
        return {
            "best_model": best_model,
            "best_score": best_score,
            "best_config": best_config,
            "n_trials": n_trials
        }
    
    def _generate_random_config(self, input_dim: int, output_dim: int) -> Dict[str, Any]:
        """Genera configuración aleatoria"""
        num_layers = random.randint(*self.search_space.num_layers)
        hidden_dims = [
            random.randint(*self.search_space.hidden_dims)
            for _ in range(num_layers)
        ]
        dropout_rates = [
            random.uniform(*self.search_space.dropout_rates)
            for _ in range(num_layers)
        ]
        activations = [
            random.choice(self.search_space.activation_functions)
            for _ in range(num_layers)
        ]
        
        return {
            "input_dim": input_dim,
            "output_dim": output_dim,
            "num_layers": num_layers,
            "hidden_dims": hidden_dims,
            "dropout_rates": dropout_rates,
            "activations": activations
        }
    
    def _create_model_from_config(self, config: Dict[str, Any]) -> nn.Module:
        """Crea modelo desde configuración"""
        layers = []
        prev_dim = config["input_dim"]
        
        for i in range(config["num_layers"]):
            hidden_dim = config["hidden_dims"][i]
            dropout = config["dropout_rates"][i]
            activation = config["activations"][i]
            
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            # Activation
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            elif activation == "gelu":
                layers.append(nn.GELU())
            elif activation == "swish":
                layers.append(nn.SiLU())  # SiLU es similar a Swish
            
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, config["output_dim"]))
        
        return nn.Sequential(*layers)
    
    def automated_feature_engineering(
        self,
        data: torch.Tensor,
        target: torch.Tensor
    ) -> Dict[str, Any]:
        """Feature engineering automático"""
        features = {
            "original_features": data.shape[1],
            "statistical_features": self._extract_statistical_features(data),
            "interaction_features": self._create_interaction_features(data),
            "polynomial_features": self._create_polynomial_features(data, degree=2)
        }
        
        return features
    
    def _extract_statistical_features(self, data: torch.Tensor) -> torch.Tensor:
        """Extrae features estadísticos"""
        mean = data.mean(dim=0, keepdim=True)
        std = data.std(dim=0, keepdim=True)
        min_vals = data.min(dim=0, keepdim=True)[0]
        max_vals = data.max(dim=0, keepdim=True)[0]
        
        return torch.cat([mean, std, min_vals, max_vals], dim=1)
    
    def _create_interaction_features(self, data: torch.Tensor) -> torch.Tensor:
        """Crea features de interacción"""
        # Productos de pares de features (simplificado)
        n_features = data.shape[1]
        interactions = []
        
        for i in range(min(5, n_features)):  # Limitar para eficiencia
            for j in range(i+1, min(5, n_features)):
                interactions.append((data[:, i] * data[:, j]).unsqueeze(1))
        
        return torch.cat(interactions, dim=1) if interactions else data
    
    def _create_polynomial_features(self, data: torch.Tensor, degree: int = 2) -> torch.Tensor:
        """Crea features polinomiales"""
        if degree == 2:
            # Cuadrados de features principales
            squares = data[:, :min(10, data.shape[1])] ** 2
            return torch.cat([data, squares], dim=1)
        return data




