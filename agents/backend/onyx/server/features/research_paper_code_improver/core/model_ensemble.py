"""
Model Ensemble Manager - Gestor de ensembles de modelos
========================================================
"""

import logging
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


class EnsembleMethod(Enum):
    """Métodos de ensemble"""
    AVERAGE = "average"
    WEIGHTED_AVERAGE = "weighted_average"
    VOTING = "voting"
    STACKING = "stacking"


@dataclass
class EnsembleConfig:
    """Configuración de ensemble"""
    method: EnsembleMethod = EnsembleMethod.AVERAGE
    weights: Optional[List[float]] = None
    voting_strategy: str = "hard"  # "hard" or "soft"


class ModelEnsemble:
    """Ensemble de modelos"""
    
    def __init__(self, config: EnsembleConfig):
        self.config = config
        self.models: List[nn.Module] = []
        self.weights: List[float] = []
    
    def add_model(self, model: nn.Module, weight: float = 1.0):
        """Agrega un modelo al ensemble"""
        self.models.append(model)
        self.weights.append(weight)
        logger.info(f"Modelo agregado al ensemble. Total: {len(self.models)}")
    
    def predict(
        self,
        inputs: Any,
        device: str = "cuda"
    ) -> torch.Tensor:
        """Predice con el ensemble"""
        if not self.models:
            raise ValueError("No hay modelos en el ensemble")
        
        device = torch.device(device)
        predictions = []
        
        for model in self.models:
            model.eval()
            model = model.to(device)
            
            with torch.no_grad():
                if isinstance(inputs, dict):
                    inputs_device = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                                   for k, v in inputs.items()}
                    outputs = model(**inputs_device)
                else:
                    inputs_device = inputs.to(device)
                    outputs = model(inputs_device)
                
                if hasattr(outputs, 'logits'):
                    pred = outputs.logits
                elif isinstance(outputs, torch.Tensor):
                    pred = outputs
                else:
                    pred = outputs
                
                predictions.append(pred.cpu())
        
        # Combinar predicciones
        if self.config.method == EnsembleMethod.AVERAGE:
            ensemble_pred = torch.stack(predictions).mean(dim=0)
        
        elif self.config.method == EnsembleMethod.WEIGHTED_AVERAGE:
            weights = torch.tensor(self.weights, dtype=torch.float32)
            weights = weights / weights.sum()  # Normalizar
            weighted_preds = [pred * w for pred, w in zip(predictions, weights)]
            ensemble_pred = torch.stack(weighted_preds).sum(dim=0)
        
        elif self.config.method == EnsembleMethod.VOTING:
            if self.config.voting_strategy == "hard":
                # Hard voting: mayoría
                pred_classes = [torch.argmax(pred, dim=-1) for pred in predictions]
                ensemble_pred = torch.mode(torch.stack(pred_classes), dim=0)[0]
            else:
                # Soft voting: promedio de probabilidades
                ensemble_pred = torch.stack(predictions).mean(dim=0)
        
        else:  # STACKING
            # Stacking: usar predicciones como features para meta-modelo
            stacked = torch.cat(predictions, dim=-1)
            ensemble_pred = stacked  # En producción, pasar por meta-modelo
        
        return ensemble_pred
    
    def get_ensemble_info(self) -> Dict[str, Any]:
        """Obtiene información del ensemble"""
        return {
            "num_models": len(self.models),
            "method": self.config.method.value,
            "weights": self.weights,
            "model_types": [type(m).__name__ for m in self.models]
        }




