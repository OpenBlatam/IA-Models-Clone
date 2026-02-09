"""
Uncertainty Estimation - Estimación de incertidumbre
======================================================
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class UncertaintyMethod(Enum):
    """Métodos de estimación de incertidumbre"""
    DROPOUT = "dropout"  # Monte Carlo Dropout
    ENSEMBLE = "ensemble"
    DEEP_ENSEMBLE = "deep_ensemble"
    VARIATIONAL = "variational"


@dataclass
class UncertaintyResult:
    """Resultado de estimación de incertidumbre"""
    prediction: torch.Tensor
    aleatoric_uncertainty: torch.Tensor  # Incertidumbre de datos
    epistemic_uncertainty: torch.Tensor  # Incertidumbre del modelo
    total_uncertainty: torch.Tensor
    confidence_interval: Tuple[torch.Tensor, torch.Tensor]


class UncertaintyEstimator:
    """Estimador de incertidumbre"""
    
    def __init__(self, method: UncertaintyMethod = UncertaintyMethod.DROPOUT):
        self.method = method
        self.uncertainty_results: List[UncertaintyResult] = []
    
    def estimate_uncertainty(
        self,
        model: nn.Module,
        inputs: torch.Tensor,
        num_samples: int = 10,
        device: str = "cuda"
    ) -> UncertaintyResult:
        """Estima incertidumbre"""
        if self.method == UncertaintyMethod.DROPOUT:
            return self._monte_carlo_dropout(model, inputs, num_samples, device)
        elif self.method == UncertaintyMethod.ENSEMBLE:
            return self._ensemble_uncertainty(model, inputs, device)
        else:
            return self._monte_carlo_dropout(model, inputs, num_samples, device)
    
    def _monte_carlo_dropout(
        self,
        model: nn.Module,
        inputs: torch.Tensor,
        num_samples: int,
        device: str
    ) -> UncertaintyResult:
        """Monte Carlo Dropout"""
        device = torch.device(device)
        model = model.to(device)
        
        # Habilitar dropout en evaluación
        model.train()
        
        inputs = inputs.to(device)
        predictions = []
        
        with torch.no_grad():
            for _ in range(num_samples):
                outputs = model(inputs)
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                else:
                    logits = outputs
                
                probs = F.softmax(logits, dim=-1)
                predictions.append(probs)
        
        predictions = torch.stack(predictions)  # [num_samples, batch_size, num_classes]
        
        # Calcular estadísticas
        mean_prediction = predictions.mean(dim=0)
        variance = predictions.var(dim=0)
        
        # Aleatoric uncertainty (promedio de varianza)
        aleatoric = variance.mean(dim=-1)
        
        # Epistemic uncertainty (varianza del promedio)
        epistemic = mean_prediction.var(dim=-1)
        
        # Total uncertainty
        total = aleatoric + epistemic
        
        # Confidence interval (95%)
        std = torch.sqrt(total)
        lower = mean_prediction.argmax(dim=-1) - 1.96 * std
        upper = mean_prediction.argmax(dim=-1) + 1.96 * std
        
        return UncertaintyResult(
            prediction=mean_prediction.argmax(dim=-1),
            aleatoric_uncertainty=aleatoric,
            epistemic_uncertainty=epistemic,
            total_uncertainty=total,
            confidence_interval=(lower, upper)
        )
    
    def _ensemble_uncertainty(
        self,
        models: List[nn.Module],
        inputs: torch.Tensor,
        device: str
    ) -> UncertaintyResult:
        """Incertidumbre usando ensemble"""
        device = torch.device(device)
        inputs = inputs.to(device)
        
        predictions = []
        
        for model in models:
            model.eval()
            model = model.to(device)
            
            with torch.no_grad():
                outputs = model(inputs)
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                else:
                    logits = outputs
                
                probs = F.softmax(logits, dim=-1)
                predictions.append(probs)
        
        predictions = torch.stack(predictions)
        
        mean_prediction = predictions.mean(dim=0)
        variance = predictions.var(dim=0)
        
        aleatoric = variance.mean(dim=-1)
        epistemic = mean_prediction.var(dim=-1)
        total = aleatoric + epistemic
        
        std = torch.sqrt(total)
        lower = mean_prediction.argmax(dim=-1) - 1.96 * std
        upper = mean_prediction.argmax(dim=-1) + 1.96 * std
        
        return UncertaintyResult(
            prediction=mean_prediction.argmax(dim=-1),
            aleatoric_uncertainty=aleatoric,
            epistemic_uncertainty=epistemic,
            total_uncertainty=total,
            confidence_interval=(lower, upper)
        )




