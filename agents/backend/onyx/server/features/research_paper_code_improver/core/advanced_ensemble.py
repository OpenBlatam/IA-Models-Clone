"""
Advanced Ensemble Methods - Métodos avanzados de ensemble
===========================================================
"""

import logging
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class EnsembleMethod(Enum):
    """Métodos de ensemble avanzados"""
    AVERAGE = "average"
    WEIGHTED_AVERAGE = "weighted_average"
    VOTING = "voting"
    STACKING = "stacking"
    BAGGING = "bagging"
    BOOSTING = "boosting"
    BLAENDING = "blending"


@dataclass
class EnsembleMember:
    """Miembro del ensemble"""
    model: nn.Module
    weight: float = 1.0
    performance: float = 0.0
    name: str = ""


class AdvancedEnsemble:
    """Ensemble avanzado"""
    
    def __init__(self, method: EnsembleMethod = EnsembleMethod.WEIGHTED_AVERAGE):
        self.method = method
        self.members: List[EnsembleMember] = []
        self.meta_model: Optional[nn.Module] = None  # Para stacking
    
    def add_member(self, model: nn.Module, weight: float = 1.0, name: str = "", performance: float = 0.0):
        """Agrega miembro al ensemble"""
        member = EnsembleMember(
            model=model,
            weight=weight,
            performance=performance,
            name=name or f"model_{len(self.members)}"
        )
        self.members.append(member)
        logger.info(f"Miembro agregado: {member.name}")
    
    def predict(
        self,
        inputs: Any,
        device: str = "cuda"
    ) -> torch.Tensor:
        """Predice con el ensemble"""
        if not self.members:
            raise ValueError("No hay miembros en el ensemble")
        
        device = torch.device(device)
        predictions = []
        
        for member in self.members:
            member.model.eval()
            member.model = member.model.to(device)
            
            with torch.no_grad():
                if isinstance(inputs, dict):
                    inputs_device = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                                   for k, v in inputs.items()}
                    outputs = member.model(**inputs_device)
                else:
                    inputs_device = inputs.to(device)
                    outputs = member.model(inputs_device)
                
                if hasattr(outputs, 'logits'):
                    pred = outputs.logits
                elif isinstance(outputs, torch.Tensor):
                    pred = outputs
                else:
                    pred = outputs
                
                predictions.append(pred.cpu())
        
        # Combinar predicciones según método
        if self.method == EnsembleMethod.AVERAGE:
            ensemble_pred = torch.stack(predictions).mean(dim=0)
        
        elif self.method == EnsembleMethod.WEIGHTED_AVERAGE:
            weights = torch.tensor([m.weight for m in self.members], dtype=torch.float32)
            weights = weights / weights.sum()  # Normalizar
            
            weighted_preds = [pred * w for pred, w in zip(predictions, weights)]
            ensemble_pred = torch.stack(weighted_preds).sum(dim=0)
        
        elif self.method == EnsembleMethod.VOTING:
            # Hard voting
            pred_classes = [torch.argmax(pred, dim=-1) for pred in predictions]
            ensemble_pred = torch.mode(torch.stack(pred_classes), dim=0)[0]
        
        elif self.method == EnsembleMethod.STACKING:
            # Stacking: usar meta-modelo
            if self.meta_model:
                stacked = torch.cat(predictions, dim=-1)
                self.meta_model.eval()
                with torch.no_grad():
                    ensemble_pred = self.meta_model(stacked)
            else:
                # Fallback a promedio si no hay meta-modelo
                ensemble_pred = torch.stack(predictions).mean(dim=0)
        
        elif self.method == EnsembleMethod.BAGGING:
            # Bagging: promedio simple
            ensemble_pred = torch.stack(predictions).mean(dim=0)
        
        else:
            # Default: promedio
            ensemble_pred = torch.stack(predictions).mean(dim=0)
        
        return ensemble_pred
    
    def set_meta_model(self, meta_model: nn.Module):
        """Establece meta-modelo para stacking"""
        self.meta_model = meta_model
        logger.info("Meta-modelo establecido para stacking")
    
    def optimize_weights(self, validation_data: Any, device: str = "cuda"):
        """Optimiza pesos del ensemble"""
        # Optimización simple basada en performance
        total_performance = sum(m.performance for m in self.members)
        
        if total_performance > 0:
            for member in self.members:
                member.weight = member.performance / total_performance
        else:
            # Pesos iguales si no hay performance
            weight = 1.0 / len(self.members)
            for member in self.members:
                member.weight = weight
        
        logger.info("Pesos del ensemble optimizados")




