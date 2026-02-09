"""
Active Learning System - Sistema de aprendizaje activo
=======================================================
"""

import logging
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class QueryStrategy(Enum):
    """Estrategias de consulta"""
    UNCERTAINTY = "uncertainty"
    DIVERSITY = "diversity"
    REPRESENTATIVE = "representative"
    RANDOM = "random"


@dataclass
class QueryResult:
    """Resultado de consulta"""
    indices: List[int]
    scores: List[float]
    strategy: str


class ActiveLearningSystem:
    """Sistema de aprendizaje activo"""
    
    def __init__(self, query_strategy: QueryStrategy = QueryStrategy.UNCERTAINTY):
        self.query_strategy = query_strategy
        self.query_history: List[QueryResult] = []
        self.labeled_indices: List[int] = []
        self.unlabeled_indices: List[int] = []
    
    def query_samples(
        self,
        model: nn.Module,
        unlabeled_data: Any,
        num_samples: int,
        device: str = "cuda"
    ) -> QueryResult:
        """Consulta muestras para etiquetar"""
        if self.query_strategy == QueryStrategy.UNCERTAINTY:
            return self._uncertainty_sampling(model, unlabeled_data, num_samples, device)
        elif self.query_strategy == QueryStrategy.DIVERSITY:
            return self._diversity_sampling(model, unlabeled_data, num_samples, device)
        elif self.query_strategy == QueryStrategy.RANDOM:
            return self._random_sampling(unlabeled_data, num_samples)
        else:
            return self._uncertainty_sampling(model, unlabeled_data, num_samples, device)
    
    def _uncertainty_sampling(
        self,
        model: nn.Module,
        unlabeled_data: Any,
        num_samples: int,
        device: str
    ) -> QueryResult:
        """Muestreo por incertidumbre"""
        device = torch.device(device)
        model = model.to(device)
        model.eval()
        
        uncertainties = []
        indices = []
        
        with torch.no_grad():
            for idx, batch in enumerate(unlabeled_data):
                if isinstance(batch, dict):
                    inputs = batch.get("input_ids") or batch.get("inputs")
                    inputs = inputs.to(device)
                    outputs = model(**batch)
                else:
                    inputs = batch[0] if isinstance(batch, tuple) else batch
                    inputs = inputs.to(device)
                    outputs = model(inputs)
                
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                else:
                    logits = outputs
                
                probs = torch.nn.functional.softmax(logits, dim=-1)
                
                # Entropía como medida de incertidumbre
                entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
                uncertainties.extend(entropy.cpu().numpy())
                indices.extend([idx] * len(entropy))
        
        # Seleccionar muestras con mayor incertidumbre
        uncertainties = np.array(uncertainties)
        top_indices = np.argsort(uncertainties)[-num_samples:][::-1]
        
        selected_indices = [indices[i] for i in top_indices]
        selected_scores = [uncertainties[i] for i in top_indices]
        
        result = QueryResult(
            indices=selected_indices,
            scores=selected_scores,
            strategy="uncertainty"
        )
        
        self.query_history.append(result)
        return result
    
    def _diversity_sampling(
        self,
        model: nn.Module,
        unlabeled_data: Any,
        num_samples: int,
        device: str
    ) -> QueryResult:
        """Muestreo por diversidad (simplificado)"""
        # Fallback a uncertainty sampling
        return self._uncertainty_sampling(model, unlabeled_data, num_samples, device)
    
    def _random_sampling(
        self,
        unlabeled_data: Any,
        num_samples: int
    ) -> QueryResult:
        """Muestreo aleatorio"""
        total_samples = len(unlabeled_data) if hasattr(unlabeled_data, '__len__') else num_samples
        selected_indices = np.random.choice(total_samples, size=min(num_samples, total_samples), replace=False).tolist()
        
        result = QueryResult(
            indices=selected_indices,
            scores=[1.0] * len(selected_indices),
            strategy="random"
        )
        
        self.query_history.append(result)
        return result
    
    def update_labeled_set(self, indices: List[int]):
        """Actualiza conjunto etiquetado"""
        self.labeled_indices.extend(indices)
        self.unlabeled_indices = [i for i in self.unlabeled_indices if i not in indices]
        logger.info(f"Conjunto etiquetado actualizado: {len(self.labeled_indices)} muestras")




