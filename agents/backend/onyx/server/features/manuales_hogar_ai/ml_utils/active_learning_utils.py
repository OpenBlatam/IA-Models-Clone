"""
Active Learning Utils - Utilidades de Active Learning
======================================================

Utilidades para active learning y selección de muestras.
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Callable, Tuple
import numpy as np
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class UncertaintySampler:
    """
    Muestreador basado en incertidumbre.
    """
    
    def __init__(self, strategy: str = "entropy"):
        """
        Inicializar muestreador.
        
        Args:
            strategy: Estrategia ('entropy', 'margin', 'least_confidence')
        """
        self.strategy = strategy
    
    def compute_uncertainty(
        self,
        predictions: torch.Tensor
    ) -> torch.Tensor:
        """
        Calcular incertidumbre.
        
        Args:
            predictions: Predicciones (logits o probabilidades)
            
        Returns:
            Incertidumbre por muestra
        """
        if self.strategy == "entropy":
            probs = F.softmax(predictions, dim=1)
            entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=1)
            return entropy
        
        elif self.strategy == "margin":
            probs = F.softmax(predictions, dim=1)
            sorted_probs, _ = torch.sort(probs, dim=1, descending=True)
            margin = sorted_probs[:, 0] - sorted_probs[:, 1]
            return -margin  # Negativo porque queremos alta incertidumbre
        
        elif self.strategy == "least_confidence":
            probs = F.softmax(predictions, dim=1)
            max_probs = probs.max(dim=1)[0]
            return 1 - max_probs  # Alta incertidumbre = baja confianza
        
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
    
    def select_samples(
        self,
        uncertainties: torch.Tensor,
        num_samples: int
    ) -> List[int]:
        """
        Seleccionar muestras más inciertas.
        
        Args:
            uncertainties: Incertidumbres
            num_samples: Número de muestras a seleccionar
            
        Returns:
            Índices de muestras seleccionadas
        """
        _, indices = torch.topk(uncertainties, num_samples, largest=True)
        return indices.cpu().tolist()


class DiversitySampler:
    """
    Muestreador basado en diversidad.
    """
    
    def __init__(self, metric: str = "cosine"):
        """
        Inicializar muestreador de diversidad.
        
        Args:
            metric: Métrica de distancia ('cosine', 'euclidean')
        """
        self.metric = metric
    
    def compute_diversity(
        self,
        features: torch.Tensor,
        selected_indices: List[int]
    ) -> torch.Tensor:
        """
        Calcular diversidad.
        
        Args:
            features: Features de todas las muestras
            selected_indices: Índices ya seleccionados
            
        Returns:
            Scores de diversidad
        """
        if len(selected_indices) == 0:
            # Si no hay muestras seleccionadas, todas son igualmente diversas
            return torch.ones(features.shape[0])
        
        selected_features = features[selected_indices]
        
        if self.metric == "cosine":
            # Distancia coseno promedio
            similarities = F.cosine_similarity(
                features.unsqueeze(1),
                selected_features.unsqueeze(0),
                dim=2
            )
            diversity = 1 - similarities.max(dim=1)[0]
        
        elif self.metric == "euclidean":
            # Distancia euclidiana mínima
            distances = torch.cdist(features, selected_features)
            diversity = distances.min(dim=1)[0]
        
        else:
            raise ValueError(f"Unknown metric: {self.metric}")
        
        return diversity
    
    def select_diverse_samples(
        self,
        features: torch.Tensor,
        num_samples: int,
        initial_indices: Optional[List[int]] = None
    ) -> List[int]:
        """
        Seleccionar muestras diversas.
        
        Args:
            features: Features de todas las muestras
            num_samples: Número de muestras a seleccionar
            initial_indices: Índices iniciales (opcional)
            
        Returns:
            Índices de muestras seleccionadas
        """
        selected = initial_indices or []
        
        for _ in range(num_samples - len(selected)):
            diversity = self.compute_diversity(features, selected)
            # Seleccionar muestra más diversa
            max_diversity_idx = diversity.argmax().item()
            selected.append(max_diversity_idx)
        
        return selected


class QueryByCommittee:
    """
    Query by Committee para active learning.
    """
    
    def __init__(self, models: List[nn.Module]):
        """
        Inicializar QBC.
        
        Args:
            models: Lista de modelos (comité)
        """
        self.models = models
    
    def compute_disagreement(
        self,
        inputs: torch.Tensor
    ) -> torch.Tensor:
        """
        Calcular desacuerdo entre modelos.
        
        Args:
            inputs: Inputs
            
        Returns:
            Scores de desacuerdo
        """
        predictions = []
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                pred = model(inputs)
                if len(pred.shape) > 1:
                    pred = pred.argmax(dim=1)
                predictions.append(pred)
        
        predictions = torch.stack(predictions)
        
        # Calcular varianza de predicciones
        disagreement = predictions.float().var(dim=0)
        
        return disagreement
    
    def select_samples(
        self,
        inputs: torch.Tensor,
        num_samples: int
    ) -> List[int]:
        """
        Seleccionar muestras con mayor desacuerdo.
        
        Args:
            inputs: Inputs
            num_samples: Número de muestras
            
        Returns:
            Índices seleccionados
        """
        disagreement = self.compute_disagreement(inputs)
        _, indices = torch.topk(disagreement, num_samples, largest=True)
        return indices.cpu().tolist()


class ActiveLearningLoop:
    """
    Loop completo de active learning.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_dataset: Dataset,
        unlabeled_dataset: Dataset,
        sampler: UncertaintySampler,
        query_size: int = 100
    ):
        """
        Inicializar loop de active learning.
        
        Args:
            model: Modelo
            train_dataset: Dataset etiquetado
            unlabeled_dataset: Dataset sin etiquetar
            sampler: Muestreador
            query_size: Tamaño de query
        """
        self.model = model
        self.train_dataset = train_dataset
        self.unlabeled_dataset = unlabeled_dataset
        self.sampler = sampler
        self.query_size = query_size
    
    def query(
        self,
        unlabeled_loader: torch.utils.data.DataLoader
    ) -> List[int]:
        """
        Realizar query de muestras.
        
        Args:
            unlabeled_loader: DataLoader de datos sin etiquetar
            
        Returns:
            Índices de muestras a etiquetar
        """
        all_predictions = []
        all_indices = []
        
        self.model.eval()
        with torch.no_grad():
            for batch_idx, (inputs, _) in enumerate(unlabeled_loader):
                outputs = self.model(inputs)
                all_predictions.append(outputs)
                # Calcular índices globales
                start_idx = batch_idx * unlabeled_loader.batch_size
                indices = list(range(start_idx, start_idx + len(inputs)))
                all_indices.extend(indices)
        
        all_predictions = torch.cat(all_predictions, dim=0)
        uncertainties = self.sampler.compute_uncertainty(all_predictions)
        selected_indices = self.sampler.select_samples(
            uncertainties,
            self.query_size
        )
        
        return [all_indices[i] for i in selected_indices]
    
    def update(
        self,
        selected_indices: List[int],
        labels: List
    ):
        """
        Actualizar dataset etiquetado.
        
        Args:
            selected_indices: Índices seleccionados
            labels: Labels correspondientes
        """
        # En una implementación real, esto actualizaría el dataset
        # Por ahora, solo loggeamos
        logger.info(f"Adding {len(selected_indices)} labeled samples")
        # Aquí se actualizaría el train_dataset con las nuevas muestras




