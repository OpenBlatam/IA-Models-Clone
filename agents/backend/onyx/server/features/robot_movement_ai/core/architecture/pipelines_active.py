"""
Active Learning Module
=======================

Sistema profesional de active learning.
Incluye estrategias de selección de muestras y aprendizaje iterativo.
"""

import logging
from typing import Dict, Any, Optional, List, Tuple, Callable
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    F = None

logger = logging.getLogger(__name__)


class ActiveLearningStrategy:
    """
    Estrategias de active learning.
    
    Soporta:
    - Uncertainty sampling
    - Diversity sampling
    - Query by committee
    - Expected model change
    """
    
    @staticmethod
    def uncertainty_sampling(
        model: nn.Module,
        unlabeled_data: torch.Tensor,
        n_samples: int = 100,
        method: str = "entropy"
    ) -> List[int]:
        """
        Seleccionar muestras basado en incertidumbre.
        
        Args:
            model: Modelo para predecir
            unlabeled_data: Datos sin etiquetar
            n_samples: Número de muestras a seleccionar
            method: Método ("entropy", "margin", "least_confidence")
            
        Returns:
            Índices de muestras seleccionadas
        """
        model.eval()
        uncertainties = []
        
        with torch.no_grad():
            predictions = model(unlabeled_data)
            probabilities = F.softmax(predictions, dim=1)
            
            if method == "entropy":
                # Entropía: mayor entropía = mayor incertidumbre
                entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-10), dim=1)
                uncertainties = entropy.cpu().numpy()
            
            elif method == "margin":
                # Margin: diferencia entre top-2 probabilidades
                top2_probs, _ = torch.topk(probabilities, 2, dim=1)
                margin = top2_probs[:, 0] - top2_probs[:, 1]
                uncertainties = (1 - margin).cpu().numpy()  # Invertir (menor margin = mayor incertidumbre)
            
            elif method == "least_confidence":
                # Menor confianza en la clase predicha
                max_probs, _ = torch.max(probabilities, dim=1)
                uncertainties = (1 - max_probs).cpu().numpy()
        
        # Seleccionar top n_samples más inciertos
        selected_indices = np.argsort(uncertainties)[-n_samples:][::-1].tolist()
        
        return selected_indices
    
    @staticmethod
    def diversity_sampling(
        embeddings: np.ndarray,
        n_samples: int = 100,
        method: str = "kmeans"
    ) -> List[int]:
        """
        Seleccionar muestras diversas.
        
        Args:
            embeddings: Embeddings de los datos
            n_samples: Número de muestras
            method: Método ("kmeans", "farthest_first")
            
        Returns:
            Índices de muestras seleccionadas
        """
        if method == "kmeans":
            try:
                from sklearn.cluster import KMeans
                kmeans = KMeans(n_clusters=n_samples, random_state=42, n_init=10)
                kmeans.fit(embeddings)
                
                # Seleccionar muestra más cercana a cada centroide
                selected_indices = []
                for center in kmeans.cluster_centers_:
                    distances = np.linalg.norm(embeddings - center, axis=1)
                    selected_indices.append(np.argmin(distances))
                
                return selected_indices
            except ImportError:
                logger.warning("sklearn not available, using random sampling")
                return np.random.choice(len(embeddings), n_samples, replace=False).tolist()
        
        elif method == "farthest_first":
            # Farthest-first traversal
            selected_indices = [0]  # Empezar con primera muestra
            remaining_indices = list(range(1, len(embeddings)))
            
            while len(selected_indices) < n_samples and remaining_indices:
                selected_embeddings = embeddings[selected_indices]
                farthest_idx = None
                max_min_distance = -1
                
                for idx in remaining_indices:
                    distances = np.linalg.norm(
                        selected_embeddings - embeddings[idx],
                        axis=1
                    )
                    min_distance = np.min(distances)
                    
                    if min_distance > max_min_distance:
                        max_min_distance = min_distance
                        farthest_idx = idx
                
                if farthest_idx is not None:
                    selected_indices.append(farthest_idx)
                    remaining_indices.remove(farthest_idx)
            
            return selected_indices
    
    @staticmethod
    def query_by_committee(
        models: List[nn.Module],
        unlabeled_data: torch.Tensor,
        n_samples: int = 100
    ) -> List[int]:
        """
        Query by Committee: seleccionar donde los modelos discrepan más.
        
        Args:
            models: Lista de modelos (comité)
            unlabeled_data: Datos sin etiquetar
            n_samples: Número de muestras
            
        Returns:
            Índices de muestras seleccionadas
        """
        all_predictions = []
        
        for model in models:
            model.eval()
            with torch.no_grad():
                predictions = model(unlabeled_data)
                all_predictions.append(predictions.argmax(dim=1).cpu().numpy())
        
        # Calcular desacuerdo (variance en predicciones)
        all_predictions = np.array(all_predictions)
        disagreements = np.var(all_predictions, axis=0)
        
        # Seleccionar muestras con mayor desacuerdo
        selected_indices = np.argsort(disagreements)[-n_samples:][::-1].tolist()
        
        return selected_indices


class ActiveLearningLoop:
    """
    Loop completo de active learning.
    
    Gestiona el ciclo iterativo de selección, etiquetado y entrenamiento.
    """
    
    def __init__(
        self,
        model: nn.Module,
        strategy: str = "uncertainty",
        initial_samples: int = 100
    ):
        """
        Inicializar loop de active learning.
        
        Args:
            model: Modelo a entrenar
            strategy: Estrategia de selección
            initial_samples: Muestras iniciales
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required")
        
        self.model = model
        self.strategy = strategy
        self.initial_samples = initial_samples
        self.labeled_indices: List[int] = []
        self.unlabeled_indices: List[int] = []
        self.iteration = 0
        logger.info(f"ActiveLearningLoop initialized with strategy: {strategy}")
    
    def initialize(
        self,
        dataset_size: int,
        initial_indices: Optional[List[int]] = None
    ):
        """
        Inicializar con muestras iniciales.
        
        Args:
            dataset_size: Tamaño total del dataset
            initial_indices: Índices iniciales (None para aleatorio)
        """
        if initial_indices is None:
            initial_indices = np.random.choice(
                dataset_size,
                min(self.initial_samples, dataset_size),
                replace=False
            ).tolist()
        
        self.labeled_indices = initial_indices
        self.unlabeled_indices = [
            i for i in range(dataset_size) if i not in initial_indices
        ]
        
        logger.info(f"Initialized with {len(self.labeled_indices)} labeled samples")
    
    def select_samples(
        self,
        unlabeled_data: torch.Tensor,
        n_samples: int = 50
    ) -> List[int]:
        """
        Seleccionar muestras para etiquetar.
        
        Args:
            unlabeled_data: Datos sin etiquetar
            n_samples: Número de muestras a seleccionar
            
        Returns:
            Índices de muestras seleccionadas (relativos a unlabeled_data)
        """
        if self.strategy == "uncertainty":
            selected = ActiveLearningStrategy.uncertainty_sampling(
                self.model,
                unlabeled_data,
                n_samples
            )
        elif self.strategy == "diversity":
            # Necesitaríamos embeddings - simplificado
            selected = np.random.choice(len(unlabeled_data), n_samples, replace=False).tolist()
        elif self.strategy == "random":
            selected = np.random.choice(len(unlabeled_data), n_samples, replace=False).tolist()
        else:
            selected = ActiveLearningStrategy.uncertainty_sampling(
                self.model,
                unlabeled_data,
                n_samples
            )
        
        return selected
    
    def update(
        self,
        selected_indices: List[int],
        new_labels: torch.Tensor
    ):
        """
        Actualizar con nuevas muestras etiquetadas.
        
        Args:
            selected_indices: Índices seleccionados (relativos a unlabeled)
            new_labels: Labels de las nuevas muestras
        """
        # Convertir índices relativos a absolutos
        absolute_indices = [self.unlabeled_indices[i] for i in selected_indices]
        
        # Actualizar listas
        self.labeled_indices.extend(absolute_indices)
        for idx in sorted(absolute_indices, reverse=True):
            self.unlabeled_indices.remove(idx)
        
        self.iteration += 1
        logger.info(
            f"Iteration {self.iteration}: Added {len(selected_indices)} samples. "
            f"Total labeled: {len(self.labeled_indices)}"
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Obtener estadísticas del loop.
        
        Returns:
            Dict con estadísticas
        """
        return {
            "iteration": self.iteration,
            "labeled_samples": len(self.labeled_indices),
            "unlabeled_samples": len(self.unlabeled_indices),
            "strategy": self.strategy
        }

