"""
Active Learning Service - Aprendizaje activo
==============================================

Sistema para aprendizaje activo y selección inteligente de muestras.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import numpy as np

logger = logging.getLogger(__name__)

# Try to import PyTorch
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available")


class SamplingStrategy(str):
    """Estrategias de muestreo"""
    UNCERTAINTY = "uncertainty"
    DIVERSITY = "diversity"
    REPRESENTATIVE = "representative"
    HYBRID = "hybrid"


@dataclass
class ActiveLearningConfig:
    """Configuración de aprendizaje activo"""
    strategy: SamplingStrategy = SamplingStrategy.UNCERTAINTY
    num_samples: int = 100
    batch_size: int = 10
    use_entropy: bool = True
    diversity_weight: float = 0.5


class ActiveLearningService:
    """Servicio de aprendizaje activo"""
    
    def __init__(self):
        """Inicializar servicio"""
        self.labeled_pool: List[Any] = []
        self.unlabeled_pool: List[Any] = []
        self.selected_samples: List[Any] = []
        logger.info("ActiveLearningService initialized")
    
    def uncertainty_sampling(
        self,
        model: nn.Module,
        unlabeled_data: List[Any],
        num_samples: int,
        device: Optional[torch.device] = None
    ) -> List[int]:
        """
        Seleccionar muestras con mayor incertidumbre.
        
        Args:
            model: Modelo entrenado
            unlabeled_data: Datos sin etiquetar
            num_samples: Número de muestras a seleccionar
            device: Dispositivo
        
        Returns:
            Índices de muestras seleccionadas
        """
        if not TORCH_AVAILABLE:
            return list(range(min(num_samples, len(unlabeled_data))))
        
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        model.eval()
        uncertainties = []
        
        with torch.no_grad():
            for i, sample in enumerate(unlabeled_data):
                if isinstance(sample, torch.Tensor):
                    inputs = sample.unsqueeze(0).to(device)
                else:
                    inputs = torch.tensor(sample).unsqueeze(0).to(device)
                
                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1)
                
                # Calcular entropía (incertidumbre)
                entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
                uncertainties.append(entropy.item())
        
        # Seleccionar muestras con mayor incertidumbre
        indices = np.argsort(uncertainties)[::-1][:num_samples]
        return indices.tolist()
    
    def diversity_sampling(
        self,
        unlabeled_data: List[Any],
        num_samples: int,
        labeled_data: Optional[List[Any]] = None
    ) -> List[int]:
        """
        Seleccionar muestras diversas.
        
        Args:
            unlabeled_data: Datos sin etiquetar
            num_samples: Número de muestras
            labeled_data: Datos ya etiquetados (opcional)
        
        Returns:
            Índices de muestras seleccionadas
        """
        # K-means clustering simplificado
        # En producción, usaría scikit-learn
        
        selected = []
        remaining_indices = list(range(len(unlabeled_data)))
        
        # Primera muestra aleatoria
        if remaining_indices:
            first_idx = np.random.choice(remaining_indices)
            selected.append(first_idx)
            remaining_indices.remove(first_idx)
        
        # Seleccionar muestras lejanas a las ya seleccionadas
        for _ in range(min(num_samples - 1, len(remaining_indices))):
            if not remaining_indices:
                break
            
            # Calcular distancias (simplificado)
            # En producción, calcularía distancias reales
            next_idx = np.random.choice(remaining_indices)
            selected.append(next_idx)
            remaining_indices.remove(next_idx)
        
        return selected
    
    def hybrid_sampling(
        self,
        model: nn.Module,
        unlabeled_data: List[Any],
        num_samples: int,
        uncertainty_weight: float = 0.7,
        device: Optional[torch.device] = None
    ) -> List[int]:
        """
        Selección híbrida: incertidumbre + diversidad.
        
        Args:
            model: Modelo
            unlabeled_data: Datos sin etiquetar
            num_samples: Número de muestras
            uncertainty_weight: Peso de incertidumbre vs diversidad
            device: Dispositivo
        
        Returns:
            Índices de muestras seleccionadas
        """
        # Combinar estrategias
        uncertainty_indices = self.uncertainty_sampling(
            model, unlabeled_data, num_samples * 2, device
        )
        diversity_indices = self.diversity_sampling(
            unlabeled_data, num_samples * 2
        )
        
        # Combinar y seleccionar
        combined = list(set(uncertainty_indices + diversity_indices))
        return combined[:num_samples]
    
    def select_samples(
        self,
        model: nn.Module,
        unlabeled_data: List[Any],
        config: ActiveLearningConfig,
        device: Optional[torch.device] = None
    ) -> List[int]:
        """
        Seleccionar muestras usando estrategia configurada.
        
        Args:
            model: Modelo
            unlabeled_data: Datos sin etiquetar
            config: Configuración
            device: Dispositivo
        
        Returns:
            Índices de muestras seleccionadas
        """
        if config.strategy == SamplingStrategy.UNCERTAINTY:
            return self.uncertainty_sampling(
                model, unlabeled_data, config.num_samples, device
            )
        elif config.strategy == SamplingStrategy.DIVERSITY:
            return self.diversity_sampling(
                unlabeled_data, config.num_samples, self.labeled_pool
            )
        elif config.strategy == SamplingStrategy.HYBRID:
            return self.hybrid_sampling(
                model, unlabeled_data, config.num_samples,
                uncertainty_weight=1.0 - config.diversity_weight,
                device=device
            )
        else:
            # Representative sampling (simplificado)
            return list(range(min(config.num_samples, len(unlabeled_data))))




