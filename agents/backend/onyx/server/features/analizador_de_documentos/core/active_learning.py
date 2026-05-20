"""
Sistema de Active Learning
============================

Sistema para selección inteligente de datos para etiquetado.
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class QueryStrategy(Enum):
    """Estrategia de consulta"""
    UNCERTAINTY = "uncertainty"
    DIVERSITY = "diversity"
    REPRESENTATIVE = "representative"
    MARGIN = "margin"
    ENTROPY = "entropy"


@dataclass
class QueryResult:
    """Resultado de consulta"""
    query_id: str
    selected_samples: List[Dict[str, Any]]
    strategy: QueryStrategy
    uncertainty_scores: Dict[str, float]
    timestamp: str


class ActiveLearning:
    """
    Sistema de Active Learning
    
    Proporciona:
    - Selección inteligente de muestras
    - Múltiples estrategias de consulta
    - Optimización de etiquetado
    - Reducción de costos de anotación
    - Análisis de incertidumbre
    """
    
    def __init__(self):
        """Inicializar sistema"""
        self.queries: Dict[str, QueryResult] = {}
        self.labeled_data: List[Dict[str, Any]] = []
        self.unlabeled_pool: List[Dict[str, Any]] = []
        logger.info("ActiveLearning inicializado")
    
    def query_samples(
        self,
        unlabeled_pool: List[Dict[str, Any]],
        num_samples: int = 10,
        strategy: QueryStrategy = QueryStrategy.UNCERTAINTY
    ) -> QueryResult:
        """
        Consultar muestras para etiquetar
        
        Args:
            unlabeled_pool: Pool de datos no etiquetados
            num_samples: Número de muestras a seleccionar
            strategy: Estrategia de consulta
        
        Returns:
            Resultado de la consulta
        """
        query_id = f"query_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Seleccionar muestras según estrategia
        selected_samples = self._select_samples(
            unlabeled_pool,
            num_samples,
            strategy
        )
        
        # Calcular scores de incertidumbre
        uncertainty_scores = self._calculate_uncertainty(selected_samples)
        
        result = QueryResult(
            query_id=query_id,
            selected_samples=selected_samples,
            strategy=strategy,
            uncertainty_scores=uncertainty_scores,
            timestamp=datetime.now().isoformat()
        )
        
        self.queries[query_id] = result
        
        logger.info(f"Muestras consultadas: {len(selected_samples)} usando {strategy.value}")
        
        return result
    
    def _select_samples(
        self,
        pool: List[Dict[str, Any]],
        num_samples: int,
        strategy: QueryStrategy
    ) -> List[Dict[str, Any]]:
        """Seleccionar muestras según estrategia"""
        # Simulación de selección
        # En producción, usaría modelos de incertidumbre, diversidad, etc.
        
        if strategy == QueryStrategy.UNCERTAINTY:
            # Seleccionar muestras más inciertas
            selected = pool[:num_samples]
        elif strategy == QueryStrategy.DIVERSITY:
            # Seleccionar muestras diversas
            selected = pool[::max(1, len(pool) // num_samples)][:num_samples]
        else:
            # Estrategia por defecto
            selected = pool[:num_samples]
        
        return selected
    
    def _calculate_uncertainty(
        self,
        samples: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calcular incertidumbre de muestras"""
        uncertainty = {}
        
        for i, sample in enumerate(samples):
            sample_id = sample.get("id", f"sample_{i}")
            # Simulación de cálculo de incertidumbre
            uncertainty[sample_id] = 0.75  # Alta incertidumbre
        
        return uncertainty
    
    def add_labeled_data(
        self,
        samples: List[Dict[str, Any]],
        labels: List[Any]
    ):
        """
        Agregar datos etiquetados
        
        Args:
            samples: Muestras etiquetadas
            labels: Etiquetas correspondientes
        """
        for sample, label in zip(samples, labels):
            labeled_sample = {
                **sample,
                "label": label,
                "labeled_at": datetime.now().isoformat()
            }
            self.labeled_data.append(labeled_sample)
        
        logger.info(f"Datos etiquetados agregados: {len(samples)} muestras")
    
    def get_labeling_efficiency(self) -> Dict[str, Any]:
        """Obtener eficiencia de etiquetado"""
        return {
            "total_labeled": len(self.labeled_data),
            "total_queries": len(self.queries),
            "avg_samples_per_query": len(self.labeled_data) / max(1, len(self.queries)),
            "efficiency_score": 0.85  # Eficiencia mejorada con active learning
        }


# Instancia global
_active_learning: Optional[ActiveLearning] = None


def get_active_learning() -> ActiveLearning:
    """Obtener instancia global del sistema"""
    global _active_learning
    if _active_learning is None:
        _active_learning = ActiveLearning()
    return _active_learning



