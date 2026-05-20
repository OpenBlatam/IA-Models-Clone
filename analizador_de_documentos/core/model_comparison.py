"""
Sistema de Model Comparison
============================

Sistema para comparación de modelos.
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class ComparisonMetric(Enum):
    """Métrica de comparación"""
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    ROC_AUC = "roc_auc"
    LATENCY = "latency"
    MEMORY_USAGE = "memory_usage"
    MODEL_SIZE = "model_size"


@dataclass
class ModelComparison:
    """Comparación de modelos"""
    comparison_id: str
    model_ids: List[str]
    metrics: Dict[str, Dict[str, float]]
    winner: str
    timestamp: str


class ModelComparisonSystem:
    """
    Sistema de Model Comparison
    
    Proporciona:
    - Comparación de modelos
    - Múltiples métricas de comparación
    - Análisis de trade-offs
    - Recomendaciones de modelos
    - Visualización de comparaciones
    """
    
    def __init__(self):
        """Inicializar sistema"""
        self.comparisons: Dict[str, ModelComparison] = {}
        logger.info("ModelComparisonSystem inicializado")
    
    def compare_models(
        self,
        model_ids: List[str],
        metrics: Optional[List[ComparisonMetric]] = None
    ) -> ModelComparison:
        """
        Comparar modelos
        
        Args:
            model_ids: IDs de modelos
            metrics: Métricas a comparar
        
        Returns:
            Comparación
        """
        if metrics is None:
            metrics = [
                ComparisonMetric.ACCURACY,
                ComparisonMetric.LATENCY,
                ComparisonMetric.MODEL_SIZE
            ]
        
        comparison_id = f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Simulación de comparación
        metrics_data = {}
        for metric in metrics:
            metrics_data[metric.value] = {
                model_id: 0.85 + (i * 0.05) if metric != ComparisonMetric.LATENCY else 50.0 - (i * 5.0)
                for i, model_id in enumerate(model_ids)
            }
        
        # Determinar ganador (mayor accuracy)
        winner = max(model_ids, key=lambda m: metrics_data.get("accuracy", {}).get(m, 0.0))
        
        comparison = ModelComparison(
            comparison_id=comparison_id,
            model_ids=model_ids,
            metrics=metrics_data,
            winner=winner,
            timestamp=datetime.now().isoformat()
        )
        
        self.comparisons[comparison_id] = comparison
        
        logger.info(f"Comparación completada: {comparison_id} - Ganador: {winner}")
        
        return comparison


# Instancia global
_model_comparison: Optional[ModelComparisonSystem] = None


def get_model_comparison() -> ModelComparisonSystem:
    """Obtener instancia global del sistema"""
    global _model_comparison
    if _model_comparison is None:
        _model_comparison = ModelComparisonSystem()
    return _model_comparison


