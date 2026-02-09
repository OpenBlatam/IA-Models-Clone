"""
Model Comparison
================

Comparación avanzada de múltiples modelos.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np
import logging

from ..routing_evaluation.evaluator import ModelEvaluator, EvaluationMetrics

logger = logging.getLogger(__name__)


@dataclass
class ModelComparisonResult:
    """Resultado de comparación de modelos."""
    model_name: str
    metrics: EvaluationMetrics
    rank: int
    score: float


class ModelComparator:
    """
    Comparador de modelos.
    """
    
    def __init__(self, evaluator: Optional[ModelEvaluator] = None):
        """
        Inicializar comparador.
        
        Args:
            evaluator: Evaluador (opcional)
        """
        self.evaluator = evaluator or ModelEvaluator()
        self.comparison_results = []
    
    def compare_models(
        self,
        models: Dict[str, nn.Module],
        dataloader: torch.utils.data.DataLoader,
        metric: str = "r2"  # "r2", "mse", "mae"
    ) -> List[ModelComparisonResult]:
        """
        Comparar múltiples modelos.
        
        Args:
            models: Diccionario {nombre: modelo}
            dataloader: DataLoader de evaluación
            metric: Métrica para ranking
            
        Returns:
            Lista de resultados ordenados
        """
        results = []
        
        for name, model in models.items():
            logger.info(f"Evaluando modelo: {name}")
            metrics = self.evaluator.evaluate(model, dataloader)
            
            # Score para ranking
            if metric == "r2":
                score = metrics.r2
            elif metric == "mse":
                score = -metrics.mse  # Negativo porque menor es mejor
            elif metric == "mae":
                score = -metrics.mae
            else:
                score = metrics.r2
            
            results.append(ModelComparisonResult(
                model_name=name,
                metrics=metrics,
                rank=0,  # Se asignará después
                score=score
            ))
        
        # Ordenar por score
        results.sort(key=lambda x: x.score, reverse=True)
        
        # Asignar ranks
        for i, result in enumerate(results):
            result.rank = i + 1
        
        self.comparison_results = results
        return results
    
    def get_best_model(self) -> Optional[ModelComparisonResult]:
        """
        Obtener mejor modelo.
        
        Returns:
            Resultado del mejor modelo
        """
        if self.comparison_results:
            return self.comparison_results[0]
        return None
    
    def get_comparison_summary(self) -> Dict[str, Any]:
        """
        Obtener resumen de comparación.
        
        Returns:
            Resumen
        """
        if not self.comparison_results:
            return {}
        
        return {
            "best_model": self.comparison_results[0].model_name,
            "best_score": self.comparison_results[0].score,
            "num_models": len(self.comparison_results),
            "score_range": (
                self.comparison_results[-1].score,
                self.comparison_results[0].score
            ),
            "models": [
                {
                    "name": r.model_name,
                    "rank": r.rank,
                    "r2": r.metrics.r2,
                    "mse": r.metrics.mse,
                    "mae": r.metrics.mae
                }
                for r in self.comparison_results
            ]
        }


def compare_models(
    models: Dict[str, nn.Module],
    dataloader: torch.utils.data.DataLoader,
    metric: str = "r2"
) -> List[ModelComparisonResult]:
    """
    Función helper para comparar modelos.
    
    Args:
        models: Diccionario de modelos
        dataloader: DataLoader
        metric: Métrica para ranking
        
    Returns:
        Resultados ordenados
    """
    comparator = ModelComparator()
    return comparator.compare_models(models, dataloader, metric)

