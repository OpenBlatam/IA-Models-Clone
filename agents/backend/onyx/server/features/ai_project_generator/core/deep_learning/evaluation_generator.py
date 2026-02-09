"""
Evaluation Generator - Generador de utilidades de evaluación
=============================================================

Genera módulos especializados para evaluación:
- Métricas estándar
- Funciones de evaluación
- Visualización de resultados
"""

import logging
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)


class EvaluationGenerator:
    """Generador de utilidades de evaluación"""
    
    def __init__(self):
        """Inicializa el generador de evaluación"""
        pass
    
    def generate(
        self,
        utils_dir: Path,
        keywords: Dict[str, Any],
        project_info: Dict[str, Any],
    ) -> None:
        """
        Genera utilidades de evaluación.
        
        Args:
            utils_dir: Directorio donde generar las utilidades
            keywords: Keywords extraídos
            project_info: Información del proyecto
        """
        utils_dir.mkdir(parents=True, exist_ok=True)
        
        # Generar módulos de evaluación
        self._generate_metrics(utils_dir, keywords, project_info)
        self._generate_evaluator(utils_dir, keywords, project_info)
        self._generate_evaluation_init(utils_dir, keywords)
    
    def _generate_evaluation_init(
        self,
        utils_dir: Path,
        keywords: Dict[str, Any],
    ) -> None:
        """Genera __init__.py del módulo de evaluación"""
        
        init_content = '''"""
Evaluation Utilities Module
============================

Utilidades para evaluación de modelos.
"""

from .metrics import calculate_metrics, ClassificationMetrics, RegressionMetrics
from .evaluator import evaluate_model

__all__ = [
    "calculate_metrics",
    "ClassificationMetrics",
    "RegressionMetrics",
    "evaluate_model",
]
'''
        
        eval_dir = utils_dir / "evaluation"
        eval_dir.mkdir(parents=True, exist_ok=True)
        (eval_dir / "__init__.py").write_text(init_content, encoding="utf-8")
    
    def _generate_metrics(
        self,
        utils_dir: Path,
        keywords: Dict[str, Any],
        project_info: Dict[str, Any],
    ) -> None:
        """Genera utilidades de métricas"""
        
        metrics_content = '''"""
Metrics - Métricas de evaluación
==================================

Implementa métricas estándar para diferentes tareas.
"""

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
)
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)


class ClassificationMetrics:
    """Métricas para tareas de clasificación"""
    
    @staticmethod
    def calculate(
        predictions: np.ndarray,
        labels: np.ndarray,
        average: str = "weighted",
    ) -> Dict[str, float]:
        """
        Calcula métricas de clasificación.
        
        Args:
            predictions: Predicciones del modelo
            labels: Etiquetas verdaderas
            average: Tipo de promedio (weighted, macro, micro)
            
        Returns:
            Diccionario con métricas
        """
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels,
            predictions,
            average=average,
            zero_division=0
        )
        
        return {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }
    
    @staticmethod
    def confusion_matrix(
        predictions: np.ndarray,
        labels: np.ndarray,
    ) -> np.ndarray:
        """Calcula matriz de confusión"""
        return confusion_matrix(labels, predictions)


class RegressionMetrics:
    """Métricas para tareas de regresión"""
    
    @staticmethod
    def calculate(
        predictions: np.ndarray,
        labels: np.ndarray,
    ) -> Dict[str, float]:
        """
        Calcula métricas de regresión.
        
        Args:
            predictions: Predicciones del modelo
            labels: Valores verdaderos
            
        Returns:
            Diccionario con métricas
        """
        mse = mean_squared_error(labels, predictions)
        mae = mean_absolute_error(labels, predictions)
        rmse = np.sqrt(mse)
        
        return {
            "mse": float(mse),
            "mae": float(mae),
            "rmse": float(rmse),
        }


def calculate_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    task_type: str = "classification",
    **kwargs,
) -> Dict[str, float]:
    """
    Calcula métricas de evaluación según el tipo de tarea.
    
    Args:
        predictions: Predicciones del modelo
        labels: Etiquetas verdaderas
        task_type: Tipo de tarea (classification, regression)
        **kwargs: Argumentos adicionales
        
    Returns:
        Diccionario con métricas
    """
    if task_type == "classification":
        return ClassificationMetrics.calculate(
            predictions,
            labels,
            average=kwargs.get("average", "weighted")
        )
    elif task_type == "regression":
        return RegressionMetrics.calculate(predictions, labels)
    else:
        raise ValueError(f"Task type {task_type} no soportado")
'''
        
        eval_dir = utils_dir / "evaluation"
        (eval_dir / "metrics.py").write_text(metrics_content, encoding="utf-8")
    
    def _generate_evaluator(
        self,
        utils_dir: Path,
        keywords: Dict[str, Any],
        project_info: Dict[str, Any],
    ) -> None:
        """Genera función de evaluación"""
        
        evaluator_content = '''"""
Model Evaluator - Evaluador de modelos
========================================

Evalúa modelos en datasets completos.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional
import logging

from .metrics import calculate_metrics

logger = logging.getLogger(__name__)


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = "cuda",
    criterion: Optional[nn.Module] = None,
    task_type: str = "classification",
) -> Dict[str, float]:
    """
    Evalúa un modelo en un dataset.
    
    Args:
        model: Modelo a evaluar
        dataloader: DataLoader con datos de evaluación
        device: Dispositivo a usar
        criterion: Función de pérdida (opcional)
        task_type: Tipo de tarea (classification, regression)
        
    Returns:
        Diccionario con métricas
    """
    model.eval()
    all_predictions = []
    all_labels = []
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in dataloader:
            # Mover batch a dispositivo
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            outputs = model(**batch)
            
            if criterion:
                loss = criterion(outputs, batch.get("labels"))
                total_loss += loss.item()
            
            # Obtener predicciones
            if hasattr(outputs, "logits"):
                predictions = torch.argmax(outputs.logits, dim=-1)
            elif isinstance(outputs, torch.Tensor):
                predictions = torch.argmax(outputs, dim=-1) if len(outputs.shape) > 1 else outputs
            else:
                predictions = outputs
            
            all_predictions.extend(predictions.cpu().numpy())
            if "labels" in batch:
                all_labels.extend(batch["labels"].cpu().numpy())
    
    metrics = {}
    if criterion:
        metrics["loss"] = total_loss / len(dataloader)
    
    if all_labels:
        import numpy as np
        metrics.update(calculate_metrics(
            np.array(all_predictions),
            np.array(all_labels),
            task_type=task_type
        ))
    
    logger.info(f"Métricas de evaluación: {metrics}")
    return metrics
'''
        
        eval_dir = utils_dir / "evaluation"
        (eval_dir / "evaluator.py").write_text(evaluator_content, encoding="utf-8")

