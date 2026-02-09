"""
Metrics - Modular Metrics Collection
====================================

Métricas modulares para evaluación de modelos.
"""

import logging
from typing import Dict, Any, List, Optional
import torch
import numpy as np
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class Metric(ABC):
    """Clase base para métricas."""
    
    def __init__(self, name: str):
        """
        Inicializar métrica.
        
        Args:
            name: Nombre de la métrica
        """
        self.name = name
        self.values: List[float] = []
    
    @abstractmethod
    def compute(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """Calcular métrica."""
        pass
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        """Actualizar métrica con nuevo batch."""
        value = self.compute(predictions, targets)
        self.values.append(value)
        return value
    
    def reset(self):
        """Resetear métrica."""
        self.values = []
    
    def mean(self) -> float:
        """Obtener promedio."""
        return np.mean(self.values) if self.values else 0.0
    
    def std(self) -> float:
        """Obtener desviación estándar."""
        return np.std(self.values) if self.values else 0.0


class MSE(Metric):
    """Mean Squared Error."""
    
    def __init__(self):
        super().__init__("MSE")
    
    def compute(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """Calcular MSE."""
        return torch.nn.functional.mse_loss(predictions, targets).item()


class MAE(Metric):
    """Mean Absolute Error."""
    
    def __init__(self):
        super().__init__("MAE")
    
    def compute(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """Calcular MAE."""
        return torch.nn.functional.l1_loss(predictions, targets).item()


class RMSE(Metric):
    """Root Mean Squared Error."""
    
    def __init__(self):
        super().__init__("RMSE")
    
    def compute(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """Calcular RMSE."""
        mse = torch.nn.functional.mse_loss(predictions, targets)
        return torch.sqrt(mse).item()


class R2Score(Metric):
    """R² Score."""
    
    def __init__(self):
        super().__init__("R2")
    
    def compute(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """Calcular R²."""
        ss_res = torch.sum((targets - predictions) ** 2)
        ss_tot = torch.sum((targets - targets.mean()) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        return r2.item() if ss_tot > 0 else 0.0


class Accuracy(Metric):
    """Accuracy para clasificación."""
    
    def __init__(self, top_k: int = 1):
        """
        Inicializar accuracy.
        
        Args:
            top_k: Top-k accuracy
        """
        super().__init__(f"Accuracy@{top_k}")
        self.top_k = top_k
    
    def compute(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """Calcular accuracy."""
        if predictions.dim() > 1:
            # Clasificación
            _, pred_classes = predictions.topk(self.top_k, dim=1)
            if targets.dim() > 1:
                targets = targets.argmax(dim=1)
            correct = pred_classes.eq(targets.view(-1, 1).expand_as(pred_classes))
            return correct.any(dim=1).float().mean().item()
        else:
            # Regresión binaria
            pred_binary = (predictions > 0.5).float()
            return (pred_binary == targets).float().mean().item()


class MetricCollection:
    """Colección de métricas."""
    
    def __init__(self, metrics: Optional[List[Metric]] = None):
        """
        Inicializar colección.
        
        Args:
            metrics: Lista de métricas
        """
        self.metrics = metrics or []
    
    def add_metric(self, metric: Metric):
        """Agregar métrica."""
        self.metrics.append(metric)
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        """Actualizar todas las métricas."""
        results = {}
        for metric in self.metrics:
            value = metric.update(predictions, targets)
            results[metric.name] = value
        return results
    
    def compute_all(self) -> Dict[str, float]:
        """Calcular todas las métricas."""
        results = {}
        for metric in self.metrics:
            results[metric.name] = metric.mean()
        return results
    
    def reset_all(self):
        """Resetear todas las métricas."""
        for metric in self.metrics:
            metric.reset()
    
    def __getitem__(self, name: str) -> Optional[Metric]:
        """Obtener métrica por nombre."""
        for metric in self.metrics:
            if metric.name == name:
                return metric
        return None


def create_regression_metrics() -> MetricCollection:
    """Crear métricas estándar para regresión."""
    return MetricCollection([
        MSE(),
        MAE(),
        RMSE(),
        R2Score()
    ])


def create_classification_metrics(top_k: int = 1) -> MetricCollection:
    """Crear métricas estándar para clasificación."""
    return MetricCollection([
        Accuracy(top_k=top_k)
    ])









