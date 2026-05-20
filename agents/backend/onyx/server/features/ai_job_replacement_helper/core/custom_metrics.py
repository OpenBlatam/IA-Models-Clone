"""
Custom Metrics - Métricas personalizadas
=========================================

Sistema para crear y gestionar métricas personalizadas.
Sigue mejores prácticas de métricas en deep learning.
"""

import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import numpy as np

logger = logging.getLogger(__name__)


class Metric(ABC):
    """Clase base para métricas"""
    
    def __init__(self, name: str):
        """
        Args:
            name: Nombre de la métrica
        """
        self.name = name
        self.reset()
    
    @abstractmethod
    def update(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        """
        Actualizar métrica con nuevas predicciones.
        
        Args:
            predictions: Predicciones del modelo
            targets: Valores objetivo
        """
        pass
    
    @abstractmethod
    def compute(self) -> float:
        """
        Calcular valor final de la métrica.
        
        Returns:
            Valor de la métrica
        """
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Resetear estado de la métrica"""
        pass


class Accuracy(Metric):
    """Métrica de accuracy"""
    
    def __init__(self, name: str = "accuracy"):
        super().__init__(name)
        self.correct = 0
        self.total = 0
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        """Actualizar accuracy"""
        if predictions.dim() > 1:
            pred_classes = predictions.argmax(dim=1)
        else:
            pred_classes = (predictions > 0.5).long()
        
        self.correct += (pred_classes == targets).sum().item()
        self.total += targets.size(0)
    
    def compute(self) -> float:
        """Calcular accuracy"""
        if self.total == 0:
            return 0.0
        return self.correct / self.total
    
    def reset(self) -> None:
        """Resetear"""
        self.correct = 0
        self.total = 0


class Precision(Metric):
    """Métrica de precision"""
    
    def __init__(self, name: str = "precision", average: str = "weighted"):
        """
        Args:
            name: Nombre de la métrica
            average: Tipo de promedio ('weighted', 'macro', 'micro', None)
        """
        super().__init__(name)
        self.average = average
        self.reset()
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        """Actualizar precision"""
        if predictions.dim() > 1:
            pred_classes = predictions.argmax(dim=1)
        else:
            pred_classes = (predictions > 0.5).long()
        
        # Store for batch computation
        if not hasattr(self, 'all_preds'):
            self.all_preds = []
            self.all_targets = []
        
        self.all_preds.append(pred_classes.cpu().numpy())
        self.all_targets.append(targets.cpu().numpy())
    
    def compute(self) -> float:
        """Calcular precision"""
        if not hasattr(self, 'all_preds') or len(self.all_preds) == 0:
            return 0.0
        
        try:
            from sklearn.metrics import precision_score
            all_preds = np.concatenate(self.all_preds)
            all_targets = np.concatenate(self.all_targets)
            return float(precision_score(all_targets, all_preds, average=self.average, zero_division=0))
        except ImportError:
            # Fallback: simple accuracy
            return float(np.mean(all_preds == all_targets))
    
    def reset(self) -> None:
        """Resetear"""
        if hasattr(self, 'all_preds'):
            self.all_preds = []
            self.all_targets = []


class Recall(Metric):
    """Métrica de recall"""
    
    def __init__(self, name: str = "recall", average: str = "weighted"):
        super().__init__(name)
        self.average = average
        self.reset()
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        """Actualizar recall"""
        if predictions.dim() > 1:
            pred_classes = predictions.argmax(dim=1)
        else:
            pred_classes = (predictions > 0.5).long()
        
        if not hasattr(self, 'all_preds'):
            self.all_preds = []
            self.all_targets = []
        
        self.all_preds.append(pred_classes.cpu().numpy())
        self.all_targets.append(targets.cpu().numpy())
    
    def compute(self) -> float:
        """Calcular recall"""
        if not hasattr(self, 'all_preds') or len(self.all_preds) == 0:
            return 0.0
        
        try:
            from sklearn.metrics import recall_score
            all_preds = np.concatenate(self.all_preds)
            all_targets = np.concatenate(self.all_targets)
            return float(recall_score(all_targets, all_preds, average=self.average, zero_division=0))
        except ImportError:
            return float(np.mean(all_preds == all_targets))
    
    def reset(self) -> None:
        """Resetear"""
        if hasattr(self, 'all_preds'):
            self.all_preds = []
            self.all_targets = []


class F1Score(Metric):
    """Métrica de F1 score"""
    
    def __init__(self, name: str = "f1_score", average: str = "weighted"):
        super().__init__(name)
        self.average = average
        self.reset()
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        """Actualizar F1"""
        if predictions.dim() > 1:
            pred_classes = predictions.argmax(dim=1)
        else:
            pred_classes = (predictions > 0.5).long()
        
        if not hasattr(self, 'all_preds'):
            self.all_preds = []
            self.all_targets = []
        
        self.all_preds.append(pred_classes.cpu().numpy())
        self.all_targets.append(targets.cpu().numpy())
    
    def compute(self) -> float:
        """Calcular F1"""
        if not hasattr(self, 'all_preds') or len(self.all_preds) == 0:
            return 0.0
        
        try:
            from sklearn.metrics import f1_score
            all_preds = np.concatenate(self.all_preds)
            all_targets = np.concatenate(self.all_targets)
            return float(f1_score(all_targets, all_preds, average=self.average, zero_division=0))
        except ImportError:
            return float(np.mean(all_preds == all_targets))
    
    def reset(self) -> None:
        """Resetear"""
        if hasattr(self, 'all_preds'):
            self.all_preds = []
            self.all_targets = []


class MeanSquaredError(Metric):
    """Métrica de Mean Squared Error"""
    
    def __init__(self, name: str = "mse"):
        super().__init__(name)
        self.reset()
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        """Actualizar MSE"""
        if not hasattr(self, 'all_preds'):
            self.all_preds = []
            self.all_targets = []
        
        self.all_preds.append(predictions.cpu().numpy())
        self.all_targets.append(targets.cpu().numpy())
    
    def compute(self) -> float:
        """Calcular MSE"""
        if not hasattr(self, 'all_preds') or len(self.all_preds) == 0:
            return 0.0
        
        all_preds = np.concatenate(self.all_preds)
        all_targets = np.concatenate(self.all_targets)
        return float(np.mean((all_preds - all_targets) ** 2))
    
    def reset(self) -> None:
        """Resetear"""
        if hasattr(self, 'all_preds'):
            self.all_preds = []
            self.all_targets = []


class MetricCollection:
    """Colección de métricas"""
    
    def __init__(self, metrics: Optional[List[Metric]] = None):
        """
        Args:
            metrics: Lista de métricas
        """
        self.metrics = metrics or []
    
    def add_metric(self, metric: Metric) -> None:
        """Agregar métrica"""
        self.metrics.append(metric)
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        """Actualizar todas las métricas"""
        for metric in self.metrics:
            metric.update(predictions, targets)
    
    def compute(self) -> Dict[str, float]:
        """
        Calcular todas las métricas.
        
        Returns:
            Diccionario con nombre y valor de cada métrica
        """
        results = {}
        for metric in self.metrics:
            results[metric.name] = metric.compute()
        return results
    
    def reset(self) -> None:
        """Resetear todas las métricas"""
        for metric in self.metrics:
            metric.reset()
    
    def __getitem__(self, name: str) -> Optional[Metric]:
        """Obtener métrica por nombre"""
        for metric in self.metrics:
            if metric.name == name:
                return metric
        return None




