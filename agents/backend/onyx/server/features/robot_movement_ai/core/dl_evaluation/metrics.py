"""
Evaluation Metrics
==================

Métricas para evaluación de modelos.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

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
    
    @abstractmethod
    def compute(self, predictions, targets) -> float:
        """
        Calcular métrica.
        
        Args:
            predictions: Predicciones
            targets: Objetivos
            
        Returns:
            Valor de la métrica
        """
        pass


class MSEMetric(Metric):
    """Mean Squared Error."""
    
    def __init__(self):
        super().__init__("MSE")
    
    def compute(self, predictions, targets) -> float:
        """Calcular MSE."""
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()
        
        return float(np.mean((predictions - targets) ** 2))


class MAEMetric(Metric):
    """Mean Absolute Error."""
    
    def __init__(self):
        super().__init__("MAE")
    
    def compute(self, predictions, targets) -> float:
        """Calcular MAE."""
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()
        
        return float(np.mean(np.abs(predictions - targets)))


class AccuracyMetric(Metric):
    """Accuracy para clasificación."""
    
    def __init__(self):
        super().__init__("Accuracy")
    
    def compute(self, predictions, targets) -> float:
        """Calcular accuracy."""
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()
        
        # Para clasificación, asumir que predictions son logits
        if len(predictions.shape) > 1 and predictions.shape[1] > 1:
            pred_classes = np.argmax(predictions, axis=1)
        else:
            pred_classes = (predictions > 0.5).astype(int).flatten()
        
        if len(targets.shape) > 1:
            target_classes = np.argmax(targets, axis=1) if targets.shape[1] > 1 else targets.flatten()
        else:
            target_classes = targets.flatten()
        
        return float(np.mean(pred_classes == target_classes))


class MetricsCalculator:
    """Calculadora de múltiples métricas."""
    
    def __init__(self, metrics: list):
        """
        Inicializar calculadora.
        
        Args:
            metrics: Lista de métricas
        """
        self.metrics = metrics
    
    def compute_all(self, predictions, targets) -> Dict[str, float]:
        """
        Calcular todas las métricas.
        
        Args:
            predictions: Predicciones
            targets: Objetivos
            
        Returns:
            Diccionario con métricas
        """
        results = {}
        for metric in self.metrics:
            try:
                results[metric.name] = metric.compute(predictions, targets)
            except Exception as e:
                logger.error(f"Error computing {metric.name}: {e}")
                results[metric.name] = float('nan')
        
        return results


