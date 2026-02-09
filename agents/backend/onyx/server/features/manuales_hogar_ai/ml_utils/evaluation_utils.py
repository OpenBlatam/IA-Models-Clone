"""
Evaluation Utils - Utilidades de Evaluación
===========================================

Utilidades para evaluación de modelos y cálculo de métricas.
"""

import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, List, Optional, Callable
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """
    Calculadora de métricas para clasificación y regresión.
    """
    
    @staticmethod
    def calculate_classification_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        average: str = "weighted"
    ) -> Dict[str, float]:
        """
        Calcular métricas de clasificación.
        
        Args:
            y_true: Labels verdaderos
            y_pred: Predicciones
            average: Tipo de promedio para métricas
            
        Returns:
            Diccionario con métricas
        """
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
            "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
            "f1": f1_score(y_true, y_pred, average=average, zero_division=0)
        }
    
    @staticmethod
    def calculate_regression_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Calcular métricas de regresión.
        
        Args:
            y_true: Valores verdaderos
            y_pred: Predicciones
            
        Returns:
            Diccionario con métricas
        """
        mse = np.mean((y_true - y_pred) ** 2)
        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(mse)
        
        # R² score
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
        
        return {
            "mse": mse,
            "mae": mae,
            "rmse": rmse,
            "r2": r2
        }
    
    @staticmethod
    def get_confusion_matrix(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        labels: Optional[List[int]] = None
    ) -> np.ndarray:
        """
        Obtener matriz de confusión.
        
        Args:
            y_true: Labels verdaderos
            y_pred: Predicciones
            labels: Labels a incluir
            
        Returns:
            Matriz de confusión
        """
        return confusion_matrix(y_true, y_pred, labels=labels)
    
    @staticmethod
    def get_classification_report(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        target_names: Optional[List[str]] = None
    ) -> str:
        """
        Obtener reporte de clasificación.
        
        Args:
            y_true: Labels verdaderos
            y_pred: Predicciones
            target_names: Nombres de clases
            
        Returns:
            Reporte de clasificación
        """
        return classification_report(y_true, y_pred, target_names=target_names)


class ModelEvaluator:
    """
    Evaluador de modelos PyTorch.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Inicializar evaluador.
        
        Args:
            model: Modelo a evaluar
            device: Dispositivo
        """
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()
    
    def evaluate(
        self,
        data_loader: DataLoader,
        task_type: str = "classification"
    ) -> Dict[str, Any]:
        """
        Evaluar modelo en dataset.
        
        Args:
            data_loader: DataLoader
            task_type: Tipo de tarea (classification, regression)
            
        Returns:
            Diccionario con métricas
        """
        all_preds = []
        all_targets = []
        total_loss = 0.0
        
        criterion = nn.CrossEntropyLoss() if task_type == "classification" else nn.MSELoss()
        
        with torch.no_grad():
            for batch in data_loader:
                if isinstance(batch, (list, tuple)):
                    inputs, targets = batch[0].to(self.device), batch[1].to(self.device)
                else:
                    inputs = batch.to(self.device)
                    targets = None
                
                outputs = self.model(inputs)
                
                if targets is not None:
                    if task_type == "classification":
                        if outputs.dim() > 1:
                            loss = criterion(outputs, targets)
                            _, preds = torch.max(outputs, 1)
                        else:
                            loss = criterion(outputs.unsqueeze(0), targets.unsqueeze(0))
                            preds = (outputs > 0.5).long()
                    else:
                        loss = criterion(outputs, targets)
                        preds = outputs
                    
                    total_loss += loss.item()
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        
        # Calcular métricas
        if task_type == "classification":
            metrics = MetricsCalculator.calculate_classification_metrics(
                all_targets, all_preds
            )
        else:
            metrics = MetricsCalculator.calculate_regression_metrics(
                all_targets, all_preds
            )
        
        metrics["loss"] = total_loss / len(data_loader)
        
        return metrics
    
    def predict(
        self,
        data_loader: DataLoader,
        return_probs: bool = False
    ) -> np.ndarray:
        """
        Hacer predicciones.
        
        Args:
            data_loader: DataLoader
            return_probs: Si retornar probabilidades
            
        Returns:
            Array de predicciones
        """
        all_preds = []
        all_probs = []
        
        with torch.no_grad():
            for batch in data_loader:
                if isinstance(batch, (list, tuple)):
                    inputs = batch[0].to(self.device)
                else:
                    inputs = batch.to(self.device)
                
                outputs = self.model(inputs)
                
                if return_probs and outputs.dim() > 1:
                    probs = torch.softmax(outputs, dim=1)
                    all_probs.extend(probs.cpu().numpy())
                
                if outputs.dim() > 1:
                    _, preds = torch.max(outputs, 1)
                else:
                    preds = (outputs > 0.5).long()
                
                all_preds.extend(preds.cpu().numpy())
        
        if return_probs:
            return np.array(all_preds), np.array(all_probs)
        return np.array(all_preds)




