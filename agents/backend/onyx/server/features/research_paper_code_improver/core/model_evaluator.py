"""
Model Evaluation Framework - Framework de evaluación de modelos
================================================================
"""

import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, mean_squared_error, mean_absolute_error
)

from .base_classes import BaseEvaluator, BaseConfig
from .common_utils import get_device, move_to_device, get_model_output, extract_predictions, calculate_accuracy

logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetrics:
    """Métricas de evaluación"""
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1: Optional[float] = None
    roc_auc: Optional[float] = None
    mse: Optional[float] = None
    mae: Optional[float] = None
    loss: Optional[float] = None
    custom_metrics: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario"""
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "roc_auc": self.roc_auc,
            "mse": self.mse,
            "mae": self.mae,
            "loss": self.loss,
            "custom_metrics": self.custom_metrics
        }


class ModelEvaluator(BaseEvaluator):
    """Evaluador de modelos"""
    
    def __init__(self, device: Optional[str] = None):
        super().__init__(BaseConfig())
        self.device = get_device(device)
    
    def evaluate_classification(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        loss_fn: Optional[Callable] = None
    ) -> EvaluationMetrics:
        """Evalúa modelo de clasificación"""
        model.eval()
        all_predictions = []
        all_labels = []
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in data_loader:
                # Mover batch a device usando utilidades compartidas
                batch = move_to_device(batch, self.device)
                
                # Obtener labels
                if isinstance(batch, dict):
                    labels = batch.get("labels")
                elif isinstance(batch, tuple):
                    labels = batch[1]
                else:
                    labels = None
                
                # Forward pass usando utilidades compartidas
                outputs = get_model_output(model, batch, str(self.device))
                
                # Obtener predicciones usando utilidades compartidas
                predictions = extract_predictions(outputs)
                all_predictions.extend(predictions.cpu().numpy())
                
                if labels is not None:
                    all_labels.extend(labels.cpu().numpy())
                
                # Calcular pérdida
                if loss_fn:
                    loss = loss_fn(logits, labels)
                    total_loss += loss.item()
                    num_batches += 1
        
        # Calcular métricas
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        
        metrics = EvaluationMetrics()
        metrics.accuracy = accuracy_score(all_labels, all_predictions)
        metrics.precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
        metrics.recall = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)
        metrics.f1 = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)
        
        if num_batches > 0:
            metrics.loss = total_loss / num_batches
        
        return metrics
    
    def evaluate_regression(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        loss_fn: Optional[Callable] = None
    ) -> EvaluationMetrics:
        """Evalúa modelo de regresión"""
        model.eval()
        all_predictions = []
        all_labels = []
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in data_loader:
                inputs = batch.get("input_ids") or batch.get("inputs")
                labels = batch.get("labels")
                
                if isinstance(inputs, torch.Tensor):
                    inputs = inputs.to(self.device)
                if isinstance(labels, torch.Tensor):
                    labels = labels.to(self.device)
                
                if isinstance(batch, dict):
                    outputs = model(**{k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                                     for k, v in batch.items()})
                else:
                    outputs = model(inputs)
                
                if isinstance(outputs, torch.Tensor):
                    predictions = outputs.squeeze()
                elif hasattr(outputs, 'logits'):
                    predictions = outputs.logits.squeeze()
                else:
                    predictions = outputs
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                if loss_fn:
                    loss = loss_fn(predictions, labels)
                    total_loss += loss.item()
                    num_batches += 1
        
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        
        metrics = EvaluationMetrics()
        metrics.mse = mean_squared_error(all_labels, all_predictions)
        metrics.mae = mean_absolute_error(all_labels, all_predictions)
        
        if num_batches > 0:
            metrics.loss = total_loss / num_batches
        
        return metrics
    
    def evaluate_custom(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        eval_fn: Callable
    ) -> EvaluationMetrics:
        """Evalúa con función personalizada"""
        model.eval()
        all_outputs = []
        all_labels = []
        
        with torch.no_grad():
            for batch in data_loader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                outputs = model(**batch)
                all_outputs.append(outputs)
                all_labels.append(batch.get("labels"))
        
        # Evaluar con función personalizada
        custom_metrics = eval_fn(all_outputs, all_labels)
        
        metrics = EvaluationMetrics()
        metrics.custom_metrics = custom_metrics
        
        return metrics

