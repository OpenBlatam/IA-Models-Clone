"""
Validation Utilities
====================
Robust validation and testing utilities
"""

from typing import Dict, Any, List, Optional, Callable
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import structlog
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

logger = structlog.get_logger()


class ModelValidator:
    """
    Comprehensive model validator
    """
    
    def __init__(self, model: nn.Module, device: Optional[torch.device] = None):
        """
        Initialize validator
        
        Args:
            model: Model to validate
            device: Device
        """
        self.model = model
        from .deep_learning_models import get_device
        self.device = device or get_device()
        self.model = self.model.to(self.device)
        self.model.eval()
    
    def validate_on_dataset(
        self,
        data_loader: DataLoader,
        metrics: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Validate model on dataset
        
        Args:
            data_loader: Data loader
            metrics: List of metrics to compute
            
        Returns:
            Validation results
        """
        if metrics is None:
            metrics = ["accuracy", "precision", "recall", "f1"]
        
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in data_loader:
                try:
                    batch = self._move_to_device(batch)
                    outputs = self.model(**batch)
                    
                    if isinstance(outputs, dict):
                        preds = outputs.get("logits", outputs.get("predictions"))
                    else:
                        preds = outputs
                    
                    predictions = torch.argmax(preds, dim=-1) if preds.dim() > 1 else preds
                    labels = batch.get("labels")
                    
                    all_predictions.extend(predictions.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    
                except Exception as e:
                    logger.error("Error in validation", error=str(e))
                    continue
        
        # Calculate metrics
        results = {}
        for metric in metrics:
            if metric == "accuracy":
                results[metric] = self._calculate_accuracy(all_labels, all_predictions)
            elif metric == "precision":
                results[metric] = self._calculate_precision(all_labels, all_predictions)
            elif metric == "recall":
                results[metric] = self._calculate_recall(all_labels, all_predictions)
            elif metric == "f1":
                results[metric] = self._calculate_f1(all_labels, all_predictions)
        
        return results
    
    def _calculate_accuracy(self, labels: List, predictions: List) -> float:
        """Calculate accuracy"""
        correct = sum(1 for l, p in zip(labels, predictions) if l == p)
        return correct / len(labels) if labels else 0.0
    
    def _calculate_precision(self, labels: List, predictions: List) -> float:
        """Calculate precision"""
        cm = confusion_matrix(labels, predictions)
        return np.diag(cm) / cm.sum(axis=0)
    
    def _calculate_recall(self, labels: List, predictions: List) -> float:
        """Calculate recall"""
        cm = confusion_matrix(labels, predictions)
        return np.diag(cm) / cm.sum(axis=1)
    
    def _calculate_f1(self, labels: List, predictions: List) -> float:
        """Calculate F1 score"""
        precision = self._calculate_precision(labels, predictions)
        recall = self._calculate_recall(labels, predictions)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        return f1.mean()
    
    def _move_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move batch to device"""
        return {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }


class GradientValidator:
    """Validate gradients during training"""
    
    @staticmethod
    def validate_gradients(
        model: nn.Module,
        max_norm: float = 10.0
    ) -> Dict[str, Any]:
        """
        Validate gradients
        
        Args:
            model: Model
            max_norm: Maximum gradient norm
            
        Returns:
            Validation results
        """
        total_norm = 0.0
        param_count = 0
        has_nan = False
        has_inf = False
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1
                
                if torch.isnan(param.grad).any():
                    has_nan = True
                if torch.isinf(param.grad).any():
                    has_inf = True
        
        total_norm = total_norm ** (1. / 2)
        is_exploding = total_norm > max_norm
        
        return {
            "total_norm": total_norm,
            "param_count": param_count,
            "has_nan": has_nan,
            "has_inf": has_inf,
            "is_exploding": is_exploding,
            "is_valid": not (has_nan or has_inf or is_exploding)
        }




