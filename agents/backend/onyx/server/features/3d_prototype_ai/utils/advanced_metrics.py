"""
Advanced Evaluation Metrics - Métricas de evaluación avanzadas
===============================================================
Métricas especializadas para diferentes tareas
"""

import logging
import torch
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from sklearn.metrics import (
    roc_auc_score, average_precision_score, cohen_kappa_score,
    matthews_corrcoef, hamming_loss, jaccard_score,
    mean_absolute_percentage_error, mean_squared_log_error
)

logger = logging.getLogger(__name__)


class AdvancedMetrics:
    """Sistema de métricas avanzadas"""
    
    def __init__(self):
        self.metrics_history: Dict[str, List[float]] = {}
    
    def calculate_classification_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Calcula métricas avanzadas de clasificación"""
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            confusion_matrix, classification_report
        )
        
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
            "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
            "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
            "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
            "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
            "cohen_kappa": cohen_kappa_score(y_true, y_pred),
            "matthews_corrcoef": matthews_corrcoef(y_true, y_pred)
        }
        
        if y_proba is not None and len(np.unique(y_true)) == 2:
            # Binary classification
            try:
                metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
                metrics["average_precision"] = average_precision_score(y_true, y_proba)
            except Exception as e:
                logger.warning(f"Could not calculate ROC/AUC: {e}")
        
        return metrics
    
    def calculate_regression_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Calcula métricas avanzadas de regresión"""
        from sklearn.metrics import (
            mean_squared_error, mean_absolute_error, r2_score
        )
        
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        
        metrics = {
            "mse": mse,
            "mae": mae,
            "rmse": rmse,
            "r2_score": r2_score(y_true, y_pred),
            "mape": self._calculate_mape(y_true, y_pred),
            "msle": self._calculate_msle(y_true, y_pred)
        }
        
        return metrics
    
    def _calculate_mape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Absolute Percentage Error"""
        mask = y_true != 0
        if mask.sum() == 0:
            return 0.0
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    def _calculate_msle(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Squared Logarithmic Error"""
        y_true_log = np.log1p(np.maximum(y_true, 0))
        y_pred_log = np.log1p(np.maximum(y_pred, 0))
        return mean_squared_log_error(y_true_log, y_pred_log)
    
    def calculate_multilabel_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Calcula métricas para multi-label"""
        metrics = {
            "hamming_loss": hamming_loss(y_true, y_pred),
            "jaccard_score_macro": jaccard_score(y_true, y_pred, average="macro", zero_division=0),
            "jaccard_score_micro": jaccard_score(y_true, y_pred, average="micro", zero_division=0),
            "jaccard_score_weighted": jaccard_score(y_true, y_pred, average="weighted", zero_division=0)
        }
        
        return metrics
    
    def calculate_ranking_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        k: int = 10
    ) -> Dict[str, float]:
        """Calcula métricas de ranking"""
        # Precision@K, Recall@K, NDCG@K (simplificado)
        precision_at_k = []
        recall_at_k = []
        
        for i in range(len(y_true)):
            true_items = set(np.where(y_true[i] > 0)[0])
            pred_items = set(np.argsort(y_pred[i])[-k:])
            
            if len(pred_items) > 0:
                precision_at_k.append(len(true_items & pred_items) / len(pred_items))
            if len(true_items) > 0:
                recall_at_k.append(len(true_items & pred_items) / len(true_items))
        
        return {
            f"precision_at_{k}": np.mean(precision_at_k) if precision_at_k else 0.0,
            f"recall_at_{k}": np.mean(recall_at_k) if recall_at_k else 0.0
        }
    
    def calculate_custom_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        metric_name: str,
        **kwargs
    ) -> float:
        """Calcula métrica personalizada"""
        if metric_name == "top_k_accuracy":
            k = kwargs.get("k", 5)
            # Implementación simplificada
            return 0.0  # Placeholder
        
        elif metric_name == "focal_loss_metric":
            # Calcular focal loss como métrica
            alpha = kwargs.get("alpha", 1.0)
            gamma = kwargs.get("gamma", 2.0)
            # Implementación simplificada
            return 0.0  # Placeholder
        
        else:
            raise ValueError(f"Unknown custom metric: {metric_name}")




