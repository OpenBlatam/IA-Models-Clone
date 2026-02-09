"""
Evaluación comprehensiva con múltiples métricas
"""

import torch
import numpy as np
from typing import Dict, Any, List, Optional
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
import logging

logger = logging.getLogger(__name__)


class ComprehensiveEvaluator:
    """Evaluador comprehensivo"""
    
    def __init__(self):
        pass
    
    def evaluate_classification_comprehensive(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        class_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Evaluación comprehensiva de clasificación
        
        Args:
            predictions: Predicciones
            labels: Labels verdaderos
            class_names: Nombres de clases
            
        Returns:
            Métricas completas
        """
        # Métricas básicas
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, support = precision_recall_fscore_support(
            labels, predictions, average=None, zero_division=0
        )
        
        # Métricas promedio
        precision_macro = precision.mean()
        recall_macro = recall.mean()
        f1_macro = f1.mean()
        
        precision_weighted = np.average(precision, weights=support)
        recall_weighted = np.average(recall, weights=support)
        f1_weighted = np.average(f1, weights=support)
        
        # Confusion matrix
        cm = confusion_matrix(labels, predictions)
        
        # Classification report
        report = classification_report(
            labels, predictions,
            target_names=class_names,
            output_dict=True,
            zero_division=0
        )
        
        # ROC AUC (si es binario o multi-class)
        try:
            if len(np.unique(labels)) == 2:
                # Binary classification
                roc_auc = roc_auc_score(labels, predictions)
            else:
                # Multi-class: usar one-vs-rest
                roc_auc = roc_auc_score(labels, predictions, multi_class='ovr', average='macro')
        except:
            roc_auc = None
        
        return {
            "accuracy": float(accuracy),
            "precision": {
                "macro": float(precision_macro),
                "weighted": float(precision_weighted),
                "per_class": precision.tolist()
            },
            "recall": {
                "macro": float(recall_macro),
                "weighted": float(recall_weighted),
                "per_class": recall.tolist()
            },
            "f1": {
                "macro": float(f1_macro),
                "weighted": float(f1_weighted),
                "per_class": f1.tolist()
            },
            "support": support.tolist(),
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
            "roc_auc": float(roc_auc) if roc_auc is not None else None
        }
    
    def evaluate_regression_comprehensive(
        self,
        predictions: np.ndarray,
        labels: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluación comprehensiva de regresión
        
        Args:
            predictions: Predicciones
            labels: Labels verdaderos
            
        Returns:
            Métricas completas
        """
        from sklearn.metrics import (
            mean_squared_error,
            mean_absolute_error,
            r2_score,
            mean_absolute_percentage_error
        )
        
        mse = mean_squared_error(labels, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(labels, predictions)
        r2 = r2_score(labels, predictions)
        
        # MAPE (evitar división por cero)
        mape = np.mean(np.abs((labels - predictions) / (labels + 1e-10))) * 100
        
        return {
            "mse": float(mse),
            "rmse": float(rmse),
            "mae": float(mae),
            "r2": float(r2),
            "mape": float(mape)
        }
    
    def evaluate_multilabel(
        self,
        predictions: np.ndarray,
        labels: np.ndarray
    ) -> Dict[str, Any]:
        """
        Evaluación para multi-label
        
        Args:
            predictions: Predicciones (binary matrix)
            labels: Labels verdaderos (binary matrix)
            
        Returns:
            Métricas multi-label
        """
        from sklearn.metrics import (
            hamming_loss,
            jaccard_score,
            f1_score
        )
        
        hamming = hamming_loss(labels, predictions)
        jaccard = jaccard_score(labels, predictions, average='macro', zero_division=0)
        f1_macro = f1_score(labels, predictions, average='macro', zero_division=0)
        f1_micro = f1_score(labels, predictions, average='micro', zero_division=0)
        
        return {
            "hamming_loss": float(hamming),
            "jaccard_score": float(jaccard),
            "f1_macro": float(f1_macro),
            "f1_micro": float(f1_micro)
        }




