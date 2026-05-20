"""
Advanced Evaluation Metrics
============================

Advanced metrics for model evaluation.
"""

import torch
import numpy as np
import logging
from typing import Dict, Any, List, Optional
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

logger = logging.getLogger(__name__)


class AdvancedMetrics:
    """
    Advanced metrics calculator.
    
    Features:
    - Classification metrics
    - Regression metrics
    - Confusion matrix
    - Classification report
    - Custom metrics
    """
    
    @staticmethod
    def calculate_classification_metrics(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        threshold: float = 0.5,
        average: str = "binary"
    ) -> Dict[str, float]:
        """
        Calculate comprehensive classification metrics.
        
        Args:
            predictions: Predicted probabilities
            targets: True labels
            threshold: Classification threshold
            average: Averaging method
        
        Returns:
            Dictionary of metrics
        """
        predictions_np = predictions.cpu().numpy().flatten()
        targets_np = targets.cpu().numpy().flatten()
        
        # Binary predictions
        pred_binary = (predictions_np >= threshold).astype(int)
        
        metrics = {
            "accuracy": float(accuracy_score(targets_np, pred_binary)),
            "precision": float(precision_score(targets_np, pred_binary, average=average, zero_division=0)),
            "recall": float(recall_score(targets_np, pred_binary, average=average, zero_division=0)),
            "f1": float(f1_score(targets_np, pred_binary, average=average, zero_division=0))
        }
        
        # AUC for binary classification
        if len(np.unique(targets_np)) == 2:
            try:
                metrics["auc"] = float(roc_auc_score(targets_np, predictions_np))
            except:
                metrics["auc"] = 0.0
        
        # Confusion matrix
        cm = confusion_matrix(targets_np, pred_binary)
        metrics["confusion_matrix"] = cm.tolist()
        
        return metrics
    
    @staticmethod
    def calculate_regression_metrics(
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> Dict[str, float]:
        """
        Calculate comprehensive regression metrics.
        
        Args:
            predictions: Predictions
            targets: True values
        
        Returns:
            Dictionary of metrics
        """
        predictions_np = predictions.cpu().numpy().flatten()
        targets_np = targets.cpu().numpy().flatten()
        
        mse = np.mean((predictions_np - targets_np) ** 2)
        mae = np.mean(np.abs(predictions_np - targets_np))
        rmse = np.sqrt(mse)
        
        # R²
        ss_res = np.sum((targets_np - predictions_np) ** 2)
        ss_tot = np.sum((targets_np - np.mean(targets_np)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        # MAPE
        mape = np.mean(np.abs((targets_np - predictions_np) / (targets_np + 1e-8))) * 100
        
        # Correlation
        correlation = np.corrcoef(predictions_np, targets_np)[0, 1]
        
        return {
            "mse": float(mse),
            "mae": float(mae),
            "rmse": float(rmse),
            "r2": float(r2),
            "mape": float(mape),
            "correlation": float(correlation)
        }
    
    @staticmethod
    def get_classification_report(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        threshold: float = 0.5,
        target_names: Optional[List[str]] = None
    ) -> str:
        """
        Get detailed classification report.
        
        Args:
            predictions: Predicted probabilities
            targets: True labels
            threshold: Classification threshold
            target_names: Class names
        
        Returns:
            Classification report string
        """
        predictions_np = predictions.cpu().numpy().flatten()
        targets_np = targets.cpu().numpy().flatten()
        pred_binary = (predictions_np >= threshold).astype(int)
        
        return classification_report(
            targets_np,
            pred_binary,
            target_names=target_names,
            zero_division=0
        )

