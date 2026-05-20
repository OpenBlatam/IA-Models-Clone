"""
Advanced Metrics
Additional evaluation metrics
"""

import torch
import numpy as np
from typing import List, Dict, Any, Optional
from sklearn.metrics import (
    mean_absolute_percentage_error,
    mean_squared_log_error,
    r2_score,
    explained_variance_score
)


class AdvancedMetrics:
    """
    Advanced evaluation metrics
    """
    
    @staticmethod
    def calculate_regression_metrics(
        y_true: List[float],
        y_pred: List[float]
    ) -> Dict[str, float]:
        """
        Calculate comprehensive regression metrics
        
        Args:
            y_true: True values
            y_pred: Predictions
            
        Returns:
            Dictionary of metrics
        """
        y_true_np = np.array(y_true)
        y_pred_np = np.array(y_pred)
        
        metrics = {
            "mse": float(np.mean((y_true_np - y_pred_np) ** 2)),
            "rmse": float(np.sqrt(np.mean((y_true_np - y_pred_np) ** 2))),
            "mae": float(np.mean(np.abs(y_true_np - y_pred_np))),
            "mape": float(mean_absolute_percentage_error(y_true_np, y_pred_np)),
            "r2": float(r2_score(y_true_np, y_pred_np)),
            "explained_variance": float(explained_variance_score(y_true_np, y_pred_np))
        }
        
        # Calculate percentage errors
        with np.errstate(divide='ignore', invalid='ignore'):
            percentage_errors = np.abs((y_true_np - y_pred_np) / (y_true_np + 1e-8)) * 100
            metrics["mean_percentage_error"] = float(np.nanmean(percentage_errors))
            metrics["median_percentage_error"] = float(np.nanmedian(percentage_errors))
        
        return metrics
    
    @staticmethod
    def calculate_classification_metrics(
        y_true: List[int],
        y_pred: List[int],
        num_classes: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive classification metrics
        
        Args:
            y_true: True labels
            y_pred: Predictions
            num_classes: Number of classes
            
        Returns:
            Dictionary of metrics
        """
        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
            confusion_matrix,
            classification_report
        )
        
        y_true_np = np.array(y_true)
        y_pred_np = np.array(y_pred)
        
        metrics = {
            "accuracy": float(accuracy_score(y_true_np, y_pred_np)),
            "precision": float(precision_score(y_true_np, y_pred_np, average='weighted', zero_division=0)),
            "recall": float(recall_score(y_true_np, y_pred_np, average='weighted', zero_division=0)),
            "f1": float(f1_score(y_true_np, y_pred_np, average='weighted', zero_division=0)),
            "confusion_matrix": confusion_matrix(y_true_np, y_pred_np).tolist()
        }
        
        if num_classes:
            metrics["per_class_precision"] = precision_score(
                y_true_np, y_pred_np, average=None, zero_division=0
            ).tolist()
            metrics["per_class_recall"] = recall_score(
                y_true_np, y_pred_np, average=None, zero_division=0
            ).tolist()
            metrics["per_class_f1"] = f1_score(
                y_true_np, y_pred_np, average=None, zero_division=0
            ).tolist()
        
        return metrics
    
    @staticmethod
    def calculate_correlation(
        y_true: List[float],
        y_pred: List[float]
    ) -> Dict[str, float]:
        """
        Calculate correlation metrics
        
        Args:
            y_true: True values
            y_pred: Predictions
            
        Returns:
            Correlation metrics
        """
        from scipy.stats import pearsonr, spearmanr
        
        y_true_np = np.array(y_true)
        y_pred_np = np.array(y_pred)
        
        pearson_r, pearson_p = pearsonr(y_true_np, y_pred_np)
        spearman_r, spearman_p = spearmanr(y_true_np, y_pred_np)
        
        return {
            "pearson_r": float(pearson_r),
            "pearson_p": float(pearson_p),
            "spearman_r": float(spearman_r),
            "spearman_p": float(spearman_p)
        }


def calculate_regression_metrics(y_true: List[float], y_pred: List[float]) -> Dict[str, float]:
    """Calculate regression metrics"""
    return AdvancedMetrics.calculate_regression_metrics(y_true, y_pred)


def calculate_classification_metrics(y_true: List[int], y_pred: List[int], **kwargs) -> Dict[str, Any]:
    """Calculate classification metrics"""
    return AdvancedMetrics.calculate_classification_metrics(y_true, y_pred, **kwargs)


def calculate_correlation(y_true: List[float], y_pred: List[float]) -> Dict[str, float]:
    """Calculate correlation"""
    return AdvancedMetrics.calculate_correlation(y_true, y_pred)








