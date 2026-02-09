"""
Evaluation Metrics
==================

Metrics for model evaluation.
"""

import torch
import numpy as np
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class RegressionMetrics:
    """Regression metrics."""
    mse: float
    mae: float
    rmse: float
    r2: float
    mape: float


@dataclass
class ClassificationMetrics:
    """Classification metrics."""
    accuracy: float
    precision: float
    recall: float
    f1: float
    auc: Optional[float] = None


def compute_regression_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor
) -> RegressionMetrics:
    """
    Compute regression metrics.
    
    Args:
        predictions: Predicted values
        targets: Target values
    
    Returns:
        Regression metrics
    """
    predictions = predictions.cpu().numpy().flatten()
    targets = targets.cpu().numpy().flatten()
    
    # MSE
    mse = np.mean((predictions - targets) ** 2)
    
    # MAE
    mae = np.mean(np.abs(predictions - targets))
    
    # RMSE
    rmse = np.sqrt(mse)
    
    # R²
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    # MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((targets - predictions) / (targets + 1e-8))) * 100
    
    return RegressionMetrics(
        mse=float(mse),
        mae=float(mae),
        rmse=float(rmse),
        r2=float(r2),
        mape=float(mape)
    )


def compute_classification_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5
) -> ClassificationMetrics:
    """
    Compute classification metrics.
    
    Args:
        predictions: Predicted probabilities
        targets: Target labels
        threshold: Classification threshold
    
    Returns:
        Classification metrics
    """
    predictions = predictions.cpu().numpy().flatten()
    targets = targets.cpu().numpy().flatten()
    
    # Binary predictions
    pred_binary = (predictions >= threshold).astype(int)
    
    # True positives, false positives, true negatives, false negatives
    tp = np.sum((pred_binary == 1) & (targets == 1))
    fp = np.sum((pred_binary == 1) & (targets == 0))
    tn = np.sum((pred_binary == 0) & (targets == 0))
    fn = np.sum((pred_binary == 0) & (targets == 1))
    
    # Accuracy
    accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0.0
    
    # Precision
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    
    # Recall
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    # F1
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # AUC (simplified, for binary classification)
    try:
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(targets, predictions)
    except ImportError:
        auc = None
    
    return ClassificationMetrics(
        accuracy=float(accuracy),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        auc=float(auc) if auc is not None else None
    )


def compute_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    task_type: str = "regression",
    **kwargs
) -> Dict[str, float]:
    """
    Compute metrics based on task type.
    
    Args:
        predictions: Predictions
        targets: Targets
        task_type: "regression" or "classification"
        **kwargs: Additional arguments
    
    Returns:
        Dictionary of metrics
    """
    if task_type == "regression":
        metrics = compute_regression_metrics(predictions, targets)
        return {
            "mse": metrics.mse,
            "mae": metrics.mae,
            "rmse": metrics.rmse,
            "r2": metrics.r2,
            "mape": metrics.mape
        }
    elif task_type == "classification":
        threshold = kwargs.get("threshold", 0.5)
        metrics = compute_classification_metrics(predictions, targets, threshold)
        result = {
            "accuracy": metrics.accuracy,
            "precision": metrics.precision,
            "recall": metrics.recall,
            "f1": metrics.f1
        }
        if metrics.auc is not None:
            result["auc"] = metrics.auc
        return result
    else:
        raise ValueError(f"Unknown task type: {task_type}")




