"""Evaluation metrics for model assessment."""

import torch
import numpy as np
from typing import List, Dict, Any, Optional
from sklearn.metrics import (
    accuracy_score as sk_accuracy,
    precision_score as sk_precision,
    recall_score as sk_recall,
    f1_score as sk_f1,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    auc
)


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute accuracy score."""
    return float(sk_accuracy(y_true, y_pred))


def precision_score(y_true: np.ndarray, y_pred: np.ndarray, average: str = "weighted") -> float:
    """Compute precision score."""
    return float(sk_precision(y_true, y_pred, average=average, zero_division=0))


def recall_score(y_true: np.ndarray, y_pred: np.ndarray, average: str = "weighted") -> float:
    """Compute recall score."""
    return float(sk_recall(y_true, y_pred, average=average, zero_division=0))


def f1_score(y_true: np.ndarray, y_pred: np.ndarray, average: str = "weighted") -> float:
    """Compute F1 score."""
    return float(sk_f1(y_true, y_pred, average=average, zero_division=0))


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    compute_roc: bool = True,
    compute_pr: bool = True
) -> Dict[str, Any]:
    """
    Compute comprehensive evaluation metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (for ROC/PR curves)
        compute_roc: Whether to compute ROC curve
        compute_pr: Whether to compute PR curve
    
    Returns:
        Dictionary of metrics
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
    }
    
    # Confusion matrix
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()
    
    # ROC curve
    if compute_roc and y_proba is not None:
        try:
            fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1] if y_proba.shape[1] > 1 else y_proba)
            roc_auc = auc(fpr, tpr)
            metrics["roc_curve"] = {
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
                "auc": float(roc_auc)
            }
            metrics["auc_roc"] = float(roc_auc)
        except Exception:
            pass
    
    # Precision-Recall curve
    if compute_pr and y_proba is not None:
        try:
            precision, recall, _ = precision_recall_curve(
                y_true, y_proba[:, 1] if y_proba.shape[1] > 1 else y_proba
            )
            pr_auc = auc(recall, precision)
            metrics["precision_recall_curve"] = {
                "precision": precision.tolist(),
                "recall": recall.tolist(),
                "auc": float(pr_auc)
            }
            metrics["auc_pr"] = float(pr_auc)
        except Exception:
            pass
    
    return metrics



