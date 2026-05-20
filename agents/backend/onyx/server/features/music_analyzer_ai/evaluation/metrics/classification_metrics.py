"""
Classification Metrics

Specialized metrics for classification tasks following
PyTorch and scikit-learn best practices.
"""

from typing import Dict, List, Optional, Tuple
import torch
import numpy as np

try:
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        confusion_matrix,
        classification_report
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class ClassificationMetrics:
    """
    Classification metrics calculator.
    
    Computes various classification metrics including:
    - Accuracy
    - Precision, Recall, F1
    - Confusion matrix
    - Per-class metrics
    """
    
    def __init__(
        self,
        num_classes: Optional[int] = None,
        average: str = "weighted"
    ):
        """
        Initialize classification metrics.
        
        Args:
            num_classes: Number of classes (auto-detect if None)
            average: Averaging method for multi-class ("macro", "weighted", "micro")
        """
        self.num_classes = num_classes
        self.average = average
        self.reset()
    
    def reset(self):
        """Reset all accumulated metrics."""
        self.predictions = []
        self.targets = []
    
    def update(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ):
        """
        Update metrics with new predictions.
        
        Args:
            predictions: Model predictions [batch, num_classes] or [batch]
            targets: Target labels [batch]
        """
        # Convert to numpy
        if isinstance(predictions, torch.Tensor):
            if predictions.dim() > 1:
                preds = predictions.argmax(dim=-1).cpu().numpy()
            else:
                preds = predictions.cpu().numpy()
        else:
            preds = np.array(predictions)
        
        if isinstance(targets, torch.Tensor):
            targets_np = targets.cpu().numpy()
        else:
            targets_np = np.array(targets)
        
        self.predictions.extend(preds)
        self.targets.extend(targets_np)
    
    def compute(self) -> Dict[str, float]:
        """
        Compute all metrics.
        
        Returns:
            Dictionary of metric names and values
        """
        if not self.predictions:
            return {}
        
        preds = np.array(self.predictions)
        targets = np.array(self.targets)
        
        if not SKLEARN_AVAILABLE:
            # Fallback to basic metrics
            return self._compute_basic_metrics(preds, targets)
        
        metrics = {
            "accuracy": accuracy_score(targets, preds),
            "precision": precision_score(
                targets, preds, average=self.average, zero_division=0
            ),
            "recall": recall_score(
                targets, preds, average=self.average, zero_division=0
            ),
            "f1": f1_score(
                targets, preds, average=self.average, zero_division=0
            ),
        }
        
        # Per-class metrics if binary or small number of classes
        if self.num_classes is None:
            self.num_classes = len(np.unique(targets))
        
        if self.num_classes <= 10:
            metrics["per_class_precision"] = precision_score(
                targets, preds, average=None, zero_division=0
            ).tolist()
            metrics["per_class_recall"] = recall_score(
                targets, preds, average=None, zero_division=0
            ).tolist()
            metrics["per_class_f1"] = f1_score(
                targets, preds, average=None, zero_division=0
            ).tolist()
        
        return metrics
    
    def _compute_basic_metrics(
        self,
        preds: np.ndarray,
        targets: np.ndarray
    ) -> Dict[str, float]:
        """Compute basic metrics without sklearn."""
        correct = (preds == targets).sum()
        total = len(preds)
        
        return {
            "accuracy": correct / total if total > 0 else 0.0,
            "correct": int(correct),
            "total": int(total)
        }
    
    def get_confusion_matrix(self) -> np.ndarray:
        """Get confusion matrix."""
        if not self.predictions:
            return np.array([])
        
        preds = np.array(self.predictions)
        targets = np.array(self.targets)
        
        if SKLEARN_AVAILABLE:
            return confusion_matrix(targets, preds)
        else:
            # Simple confusion matrix
            if self.num_classes is None:
                self.num_classes = len(np.unique(targets))
            
            cm = np.zeros((self.num_classes, self.num_classes), dtype=int)
            for t, p in zip(targets, preds):
                cm[int(t), int(p)] += 1
            return cm


__all__ = [
    "ClassificationMetrics",
]



