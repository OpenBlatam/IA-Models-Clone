"""
Modular Evaluation Metrics
Separated metrics for different tasks
"""

from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import numpy as np
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available")

try:
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        confusion_matrix,
        classification_report,
        mean_squared_error,
        mean_absolute_error,
        r2_score
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("Scikit-learn not available")


class ClassificationMetrics:
    """Metrics for classification tasks"""
    
    def __init__(self, num_classes: int, class_names: Optional[List[str]] = None):
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class_{i}" for i in range(num_classes)]
    
    def compute(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        average: str = "weighted"
    ) -> Dict[str, float]:
        """
        Compute classification metrics
        
        Args:
            predictions: Predicted logits [batch, num_classes]
            targets: True labels [batch]
            average: Averaging strategy for multi-class
        
        Returns:
            Dictionary of metrics
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required")
        
        # Convert to numpy
        if isinstance(predictions, torch.Tensor):
            pred_probs = torch.softmax(predictions, dim=-1)
            pred_classes = torch.argmax(pred_probs, dim=-1).cpu().numpy()
        else:
            pred_probs = predictions
            pred_classes = np.argmax(pred_probs, axis=-1)
        
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()
        
        if not SKLEARN_AVAILABLE:
            # Basic metrics without sklearn
            accuracy = (pred_classes == targets).mean()
            return {"accuracy": float(accuracy)}
        
        # Compute metrics
        accuracy = accuracy_score(targets, pred_classes)
        precision = precision_score(targets, pred_classes, average=average, zero_division=0)
        recall = recall_score(targets, pred_classes, average=average, zero_division=0)
        f1 = f1_score(targets, pred_classes, average=average, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(targets, pred_classes)
        
        return {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "confusion_matrix": cm.tolist()
        }
    
    def compute_per_class(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> Dict[str, Dict[str, float]]:
        """Compute per-class metrics"""
        if not SKLEARN_AVAILABLE:
            return {}
        
        if isinstance(predictions, torch.Tensor):
            pred_classes = torch.argmax(predictions, dim=-1).cpu().numpy()
        else:
            pred_classes = np.argmax(predictions, axis=-1)
        
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()
        
        report = classification_report(
            targets, pred_classes,
            target_names=self.class_names,
            output_dict=True,
            zero_division=0
        )
        
        return report


class RegressionMetrics:
    """Metrics for regression tasks"""
    
    def compute(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute regression metrics
        
        Args:
            predictions: Predicted values [batch] or [batch, 1]
            targets: True values [batch] or [batch, 1]
        
        Returns:
            Dictionary of metrics
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required")
        
        # Convert to numpy
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()
        
        # Flatten if needed
        predictions = predictions.flatten()
        targets = targets.flatten()
        
        if not SKLEARN_AVAILABLE:
            # Basic metrics
            mse = np.mean((predictions - targets) ** 2)
            mae = np.mean(np.abs(predictions - targets))
            return {
                "mse": float(mse),
                "mae": float(mae),
                "rmse": float(np.sqrt(mse))
            }
        
        # Compute metrics
        mse = mean_squared_error(targets, predictions)
        mae = mean_absolute_error(targets, predictions)
        r2 = r2_score(targets, predictions)
        
        return {
            "mse": float(mse),
            "mae": float(mae),
            "rmse": float(np.sqrt(mse)),
            "r2_score": float(r2)
        }


class MultiTaskMetrics:
    """Metrics for multi-task learning"""
    
    def __init__(
        self,
        classification_tasks: Optional[Dict[str, int]] = None,
        regression_tasks: Optional[List[str]] = None
    ):
        self.classification_tasks = classification_tasks or {}
        self.regression_tasks = regression_tasks or []
        
        # Initialize metric calculators
        self.class_metrics = {
            task: ClassificationMetrics(num_classes)
            for task, num_classes in self.classification_tasks.items()
        }
        self.reg_metrics = {
            task: RegressionMetrics()
            for task in self.regression_tasks
        }
    
    def compute(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute metrics for all tasks
        
        Args:
            predictions: Dictionary of predictions for each task
            targets: Dictionary of targets for each task
        
        Returns:
            Dictionary of metrics for each task
        """
        all_metrics = {}
        
        # Classification tasks
        for task_name, metrics_calc in self.class_metrics.items():
            if task_name in predictions and task_name in targets:
                all_metrics[task_name] = metrics_calc.compute(
                    predictions[task_name],
                    targets[task_name]
                )
        
        # Regression tasks
        for task_name, metrics_calc in self.reg_metrics.items():
            if task_name in predictions and task_name in targets:
                all_metrics[task_name] = metrics_calc.compute(
                    predictions[task_name],
                    targets[task_name]
                )
        
        return all_metrics



