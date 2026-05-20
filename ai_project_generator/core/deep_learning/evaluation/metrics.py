"""
Evaluation Metrics - Comprehensive Metrics for Model Evaluation
================================================================

Implements various metrics for classification, regression, and custom tasks.
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

logger = logging.getLogger(__name__)


class Metrics:
    """Container for evaluation metrics."""
    
    def __init__(self, metrics_dict: Dict[str, float]):
        """
        Initialize metrics.
        
        Args:
            metrics_dict: Dictionary of metric names and values
        """
        self.metrics = metrics_dict
    
    def __getitem__(self, key: str) -> float:
        """Get metric value."""
        return self.metrics[key]
    
    def __repr__(self) -> str:
        """String representation."""
        return f"Metrics({self.metrics})"
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return self.metrics.copy()


def compute_classification_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: Optional[int] = None,
    average: str = 'macro'
) -> Metrics:
    """
    Compute classification metrics.
    
    Args:
        predictions: Model predictions (logits or probabilities)
        targets: Ground truth labels
        num_classes: Number of classes (auto-detected if None)
        average: Averaging strategy for multi-class ('macro', 'micro', 'weighted')
        
    Returns:
        Metrics object with computed metrics
    """
    # Convert to numpy
    if isinstance(predictions, torch.Tensor):
        predictions_np = predictions.detach().cpu().numpy()
    else:
        predictions_np = np.array(predictions)
    
    if isinstance(targets, torch.Tensor):
        targets_np = targets.detach().cpu().numpy()
    else:
        targets_np = np.array(targets)
    
    # Get predicted classes
    if predictions_np.ndim > 1:
        predicted_classes = np.argmax(predictions_np, axis=-1)
    else:
        predicted_classes = (predictions_np > 0.5).astype(int)
    
    # Flatten if needed
    predicted_classes = predicted_classes.flatten()
    targets_np = targets_np.flatten()
    
    # Compute metrics
    accuracy = accuracy_score(targets_np, predicted_classes)
    
    try:
        precision = precision_score(
            targets_np,
            predicted_classes,
            average=average,
            zero_division=0
        )
        recall = recall_score(
            targets_np,
            predicted_classes,
            average=average,
            zero_division=0
        )
        f1 = f1_score(
            targets_np,
            predicted_classes,
            average=average,
            zero_division=0
        )
    except Exception as e:
        logger.warning(f"Error computing precision/recall/f1: {e}")
        precision = recall = f1 = 0.0
    
    # Confusion matrix
    try:
        cm = confusion_matrix(targets_np, predicted_classes)
    except Exception:
        cm = None
    
    # ROC AUC (for binary classification)
    roc_auc = None
    if num_classes is None or num_classes == 2:
        try:
            if predictions_np.ndim > 1:
                # Use probability of positive class
                if predictions_np.shape[1] == 2:
                    proba = predictions_np[:, 1]
                else:
                    proba = torch.softmax(torch.from_numpy(predictions_np), dim=-1)[:, 1].numpy()
            else:
                proba = predictions_np
            roc_auc = roc_auc_score(targets_np, proba)
        except Exception as e:
            logger.warning(f"Error computing ROC AUC: {e}")
    
    metrics_dict = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
    }
    
    if roc_auc is not None:
        metrics_dict['roc_auc'] = float(roc_auc)
    
    return Metrics(metrics_dict)


def compute_regression_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor
) -> Metrics:
    """
    Compute regression metrics.
    
    Args:
        predictions: Model predictions
        targets: Ground truth values
        
    Returns:
        Metrics object with computed metrics
    """
    # Convert to numpy
    if isinstance(predictions, torch.Tensor):
        predictions_np = predictions.detach().cpu().numpy()
    else:
        predictions_np = np.array(predictions)
    
    if isinstance(targets, torch.Tensor):
        targets_np = targets.detach().cpu().numpy()
    else:
        targets_np = np.array(targets)
    
    # Flatten if needed
    predictions_np = predictions_np.flatten()
    targets_np = targets_np.flatten()
    
    # Compute metrics
    mse = mean_squared_error(targets_np, predictions_np)
    mae = mean_absolute_error(targets_np, predictions_np)
    rmse = np.sqrt(mse)
    
    try:
        r2 = r2_score(targets_np, predictions_np)
    except Exception:
        r2 = 0.0
    
    metrics_dict = {
        'mse': float(mse),
        'mae': float(mae),
        'rmse': float(rmse),
        'r2_score': float(r2),
    }
    
    return Metrics(metrics_dict)


def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    task_type: str = 'classification',
    num_classes: Optional[int] = None
) -> Tuple[Metrics, Dict[str, Any]]:
    """
    Evaluate model on a dataset.
    
    Args:
        model: PyTorch model
        dataloader: DataLoader for evaluation
        device: Device to run evaluation on
        task_type: Type of task ('classification' or 'regression')
        num_classes: Number of classes (for classification)
        
    Returns:
        Tuple of (metrics, additional_info)
    """
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
            if isinstance(batch, dict):
                inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
                if 'labels' in inputs:
                    targets = inputs.pop('labels')
                else:
                    targets = None
                outputs = model(**inputs)
            elif isinstance(batch, (tuple, list)):
                inputs = batch[0].to(device)
                targets = batch[1].to(device) if len(batch) > 1 else None
                outputs = model(inputs)
            else:
                inputs = batch.to(device)
                outputs = model(inputs)
                targets = None
            
            all_predictions.append(outputs)
            if targets is not None:
                all_targets.append(targets)
    
    # Concatenate all predictions and targets
    predictions = torch.cat(all_predictions, dim=0)
    if all_targets:
        targets = torch.cat(all_targets, dim=0)
    else:
        logger.warning("No targets found in dataloader")
        return Metrics({}), {}
    
    # Compute metrics
    if task_type == 'classification':
        metrics = compute_classification_metrics(predictions, targets, num_classes)
    elif task_type == 'regression':
        metrics = compute_regression_metrics(predictions, targets)
    else:
        raise ValueError(f"Unsupported task type: {task_type}")
    
    additional_info = {
        'num_samples': len(targets),
        'predictions_shape': list(predictions.shape),
        'targets_shape': list(targets.shape),
    }
    
    return metrics, additional_info



