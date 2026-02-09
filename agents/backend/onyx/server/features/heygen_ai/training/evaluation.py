"""
Evaluation Module for HeyGen AI
=================================

Implements model evaluation following best practices:
- Multiple evaluation metrics
- Proper train/validation/test splits
- Cross-validation support
- Metric computation and tracking
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from sklearn.metrics import (
        accuracy_score,
        precision_recall_fscore_support,
        confusion_matrix,
        roc_auc_score,
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("Scikit-learn not available. Install with: pip install scikit-learn")

logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics.
    
    Attributes:
        loss: Average loss
        accuracy: Accuracy score
        precision: Precision score
        recall: Recall score
        f1: F1 score
        confusion_matrix: Confusion matrix
        roc_auc: ROC AUC score (for binary classification)
    """
    loss: float
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1: Optional[float] = None
    confusion_matrix: Optional[np.ndarray] = None
    roc_auc: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary.
        
        Returns:
            Dictionary of metrics
        """
        metrics = {"loss": self.loss}
        
        if self.accuracy is not None:
            metrics["accuracy"] = self.accuracy
        if self.precision is not None:
            metrics["precision"] = self.precision
        if self.recall is not None:
            metrics["recall"] = self.recall
        if self.f1 is not None:
            metrics["f1"] = self.f1
        if self.roc_auc is not None:
            metrics["roc_auc"] = self.roc_auc
        
        return metrics


class ModelEvaluator:
    """Evaluates models with proper metrics and error handling.
    
    Features:
    - Multiple evaluation metrics
    - Proper device management
    - Batch processing
    - Metric aggregation
    """
    
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        device: Optional[torch.device] = None,
    ):
        """Initialize model evaluator.
        
        Args:
            model: Model to evaluate
            criterion: Loss function
            device: Evaluation device
        """
        self.model = model
        self.criterion = criterion
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        self.logger = logging.getLogger(f"{__name__}.ModelEvaluator")
    
    def evaluate(
        self,
        dataloader: DataLoader,
        compute_metrics: bool = True,
    ) -> EvaluationMetrics:
        """Evaluate model on dataset.
        
        Args:
            dataloader: DataLoader for evaluation data
            compute_metrics: Whether to compute classification metrics
        
        Returns:
            EvaluationMetrics object
        """
        self.model.eval()
        
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch in dataloader:
                # Move batch to device
                batch = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                
                # Forward pass
                outputs = self.model(**batch)
                
                # Compute loss
                if hasattr(outputs, 'loss') and outputs.loss is not None:
                    loss = outputs.loss
                else:
                    logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
                    labels = batch.get('labels', batch.get('label'))
                    if labels is not None:
                        loss = self.criterion(logits, labels)
                    else:
                        loss = torch.tensor(0.0)
                
                total_loss += loss.item()
                
                # Collect predictions and labels for metrics
                if compute_metrics:
                    logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
                    labels = batch.get('labels', batch.get('label'))
                    
                    if labels is not None:
                        probs = torch.softmax(logits, dim=-1)
                        predictions = torch.argmax(logits, dim=-1)
                        
                        all_predictions.extend(predictions.cpu().numpy())
                        all_labels.extend(labels.cpu().numpy())
                        all_probs.extend(probs.cpu().numpy())
        
        avg_loss = total_loss / len(dataloader)
        
        # Compute metrics if requested
        metrics = EvaluationMetrics(loss=avg_loss)
        
        if compute_metrics and all_labels and SKLEARN_AVAILABLE:
            metrics = self._compute_metrics(
                all_labels,
                all_predictions,
                all_probs,
                metrics,
            )
        
        return metrics
    
    def _compute_metrics(
        self,
        labels: List[int],
        predictions: List[int],
        probs: List[np.ndarray],
        metrics: EvaluationMetrics,
    ) -> EvaluationMetrics:
        """Compute classification metrics.
        
        Args:
            labels: True labels
            predictions: Predicted labels
            probs: Prediction probabilities
            metrics: Metrics object to update
        
        Returns:
            Updated metrics object
        """
        try:
            # Basic metrics
            metrics.accuracy = accuracy_score(labels, predictions)
            
            # Precision, recall, F1
            precision, recall, f1, _ = precision_recall_fscore_support(
                labels,
                predictions,
                average='weighted',
                zero_division=0,
            )
            
            metrics.precision = precision
            metrics.recall = recall
            metrics.f1 = f1
            
            # Confusion matrix
            metrics.confusion_matrix = confusion_matrix(labels, predictions)
            
            # ROC AUC for binary classification
            if len(np.unique(labels)) == 2:
                try:
                    probs_array = np.array(probs)
                    if probs_array.shape[1] == 2:
                        metrics.roc_auc = roc_auc_score(
                            labels,
                            probs_array[:, 1]
                        )
                except Exception as e:
                    self.logger.warning(f"ROC AUC computation failed: {e}")
            
        except Exception as e:
            self.logger.warning(f"Metric computation failed: {e}")
        
        return metrics


def train_test_split(
    data: List[Any],
    test_size: float = 0.2,
    random_seed: Optional[int] = None,
) -> Tuple[List[Any], List[Any]]:
    """Split data into train and test sets.
    
    Args:
        data: Input data list
        test_size: Proportion of test data
        random_seed: Random seed for reproducibility
    
    Returns:
        Tuple of (train_data, test_data)
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    indices = np.random.permutation(len(data))
    split_idx = int(len(data) * (1 - test_size))
    
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    
    train_data = [data[i] for i in train_indices]
    test_data = [data[i] for i in test_indices]
    
    return train_data, test_data


def train_val_test_split(
    data: List[Any],
    val_size: float = 0.1,
    test_size: float = 0.1,
    random_seed: Optional[int] = None,
) -> Tuple[List[Any], List[Any], List[Any]]:
    """Split data into train, validation, and test sets.
    
    Args:
        data: Input data list
        val_size: Proportion of validation data
        test_size: Proportion of test data
        random_seed: Random seed for reproducibility
    
    Returns:
        Tuple of (train_data, val_data, test_data)
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    indices = np.random.permutation(len(data))
    
    val_split = int(len(data) * (1 - val_size - test_size))
    test_split = int(len(data) * (1 - test_size))
    
    train_indices = indices[:val_split]
    val_indices = indices[val_split:test_split]
    test_indices = indices[test_split:]
    
    train_data = [data[i] for i in train_indices]
    val_data = [data[i] for i in val_indices]
    test_data = [data[i] for i in test_indices]
    
    return train_data, val_data, test_data



