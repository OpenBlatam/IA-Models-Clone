"""
Evaluation Utilities

Provides comprehensive evaluation metrics and utilities for model evaluation:
- Classification metrics
- Regression metrics
- Custom metrics
- Cross-validation
- Model comparison
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score
)

logger = logging.getLogger(__name__)


class ClassificationMetrics:
    """
    Compute classification metrics
    """
    
    @staticmethod
    def compute_metrics(
        predictions: np.ndarray,
        labels: np.ndarray,
        average: str = "weighted",
        include_confusion_matrix: bool = True
    ) -> Dict[str, Any]:
        """
        Compute classification metrics
        
        Args:
            predictions: Predicted labels
            labels: True labels
            average: Averaging strategy (micro, macro, weighted)
            include_confusion_matrix: Whether to include confusion matrix
            
        Returns:
            Dictionary with metrics
        """
        metrics = {
            "accuracy": accuracy_score(labels, predictions),
            "precision": precision_score(labels, predictions, average=average, zero_division=0),
            "recall": recall_score(labels, predictions, average=average, zero_division=0),
            "f1": f1_score(labels, predictions, average=average, zero_division=0)
        }
        
        if include_confusion_matrix:
            metrics["confusion_matrix"] = confusion_matrix(labels, predictions).tolist()
        
        # Per-class metrics
        try:
            metrics["per_class_precision"] = precision_score(
                labels, predictions, average=None, zero_division=0
            ).tolist()
            metrics["per_class_recall"] = recall_score(
                labels, predictions, average=None, zero_division=0
            ).tolist()
            metrics["per_class_f1"] = f1_score(
                labels, predictions, average=None, zero_division=0
            ).tolist()
        except:
            pass
        
        return metrics
    
    @staticmethod
    def compute_metrics_with_probs(
        predictions: np.ndarray,
        probabilities: np.ndarray,
        labels: np.ndarray,
        average: str = "weighted"
    ) -> Dict[str, Any]:
        """
        Compute metrics including probability-based metrics
        
        Args:
            predictions: Predicted labels
            probabilities: Prediction probabilities
            labels: True labels
            average: Averaging strategy
            
        Returns:
            Dictionary with metrics including ROC-AUC
        """
        metrics = ClassificationMetrics.compute_metrics(
            predictions, labels, average, include_confusion_matrix=True
        )
        
        # Add probability-based metrics
        try:
            if probabilities.shape[1] == 2:  # Binary classification
                metrics["roc_auc"] = roc_auc_score(labels, probabilities[:, 1])
                metrics["average_precision"] = average_precision_score(labels, probabilities[:, 1])
            else:  # Multi-class
                metrics["roc_auc"] = roc_auc_score(
                    labels, probabilities, multi_class="ovr", average=average
                )
        except Exception as e:
            logger.warning(f"Could not compute ROC-AUC: {e}")
        
        return metrics
    
    @staticmethod
    def classification_report_dict(
        predictions: np.ndarray,
        labels: np.ndarray,
        target_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Get detailed classification report
        
        Args:
            predictions: Predicted labels
            labels: True labels
            target_names: Names of classes
            
        Returns:
            Dictionary with detailed report
        """
        report = classification_report(
            labels,
            predictions,
            target_names=target_names,
            output_dict=True,
            zero_division=0
        )
        return report


class ModelEvaluator:
    """
    Comprehensive model evaluator
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        task_type: str = "classification"
    ):
        """
        Initialize evaluator
        
        Args:
            model: Model to evaluate
            device: Device to run on
            task_type: Type of task (classification, regression)
        """
        self.model = model
        self.device = device
        self.task_type = task_type
        self.model.eval()
    
    def evaluate(
        self,
        dataloader: torch.utils.data.DataLoader,
        compute_probs: bool = False
    ) -> Dict[str, Any]:
        """
        Evaluate model on dataloader
        
        Args:
            dataloader: Data loader
            compute_probs: Whether to compute probabilities
            
        Returns:
            Dictionary with evaluation results
        """
        all_predictions = []
        all_labels = []
        all_probabilities = []
        total_loss = 0.0
        
        criterion = nn.CrossEntropyLoss() if self.task_type == "classification" else nn.MSELoss()
        
        with torch.no_grad():
            for batch in dataloader:
                # Move to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(**batch)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                
                # Compute loss
                if 'labels' in batch:
                    loss = criterion(logits, batch['labels'])
                    total_loss += loss.item()
                    all_labels.extend(batch['labels'].cpu().numpy())
                
                # Get predictions
                if self.task_type == "classification":
                    predictions = torch.argmax(logits, dim=-1)
                    all_predictions.extend(predictions.cpu().numpy())
                    
                    if compute_probs:
                        probs = torch.softmax(logits, dim=-1)
                        all_probabilities.extend(probs.cpu().numpy())
                else:
                    all_predictions.extend(logits.cpu().numpy())
        
        # Convert to numpy
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels) if all_labels else None
        
        results = {
            "loss": total_loss / len(dataloader) if len(dataloader) > 0 else 0.0,
            "num_samples": len(all_predictions)
        }
        
        # Compute metrics
        if self.task_type == "classification" and all_labels is not None:
            if compute_probs and all_probabilities:
                results["metrics"] = ClassificationMetrics.compute_metrics_with_probs(
                    all_predictions,
                    np.array(all_probabilities),
                    all_labels
                )
            else:
                results["metrics"] = ClassificationMetrics.compute_metrics(
                    all_predictions,
                    all_labels
                )
            
            results["detailed_report"] = ClassificationMetrics.classification_report_dict(
                all_predictions,
                all_labels
            )
        
        return results
    
    def predict(
        self,
        dataloader: torch.utils.data.DataLoader,
        return_probs: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Get predictions from model
        
        Args:
            dataloader: Data loader
            return_probs: Whether to return probabilities
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        all_predictions = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                outputs = self.model(**batch)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                
                if self.task_type == "classification":
                    predictions = torch.argmax(logits, dim=-1)
                    all_predictions.extend(predictions.cpu().numpy())
                    
                    if return_probs:
                        probs = torch.softmax(logits, dim=-1)
                        all_probabilities.extend(probs.cpu().numpy())
                else:
                    all_predictions.extend(logits.cpu().numpy())
        
        predictions = np.array(all_predictions)
        probabilities = np.array(all_probabilities) if all_probabilities else None
        
        return predictions, probabilities


def cross_validate(
    model_factory: callable,
    dataset: torch.utils.data.Dataset,
    k_folds: int = 5,
    batch_size: int = 32,
    device: torch.device = None
) -> Dict[str, List[float]]:
    """
    Perform k-fold cross-validation
    
    Args:
        model_factory: Function that creates a new model
        dataset: Dataset to use
        k_folds: Number of folds
        batch_size: Batch size
        device: Device to use
        
    Returns:
        Dictionary with cross-validation results
    """
    from sklearn.model_selection import KFold
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    kfold = KFold(n_splits=k_folds, shuffle=True)
    results = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": []
    }
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        logger.info(f"Fold {fold + 1}/{k_folds}")
        
        # Create splits
        train_subset = torch.utils.data.Subset(dataset, train_idx)
        val_subset = torch.utils.data.Subset(dataset, val_idx)
        
        train_loader = torch.utils.data.DataLoader(train_subset, batch_size=batch_size)
        val_loader = torch.utils.data.DataLoader(val_subset, batch_size=batch_size)
        
        # Create and train model
        model = model_factory()
        # ... training code ...
        
        # Evaluate
        evaluator = ModelEvaluator(model, device)
        val_results = evaluator.evaluate(val_loader)
        
        results["val_loss"].append(val_results["loss"])
        if "metrics" in val_results:
            results["val_accuracy"].append(val_results["metrics"]["accuracy"])
    
    # Compute statistics
    cv_results = {
        "mean_val_loss": np.mean(results["val_loss"]),
        "std_val_loss": np.std(results["val_loss"]),
        "mean_val_accuracy": np.mean(results["val_accuracy"]) if results["val_accuracy"] else None,
        "std_val_accuracy": np.std(results["val_accuracy"]) if results["val_accuracy"] else None
    }
    
    return cv_results















