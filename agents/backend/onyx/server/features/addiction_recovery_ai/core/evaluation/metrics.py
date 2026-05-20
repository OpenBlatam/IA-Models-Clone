"""
Evaluation Metrics
Comprehensive metrics for model evaluation
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score
)
import logging

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """
    Calculate various evaluation metrics
    """
    
    @staticmethod
    def classification_metrics(
        predictions: np.ndarray,
        targets: np.ndarray,
        average: str = "weighted"
    ) -> Dict[str, float]:
        """
        Calculate classification metrics
        
        Args:
            predictions: Predicted labels
            targets: True labels
            average: Averaging method
            
        Returns:
            Dictionary of metrics
        """
        metrics = {
            "accuracy": accuracy_score(targets, predictions),
            "precision": precision_score(targets, predictions, average=average, zero_division=0),
            "recall": recall_score(targets, predictions, average=average, zero_division=0),
            "f1": f1_score(targets, predictions, average=average, zero_division=0)
        }
        
        # ROC AUC for binary classification
        if len(np.unique(targets)) == 2:
            try:
                metrics["roc_auc"] = roc_auc_score(targets, predictions)
            except Exception as e:
                logger.warning(f"Could not calculate ROC AUC: {e}")
        
        return metrics
    
    @staticmethod
    def regression_metrics(
        predictions: np.ndarray,
        targets: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate regression metrics
        
        Args:
            predictions: Predicted values
            targets: True values
            
        Returns:
            Dictionary of metrics
        """
        return {
            "mse": mean_squared_error(targets, predictions),
            "rmse": np.sqrt(mean_squared_error(targets, predictions)),
            "mae": mean_absolute_error(targets, predictions),
            "r2": r2_score(targets, predictions),
            "mape": np.mean(np.abs((targets - predictions) / (targets + 1e-8))) * 100
        }
    
    @staticmethod
    def confusion_matrix_metrics(
        predictions: np.ndarray,
        targets: np.ndarray,
        labels: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Calculate confusion matrix and related metrics
        
        Args:
            predictions: Predicted labels
            targets: True labels
            labels: Label names
            
        Returns:
            Dictionary with confusion matrix and metrics
        """
        cm = confusion_matrix(targets, predictions, labels=labels)
        
        # Calculate per-class metrics
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        
        metrics = {
            "confusion_matrix": cm.tolist(),
            "true_positives": int(tp),
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "sensitivity": tp / (tp + fn) if (tp + fn) > 0 else 0.0,
            "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
            "precision": tp / (tp + fp) if (tp + fp) > 0 else 0.0,
            "recall": tp / (tp + fn) if (tp + fn) > 0 else 0.0
        }
        
        return metrics
    
    @staticmethod
    def classification_report_dict(
        predictions: np.ndarray,
        targets: np.ndarray,
        labels: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Get detailed classification report
        
        Args:
            predictions: Predicted labels
            targets: True labels
            labels: Label names
            
        Returns:
            Classification report as dictionary
        """
        report = classification_report(
            targets, predictions,
            labels=labels,
            output_dict=True,
            zero_division=0
        )
        return report


class ModelEvaluator:
    """
    Comprehensive model evaluator
    """
    
    def __init__(self, device: Optional[torch.device] = None):
        """
        Initialize evaluator
        
        Args:
            device: Device to use
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.metrics_calc = MetricsCalculator()
    
    def evaluate_classification(
        self,
        model: torch.nn.Module,
        data_loader: torch.utils.data.DataLoader,
        criterion: Optional[torch.nn.Module] = None
    ) -> Dict[str, float]:
        """
        Evaluate classification model
        
        Args:
            model: Model to evaluate
            data_loader: Data loader
            criterion: Loss function
            
        Returns:
            Dictionary of metrics
        """
        model.eval()
        all_predictions = []
        all_targets = []
        total_loss = 0.0
        
        with torch.inference_mode():
            for batch in data_loader:
                inputs = batch["features"].to(self.device)
                targets = batch["target"].to(self.device)
                
                outputs = model(inputs)
                
                if criterion:
                    loss = criterion(outputs, targets)
                    total_loss += loss.item()
                
                # Get predictions
                if outputs.dim() > 1 and outputs.size(1) > 1:
                    predictions = torch.argmax(outputs, dim=1).cpu().numpy()
                else:
                    predictions = (torch.sigmoid(outputs) > 0.5).cpu().numpy().flatten()
                
                all_predictions.extend(predictions)
                all_targets.extend(targets.cpu().numpy().flatten())
        
        metrics = self.metrics_calc.classification_metrics(
            np.array(all_predictions),
            np.array(all_targets)
        )
        
        if criterion:
            metrics["loss"] = total_loss / len(data_loader)
        
        return metrics
    
    def evaluate_regression(
        self,
        model: torch.nn.Module,
        data_loader: torch.utils.data.DataLoader,
        criterion: Optional[torch.nn.Module] = None
    ) -> Dict[str, float]:
        """
        Evaluate regression model
        
        Args:
            model: Model to evaluate
            data_loader: Data loader
            criterion: Loss function
            
        Returns:
            Dictionary of metrics
        """
        model.eval()
        all_predictions = []
        all_targets = []
        total_loss = 0.0
        
        with torch.inference_mode():
            for batch in data_loader:
                inputs = batch["features"].to(self.device)
                targets = batch["target"].to(self.device)
                
                outputs = model(inputs)
                
                if criterion:
                    loss = criterion(outputs, targets)
                    total_loss += loss.item()
                
                predictions = outputs.cpu().numpy().flatten()
                all_predictions.extend(predictions)
                all_targets.extend(targets.cpu().numpy().flatten())
        
        metrics = self.metrics_calc.regression_metrics(
            np.array(all_predictions),
            np.array(all_targets)
        )
        
        if criterion:
            metrics["loss"] = total_loss / len(data_loader)
        
        return metrics


def create_evaluator(device: Optional[torch.device] = None) -> ModelEvaluator:
    """Factory for model evaluator"""
    return ModelEvaluator(device)













