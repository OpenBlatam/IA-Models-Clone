"""
Metrics calculation for model evaluation
Separated for modularity
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


class ClassificationMetrics:
    """
    Metrics for classification tasks
    """
    
    @staticmethod
    def accuracy(predictions: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
        """Calculate accuracy"""
        if predictions.dim() > 1:
            # Multi-label: use threshold
            pred_binary = (torch.sigmoid(predictions) > threshold).float()
        else:
            # Binary: use argmax
            pred_binary = (predictions > threshold).float()
        
        correct = (pred_binary == targets).float()
        return correct.mean().item()
    
    @staticmethod
    def precision_recall_f1(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        threshold: float = 0.5
    ) -> Dict[str, float]:
        """Calculate precision, recall, and F1 score"""
        pred_binary = (torch.sigmoid(predictions) > threshold).float()
        
        tp = (pred_binary * targets).sum().item()
        fp = (pred_binary * (1 - targets)).sum().item()
        fn = ((1 - pred_binary) * targets).sum().item()
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    @staticmethod
    def roc_auc(predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """Calculate ROC AUC (requires sklearn)"""
        try:
            from sklearn.metrics import roc_auc_score
            
            if predictions.dim() > 1:
                # Multi-label: average over classes
                pred_probs = torch.sigmoid(predictions).cpu().numpy()
                targets_np = targets.cpu().numpy()
                return roc_auc_score(targets_np, pred_probs, average='macro')
            else:
                pred_probs = torch.sigmoid(predictions).cpu().numpy()
                targets_np = targets.cpu().numpy()
                return roc_auc_score(targets_np, pred_probs)
        except ImportError:
            logger.warning("sklearn not available for ROC AUC")
            return 0.0


class RegressionMetrics:
    """
    Metrics for regression tasks
    """
    
    @staticmethod
    def mse(predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """Mean Squared Error"""
        return torch.nn.functional.mse_loss(predictions, targets).item()
    
    @staticmethod
    def mae(predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """Mean Absolute Error"""
        return torch.nn.functional.l1_loss(predictions, targets).item()
    
    @staticmethod
    def rmse(predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """Root Mean Squared Error"""
        mse = torch.nn.functional.mse_loss(predictions, targets).item()
        return np.sqrt(mse)
    
    @staticmethod
    def r2_score(predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """R² score"""
        ss_res = ((targets - predictions) ** 2).sum().item()
        ss_tot = ((targets - targets.mean()) ** 2).sum().item()
        return 1 - (ss_res / (ss_tot + 1e-8))
    
    @staticmethod
    def pearson_correlation(
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> float:
        """Pearson correlation coefficient"""
        pred_centered = predictions - predictions.mean()
        target_centered = targets - targets.mean()
        
        numerator = (pred_centered * target_centered).sum().item()
        pred_std = pred_centered.pow(2).sum().sqrt().item()
        target_std = target_centered.pow(2).sum().sqrt().item()
        
        denominator = pred_std * target_std + 1e-8
        return numerator / denominator


class MetricCalculator:
    """
    Calculate metrics for model predictions
    Handles both classification and regression tasks
    """
    
    def __init__(self):
        self.classification_metrics = ClassificationMetrics()
        self.regression_metrics = RegressionMetrics()
    
    def calculate_metrics(
        self,
        predictions: List[Dict[str, torch.Tensor]],
        targets: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, float]:
        """
        Calculate metrics from predictions and targets
        
        Args:
            predictions: List of prediction dictionaries
            targets: List of target dictionaries
            
        Returns:
            Dictionary of metric values
        """
        # Concatenate all predictions and targets
        all_preds = {}
        all_targets = {}
        
        for pred_dict, target_dict in zip(predictions, targets):
            for key in pred_dict.keys():
                if key in target_dict:
                    if key not in all_preds:
                        all_preds[key] = []
                        all_targets[key] = []
                    all_preds[key].append(pred_dict[key])
                    all_targets[key].append(target_dict[key])
        
        # Calculate metrics for each task
        metrics = {}
        
        for key in all_preds.keys():
            pred_tensor = torch.cat(all_preds[key], dim=0)
            target_tensor = torch.cat(all_targets[key], dim=0)
            
            # Determine task type based on key or values
            if key == 'conditions' or (target_tensor.max() <= 1.0 and target_tensor.min() >= 0.0):
                # Classification task
                metrics[f"{key}_accuracy"] = self.classification_metrics.accuracy(
                    pred_tensor, target_tensor
                )
                prf = self.classification_metrics.precision_recall_f1(
                    pred_tensor, target_tensor
                )
                metrics[f"{key}_precision"] = prf['precision']
                metrics[f"{key}_recall"] = prf['recall']
                metrics[f"{key}_f1"] = prf['f1']
            else:
                # Regression task
                metrics[f"{key}_mse"] = self.regression_metrics.mse(pred_tensor, target_tensor)
                metrics[f"{key}_mae"] = self.regression_metrics.mae(pred_tensor, target_tensor)
                metrics[f"{key}_rmse"] = self.regression_metrics.rmse(pred_tensor, target_tensor)
                metrics[f"{key}_r2"] = self.regression_metrics.r2_score(pred_tensor, target_tensor)
                metrics[f"{key}_pearson"] = self.regression_metrics.pearson_correlation(
                    pred_tensor, target_tensor
                )
        
        return metrics













