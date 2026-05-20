"""
Model Evaluator with Advanced Metrics
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, mean_squared_error,
    mean_absolute_error, r2_score
)
import logging

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Advanced model evaluator"""
    
    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None
    ):
        """
        Initialize evaluator
        
        Args:
            model: Model to evaluate
            device: Device to use
        """
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.model.eval()
    
    def evaluate_classification(
        self,
        data_loader,
        threshold: float = 0.5
    ) -> Dict[str, float]:
        """
        Evaluate classification model
        
        Args:
            data_loader: Data loader
            threshold: Classification threshold
        
        Returns:
            Dictionary of metrics
        """
        all_preds = []
        all_targets = []
        all_probs = []
        
        with torch.no_grad():
            for batch in data_loader:
                inputs = batch[0].to(self.device)
                targets = batch[1].to(self.device)
                
                outputs = self.model(inputs)
                probs = torch.sigmoid(outputs) if outputs.dim() == 1 else torch.softmax(outputs, dim=1)
                
                if probs.dim() > 1:
                    probs = probs[:, 1] if probs.shape[1] > 1 else probs[:, 0]
                
                preds = (probs >= threshold).long().cpu().numpy()
                all_preds.extend(preds)
                all_targets.extend(targets.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_targets, all_preds)
        precision = precision_score(all_targets, all_preds, zero_division=0)
        recall = recall_score(all_targets, all_preds, zero_division=0)
        f1 = f1_score(all_targets, all_preds, zero_division=0)
        
        try:
            auc = roc_auc_score(all_targets, all_probs)
        except:
            auc = 0.0
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "auc": auc
        }
    
    def evaluate_regression(self, data_loader) -> Dict[str, float]:
        """
        Evaluate regression model
        
        Args:
            data_loader: Data loader
        
        Returns:
            Dictionary of metrics
        """
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in data_loader:
                inputs = batch[0].to(self.device)
                targets = batch[1].to(self.device)
                
                outputs = self.model(inputs)
                if outputs.dim() > 1:
                    outputs = outputs.squeeze()
                
                all_preds.extend(outputs.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        # Calculate metrics
        mse = mean_squared_error(all_targets, all_preds)
        mae = mean_absolute_error(all_targets, all_preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(all_targets, all_preds)
        
        return {
            "mse": mse,
            "mae": mae,
            "rmse": rmse,
            "r2": r2
        }
    
    def evaluate_with_early_stopping(
        self,
        val_loader,
        metric: str = "loss",
        patience: int = 5,
        min_delta: float = 0.001
    ) -> bool:
        """
        Evaluate with early stopping logic
        
        Args:
            val_loader: Validation data loader
            metric: Metric to monitor
            patience: Patience for early stopping
            min_delta: Minimum change to qualify as improvement
        
        Returns:
            True if should stop
        """
        if not hasattr(self, "best_metric"):
            self.best_metric = float("inf")
            self.patience_counter = 0
        
        # Evaluate
        metrics = self.evaluate_regression(val_loader)
        current_metric = metrics.get(metric, float("inf"))
        
        # Check improvement
        if current_metric < self.best_metric - min_delta:
            self.best_metric = current_metric
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            return self.patience_counter >= patience

