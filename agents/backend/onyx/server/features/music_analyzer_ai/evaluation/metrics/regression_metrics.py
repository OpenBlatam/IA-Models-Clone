"""
Regression Metrics

Specialized metrics for regression tasks.
"""

from typing import Dict
import torch
import numpy as np

try:
    from sklearn.metrics import (
        mean_squared_error,
        mean_absolute_error,
        r2_score,
        mean_squared_log_error
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class RegressionMetrics:
    """
    Regression metrics calculator.
    
    Computes various regression metrics including:
    - MSE, MAE, RMSE
    - R² score
    - Mean absolute percentage error
    """
    
    def __init__(self):
        """Initialize regression metrics."""
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
            predictions: Model predictions [batch, ...]
            targets: Target values [batch, ...]
        """
        # Convert to numpy
        if isinstance(predictions, torch.Tensor):
            preds = predictions.detach().cpu().numpy().flatten()
        else:
            preds = np.array(predictions).flatten()
        
        if isinstance(targets, torch.Tensor):
            targets_np = targets.detach().cpu().numpy().flatten()
        else:
            targets_np = np.array(targets).flatten()
        
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
        
        if SKLEARN_AVAILABLE:
            metrics = {
                "mse": mean_squared_error(targets, preds),
                "mae": mean_absolute_error(targets, preds),
                "rmse": np.sqrt(mean_squared_error(targets, preds)),
                "r2": r2_score(targets, preds),
            }
            
            # Mean absolute percentage error
            mask = targets != 0
            if mask.any():
                metrics["mape"] = np.mean(
                    np.abs((targets[mask] - preds[mask]) / targets[mask])
                ) * 100
        else:
            # Fallback to basic metrics
            mse = np.mean((targets - preds) ** 2)
            mae = np.mean(np.abs(targets - preds))
            
            metrics = {
                "mse": float(mse),
                "mae": float(mae),
                "rmse": float(np.sqrt(mse)),
            }
            
            # R² score
            ss_res = np.sum((targets - preds) ** 2)
            ss_tot = np.sum((targets - np.mean(targets)) ** 2)
            if ss_tot > 0:
                metrics["r2"] = float(1 - (ss_res / ss_tot))
        
        return metrics


__all__ = [
    "RegressionMetrics",
]



