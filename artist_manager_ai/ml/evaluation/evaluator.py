"""
Model Evaluator
===============

Model evaluation utilities.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from pathlib import Path
import logging

from .metrics import compute_metrics

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Model evaluator for testing and validation."""
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        task_type: str = "regression"
    ):
        """
        Initialize evaluator.
        
        Args:
            model: PyTorch model
            device: Device
            task_type: "regression" or "classification"
        """
        self.model = model.to(device)
        self.device = device
        self.task_type = task_type
        self.model.eval()
    
    def evaluate(
        self,
        dataloader: torch.utils.data.DataLoader,
        return_predictions: bool = False
    ) -> Dict[str, Any]:
        """
        Evaluate model on dataset.
        
        Args:
            dataloader: DataLoader
            return_predictions: Whether to return predictions
        
        Returns:
            Evaluation results
        """
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for features, targets in dataloader:
                features = features.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                outputs = self.model(features)
                
                # Store predictions and targets
                all_predictions.append(outputs.cpu())
                all_targets.append(targets.cpu())
        
        # Concatenate all predictions and targets
        predictions = torch.cat(all_predictions, dim=0)
        targets = torch.cat(all_targets, dim=0)
        
        # Compute metrics
        metrics = compute_metrics(predictions, targets, self.task_type)
        
        result = {"metrics": metrics}
        
        if return_predictions:
            result["predictions"] = predictions
            result["targets"] = targets
        
        return result
    
    def predict(self, features: torch.Tensor) -> torch.Tensor:
        """
        Make predictions.
        
        Args:
            features: Input features
        
        Returns:
            Predictions
        """
        self.model.eval()
        features = features.to(self.device)
        
        with torch.no_grad():
            predictions = self.model(features)
        
        return predictions.cpu()




