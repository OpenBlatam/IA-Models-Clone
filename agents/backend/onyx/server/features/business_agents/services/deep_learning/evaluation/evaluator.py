"""Model evaluator for comprehensive model assessment."""

import torch
import numpy as np
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional
import logging

try:
    import structlog
    logger = structlog.get_logger(__name__)
except ImportError:
    logger = logging.getLogger(__name__)

from .metrics import compute_metrics


class ModelEvaluator:
    """Comprehensive model evaluator."""
    
    def __init__(self, model: torch.nn.Module, device: torch.device):
        """
        Initialize evaluator.
        
        Args:
            model: Model to evaluate
            device: Target device
        """
        self.model = model
        self.device = device
        self.model.eval()
    
    def evaluate(
        self,
        dataloader: DataLoader,
        compute_roc: bool = True,
        compute_pr: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate model on dataset.
        
        Args:
            dataloader: Data loader
            compute_roc: Whether to compute ROC curve
            compute_pr: Whether to compute PR curve
        
        Returns:
            Dictionary of evaluation metrics
        """
        all_preds = []
        all_labels = []
        all_probas = []
        
        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                probas = torch.softmax(output, dim=1)
                preds = output.argmax(dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(target.cpu().numpy())
                all_probas.extend(probas.cpu().numpy())
        
        # Convert to numpy
        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)
        y_proba = np.array(all_probas)
        
        # Compute metrics
        metrics = compute_metrics(
            y_true, y_pred, y_proba,
            compute_roc=compute_roc,
            compute_pr=compute_pr
        )
        
        logger.info(f"Evaluation metrics: {metrics}")
        
        return metrics



