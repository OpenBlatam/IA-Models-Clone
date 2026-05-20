"""
Model Evaluation and Metrics
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Callable
import logging
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
import numpy as np

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Model evaluator with comprehensive metrics"""
    
    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None,
        use_mixed_precision: bool = True
    ):
        """
        Initialize evaluator
        
        Args:
            model: Model to evaluate
            device: Device to use
            use_mixed_precision: Use mixed precision
        """
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_mixed_precision = use_mixed_precision and torch.cuda.is_available()
        self.model = self.model.to(self.device)
        self.model.eval()
    
    def evaluate(
        self,
        data_loader: DataLoader,
        criterion: Optional[nn.Module] = None,
        metrics: Optional[List[str]] = None
    ) -> Dict:
        """
        Evaluate model
        
        Args:
            data_loader: Data loader
            criterion: Loss function
            metrics: List of metrics to compute
            
        Returns:
            Evaluation results
        """
        if metrics is None:
            metrics = ["accuracy", "precision", "recall", "f1"]
        
        all_predictions = []
        all_targets = []
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in data_loader:
                if isinstance(batch, (list, tuple)):
                    inputs = batch[0].to(self.device)
                    targets = batch[1].to(self.device)
                else:
                    inputs = batch.to(self.device)
                    targets = None
                
                # Forward pass
                if self.use_mixed_precision:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(inputs)
                else:
                    outputs = self.model(inputs)
                
                # Compute loss
                if criterion is not None and targets is not None:
                    loss = criterion(outputs, targets)
                    total_loss += loss.item()
                    num_batches += 1
                
                # Get predictions
                if isinstance(outputs, torch.Tensor):
                    if outputs.dim() > 1:
                        predictions = torch.argmax(outputs, dim=-1).cpu().numpy()
                    else:
                        predictions = (outputs > 0.5).cpu().numpy().astype(int)
                else:
                    predictions = outputs
                
                if targets is not None:
                    all_predictions.extend(predictions)
                    all_targets.extend(targets.cpu().numpy())
        
        results = {}
        
        # Compute metrics
        if len(all_predictions) > 0 and len(all_targets) > 0:
            all_predictions = np.array(all_predictions)
            all_targets = np.array(all_targets)
            
            if "accuracy" in metrics:
                results["accuracy"] = accuracy_score(all_targets, all_predictions)
            
            if "precision" in metrics:
                results["precision"] = precision_score(
                    all_targets, all_predictions, average="weighted", zero_division=0
                )
            
            if "recall" in metrics:
                results["recall"] = recall_score(
                    all_targets, all_predictions, average="weighted", zero_division=0
                )
            
            if "f1" in metrics:
                results["f1"] = f1_score(
                    all_targets, all_predictions, average="weighted", zero_division=0
                )
            
            if "confusion_matrix" in metrics:
                results["confusion_matrix"] = confusion_matrix(
                    all_targets, all_predictions
                ).tolist()
            
            if "classification_report" in metrics:
                results["classification_report"] = classification_report(
                    all_targets, all_predictions, output_dict=True
                )
        
        if criterion is not None and num_batches > 0:
            results["loss"] = total_loss / num_batches
        
        return results


class EarlyStopping:
    """Early stopping callback"""
    
    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.0,
        mode: str = "min",
        restore_best_weights: bool = True
    ):
        """
        Initialize early stopping
        
        Args:
            patience: Number of epochs to wait
            min_delta: Minimum change to qualify as improvement
            mode: "min" or "max"
            restore_best_weights: Restore best weights on stop
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        self.stopped_epoch = 0
    
    def __call__(self, score: float, model: nn.Module) -> bool:
        """
        Check if training should stop
        
        Args:
            score: Current score
            model: Model to save weights from
            
        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = score
            self._save_weights(model)
        elif self._is_better(score, self.best_score):
            self.best_score = score
            self.counter = 0
            self._save_weights(model)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stopped_epoch = True
                if self.restore_best_weights:
                    self._restore_weights(model)
                return True
        
        return False
    
    def _is_better(self, current: float, best: float) -> bool:
        """Check if current score is better"""
        if self.mode == "min":
            return current < best - self.min_delta
        else:
            return current > best + self.min_delta
    
    def _save_weights(self, model: nn.Module):
        """Save model weights"""
        self.best_weights = model.state_dict().copy()
    
    def _restore_weights(self, model: nn.Module):
        """Restore best weights"""
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)
            logger.info("Restored best model weights")


def create_evaluator(
    model: nn.Module,
    device: Optional[torch.device] = None
) -> ModelEvaluator:
    """Factory function to create evaluator"""
    return ModelEvaluator(model=model, device=device)

