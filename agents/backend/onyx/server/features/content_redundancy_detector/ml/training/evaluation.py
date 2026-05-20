"""
Evaluation Module
Modular evaluation utilities for model assessment
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, List
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from tqdm import tqdm


class ModelEvaluator:
    """
    Model Evaluator
    Handles model evaluation with comprehensive metrics
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        use_mixed_precision: bool = False,
    ):
        """
        Initialize evaluator
        
        Args:
            model: Model to evaluate
            device: Evaluation device
            use_mixed_precision: Use mixed precision
        """
        self.model = model
        self.device = device
        self.use_mixed_precision = use_mixed_precision
        self.model.eval()
    
    def evaluate(
        self,
        data_loader: DataLoader,
        criterion: Optional[nn.Module] = None,
        return_predictions: bool = False,
    ) -> Dict[str, Any]:
        """
        Evaluate model on dataset
        
        Args:
            data_loader: Data loader
            criterion: Loss function (optional)
            return_predictions: Whether to return predictions
            
        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()
        
        all_preds = []
        all_targets = []
        all_probs = []
        total_loss = 0.0
        
        with torch.no_grad():
            for inputs, targets in tqdm(data_loader, desc="Evaluating"):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Forward pass
                if self.use_mixed_precision and self.device.type == "cuda":
                    with torch.cuda.amp.autocast():
                        outputs = self.model(inputs)
                        if criterion:
                            loss = criterion(outputs, targets)
                            total_loss += loss.item()
                else:
                    outputs = self.model(inputs)
                    if criterion:
                        loss = criterion(outputs, targets)
                        total_loss += loss.item()
                
                # Get predictions
                probs = torch.softmax(outputs, dim=1)
                _, preds = outputs.max(1)
                
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # Calculate metrics
        metrics = self._calculate_metrics(
            all_targets,
            all_preds,
            all_probs,
            total_loss / len(data_loader) if criterion else None,
        )
        
        if return_predictions:
            metrics["predictions"] = all_preds
            metrics["probabilities"] = all_probs
        
        return metrics
    
    def _calculate_metrics(
        self,
        targets: List[int],
        predictions: List[int],
        probabilities: List[np.ndarray],
        loss: Optional[float],
    ) -> Dict[str, Any]:
        """
        Calculate evaluation metrics
        
        Args:
            targets: True labels
            predictions: Predicted labels
            probabilities: Prediction probabilities
            loss: Average loss (optional)
            
        Returns:
            Dictionary with metrics
        """
        targets = np.array(targets)
        predictions = np.array(predictions)
        
        # Basic metrics
        accuracy = accuracy_score(targets, predictions)
        precision = precision_score(targets, predictions, average='weighted', zero_division=0)
        recall = recall_score(targets, predictions, average='weighted', zero_division=0)
        f1 = f1_score(targets, predictions, average='weighted', zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(targets, predictions)
        
        # Classification report
        report = classification_report(
            targets,
            predictions,
            output_dict=True,
            zero_division=0
        )
        
        # Per-class metrics
        num_classes = len(np.unique(targets))
        per_class_metrics = {}
        for i in range(num_classes):
            per_class_metrics[f"class_{i}"] = {
                "precision": report.get(str(i), {}).get("precision", 0.0),
                "recall": report.get(str(i), {}).get("recall", 0.0),
                "f1_score": report.get(str(i), {}).get("f1-score", 0.0),
                "support": report.get(str(i), {}).get("support", 0),
            }
        
        metrics = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
            "per_class_metrics": per_class_metrics,
        }
        
        if loss is not None:
            metrics["loss"] = float(loss)
        
        return metrics
    
    def predict_batch(
        self,
        inputs: torch.Tensor,
        return_probabilities: bool = True,
    ) -> Dict[str, Any]:
        """
        Predict on a batch of inputs
        
        Args:
            inputs: Input tensor
            return_probabilities: Whether to return probabilities
            
        Returns:
            Dictionary with predictions
        """
        self.model.eval()
        inputs = inputs.to(self.device)
        
        with torch.no_grad():
            if self.use_mixed_precision and self.device.type == "cuda":
                with torch.cuda.amp.autocast():
                    outputs = self.model(inputs)
            else:
                outputs = self.model(inputs)
            
            probs = torch.softmax(outputs, dim=1)
            _, preds = outputs.max(1)
        
        result = {
            "predictions": preds.cpu().numpy().tolist(),
        }
        
        if return_probabilities:
            result["probabilities"] = probs.cpu().numpy().tolist()
            result["logits"] = outputs.cpu().numpy().tolist()
        
        return result



