"""
Custom Metrics
Specialized evaluation metrics.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
)


class MetricCalculator:
    """Calculate various metrics."""
    
    @staticmethod
    def calculate_classification_metrics(
        predictions: torch.Tensor,
        labels: torch.Tensor,
        num_classes: Optional[int] = None,
    ) -> Dict[str, float]:
        """Calculate classification metrics."""
        preds = predictions.cpu().numpy()
        labels_np = labels.cpu().numpy()
        
        if len(preds.shape) > 1:
            preds = np.argmax(preds, axis=1)
        
        accuracy = accuracy_score(labels_np, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels_np,
            preds,
            average="weighted",
            zero_division=0,
        )
        
        metrics = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }
        
        # Per-class metrics if binary or small number of classes
        if num_classes and num_classes <= 10:
            precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
                labels_np,
                preds,
                average=None,
                zero_division=0,
            )
            metrics["precision_per_class"] = precision_per_class.tolist()
            metrics["recall_per_class"] = recall_per_class.tolist()
            metrics["f1_per_class"] = f1_per_class.tolist()
        
        return metrics
    
    @staticmethod
    def calculate_regression_metrics(
        predictions: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[str, float]:
        """Calculate regression metrics."""
        preds = predictions.cpu().numpy().flatten()
        labels_np = labels.cpu().numpy().flatten()
        
        mse = mean_squared_error(labels_np, preds)
        mae = mean_absolute_error(labels_np, preds)
        rmse = np.sqrt(mse)
        
        # R-squared
        ss_res = np.sum((labels_np - preds) ** 2)
        ss_tot = np.sum((labels_np - np.mean(labels_np)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
        
        return {
            "mse": float(mse),
            "mae": float(mae),
            "rmse": float(rmse),
            "r2": float(r2),
        }
    
    @staticmethod
    def calculate_perplexity(
        logits: torch.Tensor,
        labels: torch.Tensor,
        ignore_index: int = -100,
    ) -> float:
        """Calculate perplexity for language models."""
        criterion = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction="mean")
        
        if len(logits.shape) == 3:
            logits = logits.view(-1, logits.size(-1))
            labels = labels.view(-1)
        
        loss = criterion(logits, labels)
        perplexity = torch.exp(loss).item()
        
        return perplexity
    
    @staticmethod
    def calculate_bleu_score(
        predictions: List[str],
        references: List[str],
    ) -> float:
        """Calculate BLEU score."""
        try:
            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
            
            smoothing = SmoothingFunction().method1
            scores = []
            
            for pred, ref in zip(predictions, references):
                pred_tokens = pred.split()
                ref_tokens = ref.split()
                score = sentence_bleu(
                    [ref_tokens],
                    pred_tokens,
                    smoothing_function=smoothing
                )
                scores.append(score)
            
            return float(np.mean(scores))
        except ImportError:
            return 0.0


class MetricsAggregator:
    """Aggregate metrics across batches."""
    
    def __init__(self):
        self.metrics: Dict[str, List[float]] = {}
    
    def update(self, metrics: Dict[str, float]):
        """Update aggregated metrics."""
        for key, value in metrics.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(value)
    
    def compute(self) -> Dict[str, float]:
        """Compute aggregated metrics."""
        return {
            key: float(np.mean(values))
            for key, values in self.metrics.items()
        }
    
    def reset(self):
        """Reset aggregated metrics."""
        self.metrics.clear()



