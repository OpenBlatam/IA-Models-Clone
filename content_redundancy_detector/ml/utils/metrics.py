"""
Metrics Collection
Comprehensive metrics tracking and collection
"""

import torch
import numpy as np
from typing import Dict, Any, List, Optional
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class MetricsCollector:
    """
    Collects and aggregates training metrics
    """
    
    def __init__(self):
        """Initialize metrics collector"""
        self.metrics: Dict[str, List[float]] = defaultdict(list)
        self.current_epoch = 0
    
    def update(self, metrics_dict: Dict[str, float]) -> None:
        """
        Update metrics
        
        Args:
            metrics_dict: Dictionary of metric names and values
        """
        for name, value in metrics_dict.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            self.metrics[name].append(float(value))
    
    def get_metric(self, name: str) -> List[float]:
        """
        Get metric history
        
        Args:
            name: Metric name
            
        Returns:
            List of metric values
        """
        return self.metrics.get(name, [])
    
    def get_latest(self, name: str) -> Optional[float]:
        """
        Get latest metric value
        
        Args:
            name: Metric name
            
        Returns:
            Latest value or None
        """
        values = self.metrics.get(name, [])
        return values[-1] if values else None
    
    def get_mean(self, name: str) -> Optional[float]:
        """
        Get mean of metric
        
        Args:
            name: Metric name
            
        Returns:
            Mean value or None
        """
        values = self.metrics.get(name, [])
        return np.mean(values) if values else None
    
    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """
        Get summary statistics for all metrics
        
        Returns:
            Dictionary with statistics for each metric
        """
        summary = {}
        for name, values in self.metrics.items():
            if values:
                summary[name] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'latest': float(values[-1]),
                    'count': len(values),
                }
        return summary
    
    def reset(self) -> None:
        """Reset all metrics"""
        self.metrics.clear()
        self.current_epoch = 0
    
    def set_epoch(self, epoch: int) -> None:
        """Set current epoch"""
        self.current_epoch = epoch


def calculate_classification_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    num_classes: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Calculate comprehensive classification metrics
    
    Args:
        predictions: Predicted labels
        targets: True labels
        num_classes: Number of classes (auto-detect if None)
        
    Returns:
        Dictionary with metrics
    """
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        confusion_matrix,
        classification_report,
    )
    
    accuracy = accuracy_score(targets, predictions)
    precision = precision_score(targets, predictions, average='weighted', zero_division=0)
    recall = recall_score(targets, predictions, average='weighted', zero_division=0)
    f1 = f1_score(targets, predictions, average='weighted', zero_division=0)
    
    cm = confusion_matrix(targets, predictions)
    report = classification_report(targets, predictions, output_dict=True, zero_division=0)
    
    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
    }



