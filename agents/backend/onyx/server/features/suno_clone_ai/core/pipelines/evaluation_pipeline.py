"""
Evaluation Pipeline

Functional pipeline for model evaluation.
"""

import logging
from typing import Dict, Any, List, Optional
import torch
import numpy as np

from ..evaluation import compute_all_metrics, AudioMetrics, TrainingMetrics

logger = logging.getLogger(__name__)


class EvaluationPipeline:
    """
    Functional evaluation pipeline.
    
    Composes evaluation steps.
    """
    
    def __init__(
        self,
        metrics: Optional[List[str]] = None
    ):
        """
        Initialize evaluation pipeline.
        
        Args:
            metrics: List of metrics to compute
        """
        self.metrics = metrics or ['mse', 'mae', 'rmse']
    
    def evaluate(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        reference_audio: Optional[np.ndarray] = None,
        generated_audio: Optional[np.ndarray] = None,
        sample_rate: int = 32000
    ) -> Dict[str, float]:
        """
        Evaluate model predictions.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            reference_audio: Reference audio (optional)
            generated_audio: Generated audio (optional)
            sample_rate: Sample rate for audio metrics
            
        Returns:
            Dictionary of metrics
        """
        return compute_all_metrics(
            predictions=predictions,
            targets=targets,
            reference_audio=reference_audio,
            generated_audio=generated_audio,
            sample_rate=sample_rate
        )
    
    def evaluate_batch(
        self,
        predictions_list: List[torch.Tensor],
        targets_list: List[torch.Tensor],
        **kwargs
    ) -> Dict[str, float]:
        """
        Evaluate batch of predictions.
        
        Args:
            predictions_list: List of predictions
            targets_list: List of targets
            **kwargs: Additional arguments
            
        Returns:
            Average metrics across batch
        """
        all_metrics = []
        
        for pred, target in zip(predictions_list, targets_list):
            metrics = self.evaluate(pred, target, **kwargs)
            all_metrics.append(metrics)
        
        # Average metrics
        avg_metrics = {}
        for key in all_metrics[0].keys():
            avg_metrics[key] = np.mean([m[key] for m in all_metrics])
        
        return avg_metrics



