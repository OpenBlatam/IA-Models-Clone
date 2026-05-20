"""
Postprocessing Utilities
========================

Output postprocessing utilities.
"""

import logging
from typing import Optional, Dict, Any, List, Union
import torch
import numpy as np

logger = logging.getLogger(__name__)


def format_predictions(
    predictions: Union[torch.Tensor, np.ndarray, List],
    format_type: str = 'probabilities',
    top_k: Optional[int] = None
) -> Union[Dict, List, np.ndarray]:
    """
    Format predictions for output.
    
    Args:
        predictions: Model predictions
        format_type: Output format ('probabilities', 'classes', 'logits')
        top_k: Return top-k predictions
        
    Returns:
        Formatted predictions
    """
    # Convert to numpy if tensor
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    elif isinstance(predictions, list):
        predictions = np.array(predictions)
    
    if format_type == 'probabilities':
        # Apply softmax if logits
        if predictions.ndim > 1 and predictions.shape[1] > 1:
            # Assume logits, apply softmax
            exp_preds = np.exp(predictions - np.max(predictions, axis=1, keepdims=True))
            probs = exp_preds / np.sum(exp_preds, axis=1, keepdims=True)
        else:
            # Already probabilities or binary
            probs = torch.sigmoid(torch.tensor(predictions)).numpy() if predictions.ndim == 1 else predictions
        
        if top_k:
            # Return top-k
            top_indices = np.argsort(probs, axis=-1)[:, -top_k:][:, ::-1]
            top_probs = np.take_along_axis(probs, top_indices, axis=-1)
            return {
                'indices': top_indices.tolist(),
                'probabilities': top_probs.tolist()
            }
        
        return probs
    
    elif format_type == 'classes':
        # Return class predictions
        if predictions.ndim > 1:
            classes = np.argmax(predictions, axis=-1)
        else:
            classes = (predictions > 0.5).astype(int)
        return classes.tolist() if len(classes.shape) == 0 else classes
    
    elif format_type == 'logits':
        return predictions
    
    else:
        raise ValueError(f"Unknown format type: {format_type}")


def aggregate_predictions(
    predictions: List[Union[torch.Tensor, np.ndarray]],
    method: str = 'mean'
) -> np.ndarray:
    """
    Aggregate multiple predictions.
    
    Args:
        predictions: List of predictions
        method: Aggregation method ('mean', 'max', 'vote')
        
    Returns:
        Aggregated predictions
    """
    # Convert to numpy
    preds = []
    for pred in predictions:
        if isinstance(pred, torch.Tensor):
            preds.append(pred.detach().cpu().numpy())
        else:
            preds.append(np.array(pred))
    
    preds = np.array(preds)
    
    if method == 'mean':
        return np.mean(preds, axis=0)
    
    elif method == 'max':
        return np.max(preds, axis=0)
    
    elif method == 'vote':
        # For classification, vote on classes
        if preds.ndim > 2:
            classes = np.argmax(preds, axis=-1)
        else:
            classes = (preds > 0.5).astype(int)
        
        from scipy import stats
        return stats.mode(classes, axis=0)[0].flatten()
    
    else:
        raise ValueError(f"Unknown aggregation method: {method}")


def apply_threshold(
    predictions: Union[torch.Tensor, np.ndarray],
    threshold: float = 0.5,
    return_probs: bool = False
) -> Union[np.ndarray, tuple]:
    """
    Apply threshold to predictions.
    
    Args:
        predictions: Model predictions
        threshold: Classification threshold
        return_probs: Return probabilities along with classes
        
    Returns:
        Thresholded predictions (and optionally probabilities)
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    
    # Apply sigmoid if needed (for binary classification)
    if predictions.ndim == 1 or (predictions.ndim == 2 and predictions.shape[1] == 1):
        probs = torch.sigmoid(torch.tensor(predictions)).numpy()
    else:
        # Apply softmax for multi-class
        exp_preds = np.exp(predictions - np.max(predictions, axis=-1, keepdims=True))
        probs = exp_preds / np.sum(exp_preds, axis=-1, keepdims=True)
    
    # Apply threshold
    if probs.ndim == 1:
        classes = (probs > threshold).astype(int)
    else:
        classes = np.argmax(probs, axis=-1)
    
    if return_probs:
        return classes, probs
    return classes


class PostProcessor:
    """
    Comprehensive postprocessor.
    """
    
    def __init__(
        self,
        format_type: str = 'probabilities',
        threshold: Optional[float] = None,
        top_k: Optional[int] = None
    ):
        """
        Initialize postprocessor.
        
        Args:
            format_type: Output format type
            threshold: Classification threshold
            top_k: Top-k predictions
        """
        self.format_type = format_type
        self.threshold = threshold
        self.top_k = top_k
    
    def process(
        self,
        predictions: Union[torch.Tensor, np.ndarray, List]
    ) -> Any:
        """
        Process predictions.
        
        Args:
            predictions: Model predictions
            
        Returns:
            Processed predictions
        """
        # Format
        formatted = format_predictions(predictions, self.format_type, self.top_k)
        
        # Apply threshold if specified
        if self.threshold is not None:
            if isinstance(formatted, dict):
                # Top-k format
                return formatted
            else:
                return apply_threshold(formatted, self.threshold)
        
        return formatted



