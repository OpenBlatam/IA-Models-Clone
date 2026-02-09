"""
Postprocessing Module
Output postprocessing utilities
"""

import torch
import numpy as np
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


class PredictionPostprocessor:
    """
    Postprocess model predictions
    """
    
    def __init__(
        self,
        class_names: Optional[List[str]] = None,
        confidence_threshold: float = 0.5,
    ):
        """
        Initialize postprocessor
        
        Args:
            class_names: List of class names
            confidence_threshold: Confidence threshold
        """
        self.class_names = class_names
        self.confidence_threshold = confidence_threshold
    
    def postprocess(
        self,
        predictions: Union[torch.Tensor, np.ndarray, List],
        probabilities: Optional[Union[torch.Tensor, np.ndarray, List]] = None,
    ) -> Dict[str, Any]:
        """
        Postprocess predictions
        
        Args:
            predictions: Predicted class indices
            probabilities: Prediction probabilities
            
        Returns:
            Dictionary with postprocessed results
        """
        # Convert to numpy if needed
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        if isinstance(probabilities, torch.Tensor):
            probabilities = probabilities.cpu().numpy()
        
        # Ensure predictions is array
        if isinstance(predictions, (int, np.integer)):
            predictions = np.array([predictions])
        elif isinstance(predictions, list):
            predictions = np.array(predictions)
        
        result = {
            'predictions': predictions.tolist(),
        }
        
        # Add class names if available
        if self.class_names:
            if predictions.ndim == 0:
                result['class_name'] = self.class_names[int(predictions)]
            else:
                result['class_names'] = [self.class_names[int(p)] for p in predictions]
        
        # Add probabilities if available
        if probabilities is not None:
            if isinstance(probabilities, list):
                probabilities = np.array(probabilities)
            
            if probabilities.ndim == 1:
                probabilities = probabilities.reshape(1, -1)
            
            result['probabilities'] = probabilities.tolist()
            result['confidence'] = float(np.max(probabilities))
            
            # Filter by confidence threshold
            if result['confidence'] < self.confidence_threshold:
                result['low_confidence'] = True
        
        return result
    
    def get_top_k(
        self,
        probabilities: Union[torch.Tensor, np.ndarray],
        k: int = 5,
    ) -> Dict[str, Any]:
        """
        Get top k predictions
        
        Args:
            probabilities: Probability tensor/array
            k: Number of top predictions
            
        Returns:
            Dictionary with top k results
        """
        if isinstance(probabilities, torch.Tensor):
            probabilities = probabilities.cpu().numpy()
        
        if probabilities.ndim == 1:
            probabilities = probabilities.reshape(1, -1)
        
        top_k_indices = np.argsort(probabilities, axis=1)[:, -k:][:, ::-1]
        top_k_probs = np.sort(probabilities, axis=1)[:, -k:][:, ::-1]
        
        result = {
            'top_k_indices': top_k_indices.tolist(),
            'top_k_probabilities': top_k_probs.tolist(),
        }
        
        if self.class_names:
            result['top_k_classes'] = [
                [self.class_names[int(idx)] for idx in row]
                for row in top_k_indices
            ]
        
        return result



