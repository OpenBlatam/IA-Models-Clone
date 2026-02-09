"""
Model Serving

Utilities for serving models in production.
"""

import logging
from typing import Optional, Dict, Any, Callable
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ModelServer:
    """Server for model inference."""
    
    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None,
        preprocess_fn: Optional[Callable] = None,
        postprocess_fn: Optional[Callable] = None
    ):
        """
        Initialize model server.
        
        Args:
            model: Model to serve
            device: Device to run on
            preprocess_fn: Preprocessing function
            postprocess_fn: Postprocessing function
        """
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        
        self.preprocess_fn = preprocess_fn
        self.postprocess_fn = postprocess_fn
    
    def predict(
        self,
        input_data: Any,
        **kwargs
    ) -> Any:
        """
        Run inference.
        
        Args:
            input_data: Input data
            **kwargs: Additional arguments
            
        Returns:
            Model predictions
        """
        # Preprocess
        if self.preprocess_fn:
            input_data = self.preprocess_fn(input_data)
        
        # Move to device
        if isinstance(input_data, torch.Tensor):
            input_data = input_data.to(self.device)
        
        # Inference
        with torch.no_grad():
            output = self.model(input_data, **kwargs)
        
        # Postprocess
        if self.postprocess_fn:
            output = self.postprocess_fn(output)
        
        return output
    
    def predict_batch(
        self,
        batch_data: list,
        batch_size: int = 32,
        **kwargs
    ) -> list:
        """
        Run batch inference.
        
        Args:
            batch_data: List of input data
            batch_size: Batch size
            **kwargs: Additional arguments
            
        Returns:
            List of predictions
        """
        predictions = []
        
        for i in range(0, len(batch_data), batch_size):
            batch = batch_data[i:i + batch_size]
            batch_preds = [self.predict(item, **kwargs) for item in batch]
            predictions.extend(batch_preds)
        
        return predictions


def create_model_server(
    model: nn.Module,
    device: Optional[torch.device] = None,
    **kwargs
) -> ModelServer:
    """
    Create model server.
    
    Args:
        model: Model to serve
        device: Device to run on
        **kwargs: Additional server arguments
        
    Returns:
        ModelServer instance
    """
    return ModelServer(model, device, **kwargs)


def serve_model(
    model: nn.Module,
    input_data: Any,
    device: Optional[torch.device] = None,
    **kwargs
) -> Any:
    """
    Convenience function to serve model.
    
    Args:
        model: Model to serve
        input_data: Input data
        device: Device to run on
        **kwargs: Additional arguments
        
    Returns:
        Model predictions
    """
    server = create_model_server(model, device)
    return server.predict(input_data, **kwargs)



