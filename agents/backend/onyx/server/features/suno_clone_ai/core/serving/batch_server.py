"""
Batch Serving

Utilities for batch model serving.
"""

import logging
from typing import List, Optional, Callable
import torch
import torch.nn as nn
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class BatchServer:
    """Server for batch inference."""
    
    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None,
        max_workers: int = 4,
        batch_size: int = 32
    ):
        """
        Initialize batch server.
        
        Args:
            model: Model to serve
            device: Device to run on
            max_workers: Maximum worker threads
            batch_size: Batch size
        """
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
    
    def serve_batch(
        self,
        inputs: List,
        **kwargs
    ) -> List:
        """
        Serve batch of inputs.
        
        Args:
            inputs: List of inputs
            **kwargs: Additional arguments
            
        Returns:
            List of predictions
        """
        predictions = []
        
        # Process in batches
        for i in range(0, len(inputs), self.batch_size):
            batch = inputs[i:i + self.batch_size]
            
            # Prepare batch
            if isinstance(batch[0], torch.Tensor):
                batch_tensor = torch.stack(batch).to(self.device)
            else:
                batch_tensor = torch.tensor(batch).to(self.device)
            
            # Inference
            with torch.no_grad():
                batch_preds = self.model(batch_tensor, **kwargs)
            
            # Convert to list
            if isinstance(batch_preds, torch.Tensor):
                batch_preds = batch_preds.cpu().tolist()
            
            predictions.extend(batch_preds)
        
        return predictions


def serve_batch(
    model: nn.Module,
    inputs: List,
    device: Optional[torch.device] = None,
    **kwargs
) -> List:
    """
    Convenience function to serve batch.
    
    Args:
        model: Model to serve
        inputs: List of inputs
        device: Device to run on
        **kwargs: Additional arguments
        
    Returns:
        List of predictions
    """
    server = BatchServer(model, device)
    return server.serve_batch(inputs, **kwargs)



