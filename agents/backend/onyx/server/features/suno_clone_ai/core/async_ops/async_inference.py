"""
Async Inference

Utilities for asynchronous model inference.
"""

import logging
import asyncio
import torch
import torch.nn as nn
from typing import List, Any, Optional, Callable
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class AsyncInference:
    """Asynchronous inference handler."""
    
    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None,
        max_workers: int = 4
    ):
        """
        Initialize async inference.
        
        Args:
            model: Model for inference
            device: Device to use
            max_workers: Maximum worker threads
        """
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
    
    async def predict(
        self,
        input_data: Any,
        **kwargs
    ) -> Any:
        """
        Async prediction.
        
        Args:
            input_data: Input data
            **kwargs: Additional arguments
            
        Returns:
            Prediction result
        """
        loop = asyncio.get_event_loop()
        
        # Run inference in thread pool
        result = await loop.run_in_executor(
            self.executor,
            self._predict_sync,
            input_data,
            kwargs
        )
        
        return result
    
    def _predict_sync(
        self,
        input_data: Any,
        kwargs: dict
    ) -> Any:
        """Synchronous prediction."""
        if isinstance(input_data, torch.Tensor):
            input_data = input_data.to(self.device)
        
        with torch.no_grad():
            output = self.model(input_data, **kwargs)
        
        return output
    
    async def predict_batch(
        self,
        batch_data: List[Any],
        **kwargs
    ) -> List[Any]:
        """
        Async batch prediction.
        
        Args:
            batch_data: List of inputs
            **kwargs: Additional arguments
            
        Returns:
            List of predictions
        """
        # Process in parallel
        tasks = [self.predict(item, **kwargs) for item in batch_data]
        results = await asyncio.gather(*tasks)
        
        return results
    
    def __del__(self):
        """Cleanup executor."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)


async def async_predict(
    model: nn.Module,
    input_data: Any,
    device: Optional[torch.device] = None,
    **kwargs
) -> Any:
    """Async prediction convenience function."""
    inference = AsyncInference(model, device)
    return await inference.predict(input_data, **kwargs)


async def async_predict_batch(
    model: nn.Module,
    batch_data: List[Any],
    device: Optional[torch.device] = None,
    **kwargs
) -> List[Any]:
    """Async batch prediction convenience function."""
    inference = AsyncInference(model, device)
    return await inference.predict_batch(batch_data, **kwargs)



