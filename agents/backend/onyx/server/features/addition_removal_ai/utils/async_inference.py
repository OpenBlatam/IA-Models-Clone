"""
Async Inference for Non-Blocking Operations
"""

import asyncio
import torch
from typing import Optional, Callable, Any, List
import logging
from concurrent.futures import ThreadPoolExecutor
import queue

logger = logging.getLogger(__name__)


class AsyncInferenceEngine:
    """Async inference engine for non-blocking operations"""
    
    def __init__(
        self,
        model: torch.nn.Module,
        device: Optional[torch.device] = None,
        max_workers: int = 4,
        queue_size: int = 100
    ):
        """
        Initialize async inference engine
        
        Args:
            model: Model for inference
            device: Device to use
            max_workers: Number of worker threads
            queue_size: Maximum queue size
        """
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.queue = queue.Queue(maxsize=queue_size)
        
        logger.info(f"AsyncInferenceEngine initialized with {max_workers} workers")
    
    async def infer_async(self, input_data: Any) -> Any:
        """
        Async inference
        
        Args:
            input_data: Input data
            
        Returns:
            Inference result
        """
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self.executor,
            self._infer_sync,
            input_data
        )
        return result
    
    def _infer_sync(self, input_data: Any) -> Any:
        """Synchronous inference"""
        if isinstance(input_data, torch.Tensor):
            input_data = input_data.to(self.device)
        
        with torch.no_grad():
            if isinstance(input_data, torch.Tensor):
                result = self.model(input_data)
            else:
                result = self.model(**input_data)
        
        return result.cpu() if isinstance(result, torch.Tensor) else result
    
    async def infer_batch_async(self, inputs: List[Any]) -> List[Any]:
        """
        Async batch inference
        
        Args:
            inputs: List of inputs
            
        Returns:
            List of results
        """
        tasks = [self.infer_async(inp) for inp in inputs]
        results = await asyncio.gather(*tasks)
        return results
    
    def shutdown(self):
        """Shutdown executor"""
        self.executor.shutdown(wait=True)


class FastAsyncInference:
    """Fast async inference with batching"""
    
    def __init__(
        self,
        model: torch.nn.Module,
        batch_size: int = 32,
        device: Optional[torch.device] = None
    ):
        """
        Initialize fast async inference
        
        Args:
            model: Model for inference
            batch_size: Batch size
            device: Device to use
        """
        self.model = model
        self.batch_size = batch_size
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.model.eval()
    
    async def infer_batch(self, inputs: List[Any]) -> List[Any]:
        """
        Fast batch inference
        
        Args:
            inputs: List of inputs
            
        Returns:
            List of results
        """
        results = []
        
        for i in range(0, len(inputs), self.batch_size):
            batch = inputs[i:i+self.batch_size]
            
            # Process batch in executor
            loop = asyncio.get_event_loop()
            batch_results = await loop.run_in_executor(
                None,
                self._process_batch,
                batch
            )
            
            results.extend(batch_results)
        
        return results
    
    def _process_batch(self, batch: List[Any]) -> List[Any]:
        """Process batch synchronously"""
        # Stack tensors if needed
        if isinstance(batch[0], torch.Tensor):
            batch_tensor = torch.stack(batch).to(self.device)
        else:
            batch_tensor = batch
        
        with torch.no_grad():
            outputs = self.model(batch_tensor)
        
        # Split results
        if isinstance(outputs, torch.Tensor):
            return [outputs[i].cpu() for i in range(len(batch))]
        else:
            return outputs


def create_async_engine(
    model: torch.nn.Module,
    max_workers: int = 4
) -> AsyncInferenceEngine:
    """Factory function for async engine"""
    return AsyncInferenceEngine(model, max_workers=max_workers)

