"""
Real-time Streaming for Recovery AI
"""

import asyncio
import torch
from typing import Dict, List, Optional, Any, Callable, AsyncIterator
from collections import deque
import logging
import time

logger = logging.getLogger(__name__)


class StreamProcessor:
    """Process streaming data"""
    
    def __init__(
        self,
        model: torch.nn.Module,
        batch_size: int = 32,
        window_size: int = 100
    ):
        """
        Initialize stream processor
        
        Args:
            model: Model for processing
            batch_size: Batch size for processing
            window_size: Window size for buffering
        """
        self.model = model
        self.batch_size = batch_size
        self.window_size = window_size
        self.buffer = deque(maxlen=window_size)
        self.model.eval()
        
        logger.info("StreamProcessor initialized")
    
    async def process_stream(
        self,
        data_stream: AsyncIterator[torch.Tensor]
    ) -> AsyncIterator[torch.Tensor]:
        """
        Process streaming data
        
        Args:
            data_stream: Async iterator of data
        
        Returns:
            Async iterator of predictions
        """
        async for data in data_stream:
            self.buffer.append(data)
            
            # Process when buffer is full
            if len(self.buffer) >= self.batch_size:
                batch = torch.stack(list(self.buffer))
                self.buffer.clear()
                
                with torch.no_grad():
                    predictions = self.model(batch)
                
                for pred in predictions:
                    yield pred
    
    def process_batch(self, batch: List[torch.Tensor]) -> List[torch.Tensor]:
        """Process batch of data"""
        batch_tensor = torch.stack(batch)
        
        with torch.no_grad():
            predictions = self.model(batch_tensor)
        
        return [pred for pred in predictions]


class RealTimePredictor:
    """Real-time prediction system"""
    
    def __init__(
        self,
        model: torch.nn.Module,
        device: Optional[torch.device] = None,
        max_queue_size: int = 1000
    ):
        """
        Initialize real-time predictor
        
        Args:
            model: Model for prediction
            device: Device to use
            max_queue_size: Maximum queue size
        """
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.queue = asyncio.Queue(maxsize=max_queue_size)
        self.running = False
        
        logger.info("RealTimePredictor initialized")
    
    async def start(self):
        """Start prediction worker"""
        self.running = True
        asyncio.create_task(self._worker())
    
    async def _worker(self):
        """Worker for processing queue"""
        while self.running:
            try:
                item_id, inputs, future = await asyncio.wait_for(
                    self.queue.get(),
                    timeout=1.0
                )
                
                # Process
                inputs = inputs.to(self.device)
                with torch.no_grad():
                    output = self.model(inputs)
                
                # Set result
                future.set_result(output.cpu())
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Worker error: {e}")
    
    async def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Predict asynchronously
        
        Args:
            inputs: Input tensor
        
        Returns:
            Prediction result
        """
        future = asyncio.Future()
        item_id = f"{time.time()}_{id(inputs)}"
        
        await self.queue.put((item_id, inputs, future))
        return await future
    
    def stop(self):
        """Stop predictor"""
        self.running = False

