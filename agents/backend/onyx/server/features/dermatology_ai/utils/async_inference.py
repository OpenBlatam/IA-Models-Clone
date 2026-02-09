"""
Async Inference Engine
For fast parallel inference processing
"""

import asyncio
import torch
import torch.nn as nn
from typing import List, Optional, Callable, Any, Dict
from concurrent.futures import ThreadPoolExecutor
import logging
from queue import Queue
import threading

logger = logging.getLogger(__name__)


class AsyncInferenceEngine:
    """
    Async inference engine for parallel processing
    Uses multiple workers for concurrent inference
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda",
        num_workers: int = 2,
        batch_size: int = 1,
        max_queue_size: int = 100
    ):
        self.model = model.to(device)
        self.device = torch.device(device)
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.max_queue_size = max_queue_size
        
        self.model.eval()
        
        # Queue for tasks
        self.task_queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue_size)
        self.result_queue: Queue = Queue()
        
        # Workers
        self.workers: List[threading.Thread] = []
        self.running = False
        
        # Thread pool for CPU-bound operations
        self.executor = ThreadPoolExecutor(max_workers=num_workers)
    
    async def start(self):
        """Start async inference engine"""
        if self.running:
            return
        
        self.running = True
        
        # Start worker threads
        for i in range(self.num_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                args=(i,),
                daemon=True
            )
            worker.start()
            self.workers.append(worker)
        
        logger.info(f"Async inference engine started with {self.num_workers} workers")
    
    async def stop(self):
        """Stop async inference engine"""
        self.running = False
        
        # Wait for queue to empty
        await self.task_queue.join()
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        logger.info("Async inference engine stopped")
    
    def _worker_loop(self, worker_id: int):
        """Worker loop for processing inference tasks"""
        while self.running:
            try:
                # Get task from queue (blocking)
                task = self.task_queue.get_nowait()
                
                # Process inference
                result = self._process_inference(task)
                
                # Put result
                self.result_queue.put((task['task_id'], result))
                
                self.task_queue.task_done()
            except asyncio.QueueEmpty:
                import time
                time.sleep(0.01)  # Small delay
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
                self.task_queue.task_done()
    
    def _process_inference(self, task: Dict[str, Any]) -> Any:
        """Process single inference task"""
        input_tensor = task['input'].to(self.device, non_blocking=True)
        
        with torch.no_grad():
            if self.device.type == "cuda":
                with torch.cuda.amp.autocast():
                    output = self.model(input_tensor)
            else:
                output = self.model(input_tensor)
        
        return output.cpu()
    
    async def predict_async(
        self,
        input_tensor: torch.Tensor,
        task_id: Optional[str] = None
    ) -> Any:
        """Async prediction"""
        if not self.running:
            await self.start()
        
        task_id = task_id or f"task_{id(input_tensor)}"
        
        # Create task
        task = {
            'task_id': task_id,
            'input': input_tensor
        }
        
        # Submit task
        await self.task_queue.put(task)
        
        # Wait for result (simplified - in production would use callbacks)
        import time
        while True:
            try:
                result_id, result = self.result_queue.get_nowait()
                if result_id == task_id:
                    return result
                else:
                    # Put back if not our result
                    self.result_queue.put((result_id, result))
            except:
                await asyncio.sleep(0.001)
    
    async def predict_batch_async(
        self,
        input_batch: List[torch.Tensor]
    ) -> List[Any]:
        """Async batch prediction"""
        tasks = [
            self.predict_async(tensor, f"task_{i}")
            for i, tensor in enumerate(input_batch)
        ]
        
        results = await asyncio.gather(*tasks)
        return results


class BatchInferenceEngine:
    """
    Optimized batch inference engine
    Automatically batches requests for efficiency
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda",
        max_batch_size: int = 32,
        timeout: float = 0.1
    ):
        self.model = model.to(device)
        self.device = torch.device(device)
        self.max_batch_size = max_batch_size
        self.timeout = timeout
        
        self.model.eval()
        
        self.pending_requests: List[Dict[str, Any]] = []
        self.lock = threading.Lock()
    
    def predict(
        self,
        input_tensor: torch.Tensor,
        callback: Optional[Callable] = None
    ):
        """Add prediction request to batch"""
        request = {
            'input': input_tensor,
            'callback': callback,
            'future': asyncio.Future() if callback is None else None
        }
        
        with self.lock:
            self.pending_requests.append(request)
            
            # Process batch if full
            if len(self.pending_requests) >= self.max_batch_size:
                self._process_batch()
    
    def _process_batch(self):
        """Process pending batch"""
        if not self.pending_requests:
            return
        
        with self.lock:
            batch = self.pending_requests[:self.max_batch_size]
            self.pending_requests = self.pending_requests[self.max_batch_size:]
        
        # Stack inputs
        inputs = torch.stack([req['input'] for req in batch])
        inputs = inputs.to(self.device, non_blocking=True)
        
        # Inference
        with torch.no_grad():
            if self.device.type == "cuda":
                with torch.cuda.amp.autocast():
                    outputs = self.model(inputs)
            else:
                outputs = self.model(inputs)
        
        outputs = outputs.cpu()
        
        # Call callbacks or set futures
        for i, req in enumerate(batch):
            output = outputs[i]
            if req['callback']:
                req['callback'](output)
            elif req['future']:
                req['future'].set_result(output)
    
    async def flush(self):
        """Flush remaining requests"""
        while self.pending_requests:
            self._process_batch()
            await asyncio.sleep(0.01)













