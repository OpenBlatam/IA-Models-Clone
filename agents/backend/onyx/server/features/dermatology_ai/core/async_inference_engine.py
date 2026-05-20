"""
Async Inference Engine - High-performance async inference with parallel processing
"""

import asyncio
from typing import List, Dict, Any, Optional, Callable, Awaitable
from dataclasses import dataclass
from collections import deque
import logging
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class InferenceTask:
    """Task for async inference"""
    task_id: str
    model_id: str
    input_data: Any
    priority: int = 0
    callback: Optional[Callable] = None
    created_at: float = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()


@dataclass
class InferenceResult:
    """Result of async inference"""
    task_id: str
    prediction: Any
    processing_time: float
    success: bool = True
    error: Optional[str] = None


class AsyncInferenceEngine:
    """
    High-performance async inference engine:
    - Parallel processing
    - Priority queue
    - Batch aggregation
    - Connection pooling
    - Auto-scaling workers
    """
    
    def __init__(
        self,
        max_workers: int = 4,
        max_queue_size: int = 1000,
        batch_timeout: float = 0.1,
        use_process_pool: bool = False
    ):
        self.max_workers = max_workers
        self.max_queue_size = max_queue_size
        self.batch_timeout = batch_timeout
        self.use_process_pool = use_process_pool
        
        # Queues
        self.task_queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue_size)
        self.priority_queue: deque = deque()
        self.batch_queue: Dict[str, List[InferenceTask]] = {}
        
        # Workers
        self.workers: List[asyncio.Task] = []
        self.executor = (
            ProcessPoolExecutor(max_workers=max_workers)
            if use_process_pool
            else ThreadPoolExecutor(max_workers=max_workers)
        )
        
        # Statistics
        self.stats = {
            "tasks_processed": 0,
            "tasks_failed": 0,
            "avg_processing_time": 0.0,
            "queue_size": 0,
            "active_workers": 0
        }
        
        self.running = False
    
    async def start(self):
        """Start the inference engine"""
        if self.running:
            return
        
        self.running = True
        # Start worker tasks
        for i in range(self.max_workers):
            worker = asyncio.create_task(self._worker(f"worker-{i}"))
            self.workers.append(worker)
        
        # Start batch aggregator
        asyncio.create_task(self._batch_aggregator())
        
        logger.info(f"Async inference engine started with {self.max_workers} workers")
    
    async def stop(self):
        """Stop the inference engine"""
        self.running = False
        
        # Wait for queue to empty
        await self.task_queue.join()
        
        # Cancel workers
        for worker in self.workers:
            worker.cancel()
        
        await asyncio.gather(*self.workers, return_exceptions=True)
        self.executor.shutdown(wait=True)
        
        logger.info("Async inference engine stopped")
    
    async def submit(
        self,
        model_id: str,
        input_data: Any,
        priority: int = 0,
        task_id: Optional[str] = None
    ) -> str:
        """Submit an inference task"""
        if not self.running:
            await self.start()
        
        task_id = task_id or f"task_{int(time.time() * 1000000)}"
        task = InferenceTask(
            task_id=task_id,
            model_id=model_id,
            input_data=input_data,
            priority=priority
        )
        
        try:
            if priority > 0:
                self.priority_queue.append(task)
                self.priority_queue = deque(sorted(self.priority_queue, key=lambda x: x.priority, reverse=True))
            else:
                await self.task_queue.put(task)
            
            self.stats["queue_size"] = self.task_queue.qsize() + len(self.priority_queue)
            return task_id
        
        except asyncio.QueueFull:
            raise RuntimeError("Task queue is full")
    
    async def submit_batch(
        self,
        model_id: str,
        input_batch: List[Any],
        priority: int = 0
    ) -> List[str]:
        """Submit a batch of inference tasks"""
        task_ids = []
        for input_data in input_batch:
            task_id = await self.submit(model_id, input_data, priority)
            task_ids.append(task_id)
        return task_ids
    
    async def get_result(self, task_id: str, timeout: Optional[float] = None) -> InferenceResult:
        """Get result for a task (polling)"""
        # This would typically use a result store or callback
        # For now, simplified version
        await asyncio.sleep(0.01)  # Small delay for processing
        return InferenceResult(
            task_id=task_id,
            prediction=None,
            processing_time=0.0,
            success=False
        )
    
    async def _worker(self, worker_id: str):
        """Worker coroutine for processing tasks"""
        logger.info(f"Worker {worker_id} started")
        
        while self.running:
            try:
                # Get task from priority queue or regular queue
                if self.priority_queue:
                    task = self.priority_queue.popleft()
                else:
                    task = await asyncio.wait_for(
                        self.task_queue.get(),
                        timeout=1.0
                    )
                
                self.stats["active_workers"] += 1
                self.stats["queue_size"] = self.task_queue.qsize() + len(self.priority_queue)
                
                # Process task
                result = await self._process_task(task)
                
                if result.success:
                    self.stats["tasks_processed"] += 1
                else:
                    self.stats["tasks_failed"] += 1
                
                self.task_queue.task_done()
                self.stats["active_workers"] -= 1
            
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {str(e)}", exc_info=True)
                self.stats["tasks_failed"] += 1
                self.stats["active_workers"] -= 1
    
    async def _process_task(self, task: InferenceTask) -> InferenceResult:
        """Process a single inference task"""
        start_time = time.time()
        
        try:
            # Run inference in executor (non-blocking)
            loop = asyncio.get_event_loop()
            prediction = await loop.run_in_executor(
                self.executor,
                self._run_inference,
                task.model_id,
                task.input_data
            )
            
            processing_time = time.time() - start_time
            
            # Update stats
            total_time = self.stats["avg_processing_time"] * (self.stats["tasks_processed"] - 1)
            self.stats["avg_processing_time"] = (total_time + processing_time) / self.stats["tasks_processed"]
            
            return InferenceResult(
                task_id=task.task_id,
                prediction=prediction,
                processing_time=processing_time,
                success=True
            )
        
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Error processing task {task.task_id}: {str(e)}")
            return InferenceResult(
                task_id=task.task_id,
                prediction=None,
                processing_time=processing_time,
                success=False,
                error=str(e)
            )
    
    def _run_inference(self, model_id: str, input_data: Any) -> Any:
        """Run actual inference (synchronous, runs in executor)"""
        # This would call the actual model manager
        # Placeholder implementation
        import time
        time.sleep(0.01)  # Simulate inference
        return {"prediction": "result"}
    
    async def _batch_aggregator(self):
        """Aggregate tasks into batches for efficient processing"""
        while self.running:
            try:
                await asyncio.sleep(self.batch_timeout)
                
                # Group tasks by model_id
                batches: Dict[str, List[InferenceTask]] = {}
                
                # Collect tasks from queue
                tasks_to_process = []
                while not self.task_queue.empty():
                    try:
                        task = self.task_queue.get_nowait()
                        tasks_to_process.append(task)
                    except asyncio.QueueEmpty:
                        break
                
                # Group by model
                for task in tasks_to_process:
                    if task.model_id not in batches:
                        batches[task.model_id] = []
                    batches[task.model_id].append(task)
                
                # Process batches
                for model_id, batch in batches.items():
                    if len(batch) > 1:
                        await self._process_batch(model_id, batch)
                    else:
                        # Put back single tasks
                        for task in batch:
                            await self.task_queue.put(task)
            
            except Exception as e:
                logger.error(f"Batch aggregator error: {str(e)}")
    
    async def _process_batch(self, model_id: str, batch: List[InferenceTask]):
        """Process a batch of tasks"""
        try:
            input_batch = [task.input_data for task in batch]
            
            # Run batch inference
            loop = asyncio.get_event_loop()
            predictions = await loop.run_in_executor(
                self.executor,
                self._run_batch_inference,
                model_id,
                input_batch
            )
            
            # Create results for each task
            for task, prediction in zip(batch, predictions):
                result = InferenceResult(
                    task_id=task.task_id,
                    prediction=prediction,
                    processing_time=0.0,
                    success=True
                )
                if task.callback:
                    await task.callback(result)
        
        except Exception as e:
            logger.error(f"Batch processing error: {str(e)}")
    
    def _run_batch_inference(self, model_id: str, input_batch: List[Any]) -> List[Any]:
        """Run batch inference (synchronous)"""
        # Placeholder - would call actual batch inference
        return [{"prediction": "result"} for _ in input_batch]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics"""
        return {
            **self.stats,
            "running": self.running,
            "queue_size": self.task_queue.qsize() + len(self.priority_queue)
        }


