"""
Async Processing System - High-performance async operations
Optimized for speed and concurrency
"""

import asyncio
import time
from typing import Any, List, Dict, Callable, Optional, Union, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
import queue
import threading

@dataclass
class ProcessingResult:
    """Result of async processing operation"""
    success: bool
    result: Any = None
    error: Optional[Exception] = None
    processing_time: float = 0.0
    worker_id: Optional[str] = None

class AsyncProcessor:
    """High-performance async processor with multiple execution strategies"""
    
    def __init__(
        self,
        max_workers: int = None,
        use_threads: bool = True,
        use_processes: bool = False,
        queue_size: int = 1000
    ):
        self.max_workers = max_workers or min(32, (mp.cpu_count() or 1) + 4)
        self.use_threads = use_threads
        self.use_processes = use_processes
        self.queue_size = queue_size
        
        # Execution pools
        self.thread_pool: Optional[ThreadPoolExecutor] = None
        self.process_pool: Optional[ProcessPoolExecutor] = None
        
        # Task queues
        self.task_queue: asyncio.Queue = asyncio.Queue(maxsize=queue_size)
        self.result_queue: asyncio.Queue = asyncio.Queue()
        
        # Workers
        self.workers: List[asyncio.Task] = []
        self.running = False
        
        # Statistics
        self.processed_tasks = 0
        self.failed_tasks = 0
        self.total_processing_time = 0.0

    async def start(self):
        """Start the async processor"""
        if self.running:
            return
        
        self.running = True
        
        # Initialize execution pools
        if self.use_threads:
            self.thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)
        
        if self.use_processes:
            self.process_pool = ProcessPoolExecutor(max_workers=self.max_workers)
        
        # Start worker tasks
        for i in range(self.max_workers):
            worker = asyncio.create_task(self._worker(f"worker-{i}"))
            self.workers.append(worker)

    async def stop(self):
        """Stop the async processor"""
        if not self.running:
            return
        
        self.running = False
        
        # Cancel workers
        for worker in self.workers:
            worker.cancel()
        
        # Wait for workers to finish
        await asyncio.gather(*self.workers, return_exceptions=True)
        self.workers.clear()
        
        # Shutdown execution pools
        if self.thread_pool:
            self.thread_pool.shutdown(wait=True)
            self.thread_pool = None
        
        if self.process_pool:
            self.process_pool.shutdown(wait=True)
            self.process_pool = None

    async def _worker(self, worker_id: str):
        """Worker coroutine that processes tasks"""
        while self.running:
            try:
                # Get task from queue with timeout
                task = await asyncio.wait_for(
                    self.task_queue.get(), 
                    timeout=1.0
                )
                
                # Process task
                result = await self._process_task(task, worker_id)
                
                # Put result in result queue
                await self.result_queue.put(result)
                
                # Mark task as done
                self.task_queue.task_done()
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                print(f"Worker {worker_id} error: {e}")

    async def _process_task(self, task: Dict[str, Any], worker_id: str) -> ProcessingResult:
        """Process a single task"""
        start_time = time.time()
        
        try:
            func = task["func"]
            args = task.get("args", ())
            kwargs = task.get("kwargs", {})
            execution_type = task.get("execution_type", "async")
            
            # Execute based on type
            if execution_type == "async":
                result = await func(*args, **kwargs)
            elif execution_type == "thread" and self.thread_pool:
                result = await asyncio.get_event_loop().run_in_executor(
                    self.thread_pool, func, *args, **kwargs
                )
            elif execution_type == "process" and self.process_pool:
                result = await asyncio.get_event_loop().run_in_executor(
                    self.process_pool, func, *args, **kwargs
                )
            else:
                result = await func(*args, **kwargs)
            
            processing_time = time.time() - start_time
            self.processed_tasks += 1
            self.total_processing_time += processing_time
            
            return ProcessingResult(
                success=True,
                result=result,
                processing_time=processing_time,
                worker_id=worker_id
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.failed_tasks += 1
            
            return ProcessingResult(
                success=False,
                error=e,
                processing_time=processing_time,
                worker_id=worker_id
            )

    async def submit(
        self, 
        func: Callable, 
        *args, 
        execution_type: str = "async",
        **kwargs
    ) -> str:
        """Submit a task for processing"""
        task_id = f"task-{int(time.time() * 1000)}"
        
        task = {
            "id": task_id,
            "func": func,
            "args": args,
            "kwargs": kwargs,
            "execution_type": execution_type,
            "submitted_at": time.time()
        }
        
        await self.task_queue.put(task)
        return task_id

    async def get_result(self, timeout: float = None) -> Optional[ProcessingResult]:
        """Get next result from processing"""
        try:
            if timeout:
                result = await asyncio.wait_for(
                    self.result_queue.get(), 
                    timeout=timeout
                )
            else:
                result = await self.result_queue.get()
            
            return result
        except asyncio.TimeoutError:
            return None

    async def process_batch(
        self, 
        tasks: List[Tuple[Callable, tuple, dict]], 
        max_concurrent: int = None
    ) -> List[ProcessingResult]:
        """Process a batch of tasks concurrently"""
        if not max_concurrent:
            max_concurrent = min(len(tasks), self.max_workers)
        
        # Create semaphore to limit concurrency
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_with_semaphore(task_data):
            async with semaphore:
                func, args, kwargs = task_data
                return await self._process_task({
                    "func": func,
                    "args": args,
                    "kwargs": kwargs,
                    "execution_type": "async"
                }, "batch-worker")
        
        # Process all tasks concurrently
        results = await asyncio.gather(
            *[process_with_semaphore(task) for task in tasks],
            return_exceptions=True
        )
        
        # Convert exceptions to ProcessingResult
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                processed_results.append(ProcessingResult(
                    success=False,
                    error=result
                ))
            else:
                processed_results.append(result)
        
        return processed_results

    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        avg_processing_time = (
            self.total_processing_time / self.processed_tasks 
            if self.processed_tasks > 0 else 0
        )
        
        return {
            "processed_tasks": self.processed_tasks,
            "failed_tasks": self.failed_tasks,
            "success_rate": (
                self.processed_tasks / (self.processed_tasks + self.failed_tasks)
                if (self.processed_tasks + self.failed_tasks) > 0 else 0
            ),
            "avg_processing_time": avg_processing_time,
            "total_processing_time": self.total_processing_time,
            "queue_size": self.task_queue.qsize(),
            "result_queue_size": self.result_queue.qsize(),
            "active_workers": len(self.workers),
            "running": self.running
        }

class BatchProcessor:
    """Specialized batch processor for bulk operations"""
    
    def __init__(self, batch_size: int = 100, max_wait_time: float = 1.0):
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time
        self.pending_tasks: List[Dict[str, Any]] = []
        self.last_batch_time = time.time()
        self.processor = AsyncProcessor()

    async def start(self):
        """Start the batch processor"""
        await self.processor.start()
        asyncio.create_task(self._batch_processor())

    async def stop(self):
        """Stop the batch processor"""
        # Process remaining tasks
        if self.pending_tasks:
            await self._process_batch()
        
        await self.processor.stop()

    async def add_task(self, func: Callable, *args, **kwargs):
        """Add task to batch"""
        task = {
            "func": func,
            "args": args,
            "kwargs": kwargs,
            "added_at": time.time()
        }
        
        self.pending_tasks.append(task)
        
        # Process if batch is full
        if len(self.pending_tasks) >= self.batch_size:
            await self._process_batch()

    async def _batch_processor(self):
        """Background task to process batches by time"""
        while True:
            try:
                await asyncio.sleep(0.1)  # Check every 100ms
                
                current_time = time.time()
                time_since_last_batch = current_time - self.last_batch_time
                
                # Process if we have tasks and enough time has passed
                if (self.pending_tasks and 
                    time_since_last_batch >= self.max_wait_time):
                    await self._process_batch()
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Batch processor error: {e}")

    async def _process_batch(self):
        """Process current batch of tasks"""
        if not self.pending_tasks:
            return
        
        # Extract task data
        tasks = [
            (task["func"], task["args"], task["kwargs"])
            for task in self.pending_tasks
        ]
        
        # Process batch
        results = await self.processor.process_batch(tasks)
        
        # Clear pending tasks
        self.pending_tasks.clear()
        self.last_batch_time = time.time()
        
        return results

# Utility functions for common async patterns

async def parallel_map(
    func: Callable, 
    items: List[Any], 
    max_concurrent: int = None
) -> List[Any]:
    """Apply function to items in parallel"""
    processor = AsyncProcessor(max_workers=max_concurrent)
    await processor.start()
    
    try:
        # Submit all tasks
        task_ids = []
        for item in items:
            task_id = await processor.submit(func, item)
            task_ids.append(task_id)
        
        # Collect results
        results = []
        for _ in range(len(task_ids)):
            result = await processor.get_result()
            if result and result.success:
                results.append(result.result)
            else:
                results.append(None)
        
        return results
    finally:
        await processor.stop()

async def async_filter(
    func: Callable, 
    items: List[Any], 
    max_concurrent: int = None
) -> List[Any]:
    """Filter items using async function"""
    processor = AsyncProcessor(max_workers=max_concurrent)
    await processor.start()
    
    try:
        # Submit all tasks
        task_ids = []
        for item in items:
            task_id = await processor.submit(func, item)
            task_ids.append(task_id)
        
        # Collect results and filter
        results = []
        for i, _ in enumerate(task_ids):
            result = await processor.get_result()
            if result and result.success and result.result:
                results.append(items[i])
        
        return results
    finally:
        await processor.stop()

async def async_reduce(
    func: Callable, 
    items: List[Any], 
    initial: Any = None,
    max_concurrent: int = None
) -> Any:
    """Reduce items using async function"""
    if not items:
        return initial
    
    if len(items) == 1:
        return items[0]
    
    processor = AsyncProcessor(max_workers=max_concurrent)
    await processor.start()
    
    try:
        current_items = items.copy()
        
        # Reduce in pairs until we have one result
        while len(current_items) > 1:
            # Process pairs
            pairs = [
                (current_items[i], current_items[i + 1])
                for i in range(0, len(current_items) - 1, 2)
            ]
            
            # Add odd item if exists
            if len(current_items) % 2 == 1:
                pairs.append((current_items[-1], initial))
            
            # Process pairs in parallel
            pair_results = await processor.process_batch([
                (func, (pair[0], pair[1]), {}) for pair in pairs
            ])
            
            # Update current items with results
            current_items = [
                result.result for result in pair_results
                if result.success and result.result is not None
            ]
        
        return current_items[0] if current_items else initial
    finally:
        await processor.stop()





