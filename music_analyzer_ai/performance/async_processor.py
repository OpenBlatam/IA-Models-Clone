"""
Async Processing for Better Performance
Asynchronous processing with worker pools
"""

from typing import List, Dict, Any, Optional, Callable, Coroutine
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import time

logger = logging.getLogger(__name__)


class AsyncProcessor:
    """
    Asynchronous processor with worker pool
    """
    
    def __init__(
        self,
        max_workers: int = 4,
        use_processes: bool = False
    ):
        self.max_workers = max_workers
        self.use_processes = use_processes
        
        if use_processes:
            self.executor = ProcessPoolExecutor(max_workers=max_workers)
        else:
            self.executor = ThreadPoolExecutor(max_workers=max_workers)
    
    async def process_batch(
        self,
        items: List[Any],
        func: Callable,
        batch_size: int = 10
    ) -> List[Any]:
        """Process items in batches asynchronously"""
        results = []
        
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            
            # Process batch
            tasks = [
                asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    func,
                    item
                )
                for item in batch
            ]
            
            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)
        
        return results
    
    async def process_parallel(
        self,
        items: List[Any],
        func: Callable,
        max_concurrent: int = 10
    ) -> List[Any]:
        """Process items in parallel with concurrency limit"""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_with_semaphore(item):
            async with semaphore:
                return await asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    func,
                    item
                )
        
        tasks = [process_with_semaphore(item) for item in items]
        return await asyncio.gather(*tasks)
    
    def shutdown(self):
        """Shutdown executor"""
        self.executor.shutdown(wait=True)


class AsyncInferencePool:
    """
    Pool of inference workers for parallel processing
    """
    
    def __init__(
        self,
        model_factory: Callable,
        num_workers: int = 2,
        queue_size: int = 100
    ):
        self.model_factory = model_factory
        self.num_workers = num_workers
        self.queue_size = queue_size
        
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=queue_size)
        self.workers: List[asyncio.Task] = []
        self.running = False
    
    async def _worker(self, worker_id: int):
        """Worker coroutine"""
        model = self.model_factory()
        logger.info(f"Inference worker {worker_id} started")
        
        while self.running:
            try:
                # Get task with timeout
                item = await asyncio.wait_for(
                    self.queue.get(),
                    timeout=1.0
                )
                
                if item is None:  # Shutdown signal
                    break
                
                input_data, future = item
                
                # Process
                result = model(input_data)
                
                # Set result
                if not future.done():
                    future.set_result(result)
                
                self.queue.task_done()
            
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {str(e)}")
                if not future.done():
                    future.set_exception(e)
        
        logger.info(f"Inference worker {worker_id} stopped")
    
    async def start(self):
        """Start worker pool"""
        self.running = True
        self.workers = [
            asyncio.create_task(self._worker(i))
            for i in range(self.num_workers)
        ]
        logger.info(f"Started {self.num_workers} inference workers")
    
    async def stop(self):
        """Stop worker pool"""
        self.running = False
        
        # Send shutdown signals
        for _ in range(self.num_workers):
            await self.queue.put((None, None))
        
        # Wait for workers
        await asyncio.gather(*self.workers, return_exceptions=True)
        logger.info("Inference pool stopped")
    
    async def infer(self, input_data: Any) -> Any:
        """Run inference through pool"""
        future = asyncio.Future()
        
        try:
            await self.queue.put((input_data, future))
            result = await future
            return result
        except Exception as e:
            if not future.done():
                future.set_exception(e)
            raise

