"""
Worker Pool
Pool of workers for parallel task processing.
"""

import asyncio
import logging
import time
from typing import Any, Callable, Dict, List, Optional

from .priority_queue import PriorityTaskQueue

logger = logging.getLogger(__name__)


class WorkerPool:
    """
    Pool of workers for parallel task processing.
    Manages multiple concurrent workers efficiently.
    """
    
    def __init__(
        self,
        num_workers: int = 3,
        task_queue: Optional[PriorityTaskQueue] = None
    ):
        self.num_workers = num_workers
        self.task_queue = task_queue or PriorityTaskQueue()
        self.workers: List[asyncio.Task] = []
        self.is_running = False
        self._stop_event = asyncio.Event()
        
        # Stats
        self.tasks_processed = 0
        self.tasks_failed = 0
        self.total_processing_time = 0.0
    
    async def start(self, processor_fn: Callable[[Any], Any]) -> None:
        """Start all workers."""
        if self.is_running:
            return
        
        self.is_running = True
        self._stop_event.clear()
        
        for i in range(self.num_workers):
            worker = asyncio.create_task(
                self._worker_loop(i, processor_fn)
            )
            self.workers.append(worker)
        
        logger.info(f"Started {self.num_workers} workers")
    
    async def stop(self) -> None:
        """Stop all workers gracefully."""
        self._stop_event.set()
        self.is_running = False
        
        # Wait for workers to finish current tasks
        if self.workers:
            await asyncio.gather(*self.workers, return_exceptions=True)
            self.workers.clear()
        
        logger.info("All workers stopped")
    
    async def _worker_loop(self, worker_id: int, processor_fn: Callable) -> None:
        """Worker loop that processes tasks from queue."""
        logger.debug(f"Worker {worker_id} started")
        
        while not self._stop_event.is_set():
            try:
                task = await self.task_queue.pop()
                
                if task is None:
                    # No task available, wait a bit
                    await asyncio.sleep(0.1)
                    continue
                
                start_time = time.time()
                
                try:
                    if asyncio.iscoroutinefunction(processor_fn):
                        await processor_fn(task)
                    else:
                        loop = asyncio.get_event_loop()
                        await loop.run_in_executor(None, processor_fn, task)
                    
                    self.tasks_processed += 1
                    
                except Exception as e:
                    logger.error(f"Worker {worker_id} error: {e}")
                    self.tasks_failed += 1
                
                elapsed = time.time() - start_time
                self.total_processing_time += elapsed
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker {worker_id} unexpected error: {e}")
                await asyncio.sleep(1.0)
        
        logger.debug(f"Worker {worker_id} stopped")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get worker pool statistics."""
        avg_time = 0.0
        if self.tasks_processed > 0:
            avg_time = self.total_processing_time / self.tasks_processed
        
        return {
            "num_workers": self.num_workers,
            "is_running": self.is_running,
            "tasks_processed": self.tasks_processed,
            "tasks_failed": self.tasks_failed,
            "queue_size": len(self.task_queue),
            "average_processing_time_seconds": avg_time
        }
