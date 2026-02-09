"""
Worker pool utilities
Worker pool patterns
"""

from typing import TypeVar, Callable, Optional, List, Any
from asyncio import Queue, create_task, gather
import asyncio

T = TypeVar('T')
U = TypeVar('U')


class WorkerPool:
    """
    Worker pool for parallel processing
    """
    
    def __init__(self, worker_count: int, worker_func: Callable[[T], U]):
        self.worker_count = worker_count
        self.worker_func = worker_func
        self.queue: Queue[T] = Queue()
        self.workers: List[asyncio.Task] = []
        self.running = False
    
    async def start(self) -> None:
        """Start worker pool"""
        if self.running:
            return
        
        self.running = True
        self.workers = [
            create_task(self._worker())
            for _ in range(self.worker_count)
        ]
    
    async def stop(self) -> None:
        """Stop worker pool"""
        self.running = False
        
        # Add sentinel values to stop workers
        for _ in range(self.worker_count):
            await self.queue.put(None)
        
        await gather(*self.workers, return_exceptions=True)
        self.workers.clear()
    
    async def submit(self, item: T) -> None:
        """Submit item to worker pool"""
        await self.queue.put(item)
    
    async def _worker(self) -> None:
        """Worker task"""
        while self.running:
            item = await self.queue.get()
            
            if item is None:  # Sentinel value
                break
            
            try:
                if asyncio.iscoroutinefunction(self.worker_func):
                    await self.worker_func(item)
                else:
                    self.worker_func(item)
            except Exception:
                pass  # Ignore worker errors
    
    async def submit_batch(self, items: List[T]) -> None:
        """Submit batch of items"""
        for item in items:
            await self.submit(item)


def create_worker_pool(
    worker_count: int,
    worker_func: Callable[[T], U]
) -> WorkerPool:
    """Create new worker pool"""
    return WorkerPool(worker_count, worker_func)

