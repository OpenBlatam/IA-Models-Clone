"""Worker pool utilities."""

from typing import Callable, List, TypeVar, Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import queue

T = TypeVar('T')
R = TypeVar('R')


class WorkerPool:
    """Thread pool worker."""
    
    def __init__(self, max_workers: int = 4):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self._max_workers = max_workers
    
    def submit(self, func: Callable, *args, **kwargs):
        """Submit task to pool."""
        return self.executor.submit(func, *args, **kwargs)
    
    def map(self, func: Callable, items: List[T]) -> List[R]:
        """Map function over items."""
        return list(self.executor.map(func, items))
    
    def shutdown(self, wait: bool = True) -> None:
        """Shutdown pool."""
        self.executor.shutdown(wait=wait)


class ProcessPool:
    """Process pool worker."""
    
    def __init__(self, max_workers: int = 4):
        self.executor = ProcessPoolExecutor(max_workers=max_workers)
        self._max_workers = max_workers
    
    def submit(self, func: Callable, *args, **kwargs):
        """Submit task to pool."""
        return self.executor.submit(func, *args, **kwargs)
    
    def map(self, func: Callable, items: List[T]) -> List[R]:
        """Map function over items."""
        return list(self.executor.map(func, items))
    
    def shutdown(self, wait: bool = True) -> None:
        """Shutdown pool."""
        self.executor.shutdown(wait=wait)


class AsyncWorkerPool:
    """Async worker pool."""
    
    def __init__(self, max_workers: int = 10):
        self._semaphore = asyncio.Semaphore(max_workers)
        self._max_workers = max_workers
    
    async def submit(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> R:
        """
        Submit async task.
        
        Args:
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Result
        """
        async with self._semaphore:
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            else:
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, func, *args, **kwargs)
    
    async def map(
        self,
        func: Callable,
        items: List[T]
    ) -> List[R]:
        """
        Map function over items.
        
        Args:
            func: Function to execute
            items: List of items
            
        Returns:
            List of results
        """
        tasks = [self.submit(func, item) for item in items]
        return await asyncio.gather(*tasks)


class TaskQueue:
    """Task queue with workers."""
    
    def __init__(self, num_workers: int = 4):
        self._queue = queue.Queue()
        self._workers: List = []
        self._num_workers = num_workers
        self._running = False
    
    def add_task(self, func: Callable, *args, **kwargs) -> None:
        """Add task to queue."""
        self._queue.put((func, args, kwargs))
    
    def start(self) -> None:
        """Start workers."""
        if self._running:
            return
        
        self._running = True
        
        def worker():
            while self._running:
                try:
                    func, args, kwargs = self._queue.get(timeout=1)
                    func(*args, **kwargs)
                    self._queue.task_done()
                except queue.Empty:
                    continue
        
        import threading
        self._workers = [
            threading.Thread(target=worker, daemon=True)
            for _ in range(self._num_workers)
        ]
        
        for w in self._workers:
            w.start()
    
    def stop(self) -> None:
        """Stop workers."""
        self._running = False
        for _ in self._workers:
            self._queue.put(None)
        
        for w in self._workers:
            w.join()
        
        self._workers.clear()
    
    def wait_complete(self) -> None:
        """Wait for all tasks to complete."""
        self._queue.join()



