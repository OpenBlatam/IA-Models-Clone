"""
Concurrency Utilities
=====================

Utilities for managing concurrent operations.
"""

import threading
import queue
import logging
from typing import Callable, Any, Optional, List, Dict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)


@dataclass
class TaskResult:
    """Task execution result."""
    success: bool
    result: Any = None
    error: Optional[Exception] = None
    execution_time: float = 0.0


class ThreadPool:
    """Thread pool manager."""
    
    def __init__(
        self,
        max_workers: int = 4,
        thread_name_prefix: str = "worker"
    ):
        """
        Initialize thread pool.
        
        Args:
            max_workers: Maximum number of worker threads
            thread_name_prefix: Prefix for thread names
        """
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix=thread_name_prefix
        )
        self._logger = logger
    
    def submit(self, func: Callable, *args, **kwargs):
        """Submit task to thread pool."""
        return self.executor.submit(func, *args, **kwargs)
    
    def map(self, func: Callable, iterable):
        """Map function over iterable using thread pool."""
        return self.executor.map(func, iterable)
    
    def shutdown(self, wait: bool = True):
        """Shutdown thread pool."""
        self.executor.shutdown(wait=wait)
    
    def execute_batch(
        self,
        tasks: List[Callable],
        timeout: Optional[float] = None
    ) -> List[TaskResult]:
        """
        Execute batch of tasks.
        
        Args:
            tasks: List of callable tasks
            timeout: Optional timeout in seconds
        
        Returns:
            List of task results
        """
        futures = [self.submit(task) for task in tasks]
        results = []
        
        for future in as_completed(futures, timeout=timeout):
            start_time = time.time()
            try:
                result = future.result()
                execution_time = time.time() - start_time
                results.append(TaskResult(
                    success=True,
                    result=result,
                    execution_time=execution_time
                ))
            except Exception as e:
                execution_time = time.time() - start_time
                results.append(TaskResult(
                    success=False,
                    error=e,
                    execution_time=execution_time
                ))
        
        return results


class ProcessPool:
    """Process pool manager."""
    
    def __init__(self, max_workers: Optional[int] = None):
        """
        Initialize process pool.
        
        Args:
            max_workers: Maximum number of worker processes
        """
        self.max_workers = max_workers
        self.executor = ProcessPoolExecutor(max_workers=max_workers)
        self._logger = logger
    
    def submit(self, func: Callable, *args, **kwargs):
        """Submit task to process pool."""
        return self.executor.submit(func, *args, **kwargs)
    
    def map(self, func: Callable, iterable):
        """Map function over iterable using process pool."""
        return self.executor.map(func, iterable)
    
    def shutdown(self, wait: bool = True):
        """Shutdown process pool."""
        self.executor.shutdown(wait=wait)


class TaskQueue:
    """Task queue for background processing."""
    
    def __init__(self, max_size: int = 100):
        """
        Initialize task queue.
        
        Args:
            max_size: Maximum queue size
        """
        self.queue = queue.Queue(maxsize=max_size)
        self.workers: List[threading.Thread] = []
        self.running = False
        self._logger = logger
    
    def start(self, num_workers: int = 2):
        """
        Start worker threads.
        
        Args:
            num_workers: Number of worker threads
        """
        self.running = True
        
        for i in range(num_workers):
            worker = threading.Thread(
                target=self._worker,
                name=f"task-worker-{i}",
                daemon=True
            )
            worker.start()
            self.workers.append(worker)
        
        self._logger.info(f"Started {num_workers} worker threads")
    
    def stop(self, wait: bool = True):
        """
        Stop worker threads.
        
        Args:
            wait: Wait for tasks to complete
        """
        self.running = False
        
        # Add sentinel values to wake up workers
        for _ in self.workers:
            self.queue.put(None)
        
        if wait:
            for worker in self.workers:
                worker.join()
        
        self.workers.clear()
        self._logger.info("Stopped worker threads")
    
    def enqueue(self, task: Callable, *args, **kwargs):
        """
        Enqueue task.
        
        Args:
            task: Task function
            *args: Task arguments
            **kwargs: Task keyword arguments
        """
        self.queue.put((task, args, kwargs))
    
    def _worker(self):
        """Worker thread function."""
        while self.running:
            try:
                item = self.queue.get(timeout=1.0)
                
                if item is None:  # Sentinel value
                    break
                
                task, args, kwargs = item
                try:
                    task(*args, **kwargs)
                except Exception as e:
                    self._logger.error(f"Task execution error: {str(e)}")
                finally:
                    self.queue.task_done()
            except queue.Empty:
                continue


class RateLimiter:
    """Rate limiter for function calls."""
    
    def __init__(self, max_calls: int, time_window: float):
        """
        Initialize rate limiter.
        
        Args:
            max_calls: Maximum number of calls
            time_window: Time window in seconds
        """
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = []
        self.lock = threading.Lock()
    
    def acquire(self) -> bool:
        """
        Try to acquire permission to call.
        
        Returns:
            True if allowed, False if rate limited
        """
        with self.lock:
            now = time.time()
            
            # Remove old calls outside time window
            self.calls = [t for t in self.calls if now - t < self.time_window]
            
            if len(self.calls) < self.max_calls:
                self.calls.append(now)
                return True
            
            return False
    
    def wait(self):
        """Wait until rate limit allows."""
        while not self.acquire():
            time.sleep(0.1)




