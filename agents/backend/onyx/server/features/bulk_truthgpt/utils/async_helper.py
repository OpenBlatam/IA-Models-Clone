"""
Async Helper Utilities
======================

Advanced async utilities for better performance and resource management.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Callable, TypeVar, Tuple
from functools import wraps
from datetime import datetime
import time

logger = logging.getLogger(__name__)

T = TypeVar('T')

class AsyncSemaphore:
    """Async semaphore with priority support."""
    
    def __init__(self, limit: int):
        self.limit = limit
        self.semaphore = asyncio.Semaphore(limit)
        self.waiting = []
        self.active = 0
    
    async def acquire(self, priority: int = 0):
        """Acquire semaphore with optional priority."""
        await self.semaphore.acquire()
        self.active += 1
        return self
    
    async def release(self):
        """Release semaphore."""
        self.semaphore.release()
        self.active -= 1
    
    async def __aenter__(self):
        await self.acquire()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.release()

class AsyncQueue:
    """Enhanced async queue with priority and timeout."""
    
    def __init__(self, maxsize: int = 0):
        self.queue = asyncio.Queue(maxsize=maxsize)
        self.priority_queue = asyncio.PriorityQueue(maxsize=maxsize)
        self.use_priority = False
    
    async def put(self, item: Any, priority: int = 0):
        """Put item with optional priority."""
        if self.use_priority:
            await self.priority_queue.put((priority, time.time(), item))
        else:
            await self.queue.put(item)
    
    async def get(self, timeout: Optional[float] = None) -> Any:
        """Get item with optional timeout."""
        try:
            if self.use_priority:
                _, _, item = await asyncio.wait_for(
                    self.priority_queue.get(),
                    timeout=timeout
                )
            else:
                item = await asyncio.wait_for(
                    self.queue.get(),
                    timeout=timeout
                )
            return item
        except asyncio.TimeoutError:
            raise TimeoutError(f"Queue get timeout after {timeout}s")
    
    def qsize(self) -> int:
        """Get queue size."""
        if self.use_priority:
            return self.priority_queue.qsize()
        return self.queue.qsize()
    
    def empty(self) -> bool:
        """Check if queue is empty."""
        if self.use_priority:
            return self.priority_queue.empty()
        return self.queue.empty()

class AsyncPool:
    """Async worker pool for parallel processing."""
    
    def __init__(self, size: int = 5):
        self.size = size
        self.semaphore = asyncio.Semaphore(size)
        self.active_tasks = 0
    
    async def execute(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function in pool."""
        async with self.semaphore:
            self.active_tasks += 1
            try:
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    # Run in executor for sync functions
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(None, func, *args, **kwargs)
                return result
            finally:
                self.active_tasks -= 1
    
    async def execute_batch(
        self,
        items: List[Any],
        func: Callable,
        *args,
        **kwargs
    ) -> List[Any]:
        """Execute function on batch of items."""
        tasks = [
            self.execute(func, item, *args, **kwargs)
            for item in items
        ]
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        return {
            "size": self.size,
            "active_tasks": self.active_tasks,
            "available": self.size - self.active_tasks
        }

def async_timeout(seconds: float):
    """Decorator to add timeout to async functions."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=seconds
                )
            except asyncio.TimeoutError:
                logger.error(f"{func.__name__} timed out after {seconds}s")
                raise TimeoutError(f"Function {func.__name__} timed out")
        return wrapper
    return decorator

def async_retry(max_attempts: int = 3, delay: float = 1.0):
    """Decorator for async retry logic."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_attempts:
                        wait_time = delay * (2 ** (attempt - 1))
                        logger.warning(f"Attempt {attempt}/{max_attempts} failed, retrying in {wait_time}s...")
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(f"All {max_attempts} attempts failed")
            raise last_exception
        return wrapper
    return decorator

class AsyncThrottle:
    """Async throttle to limit execution rate."""
    
    def __init__(self, rate: float, per: float = 1.0):
        """
        Args:
            rate: Number of calls allowed
            per: Time period in seconds
        """
        self.rate = rate
        self.per = per
        self.calls = []
        self.lock = asyncio.Lock()
    
    async def acquire(self):
        """Acquire throttle permission."""
        async with self.lock:
            now = time.time()
            
            # Remove old calls
            self.calls = [t for t in self.calls if now - t < self.per]
            
            # Check if we can proceed
            if len(self.calls) >= self.rate:
                # Wait until we can proceed
                oldest = min(self.calls)
                wait_time = self.per - (now - oldest)
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
                    # Clean up again
                    now = time.time()
                    self.calls = [t for t in self.calls if now - t < self.per]
            
            # Record this call
            self.calls.append(time.time())

async def gather_with_concurrency(
    limit: int,
    *tasks,
    return_exceptions: bool = False
) -> List[Any]:
    """Gather tasks with concurrency limit."""
    semaphore = asyncio.Semaphore(limit)
    
    async def run_with_semaphore(task):
        async with semaphore:
            return await task
    
    tasks_with_semaphore = [run_with_semaphore(task) for task in tasks]
    return await asyncio.gather(*tasks_with_semaphore, return_exceptions=return_exceptions)

async def race(*tasks) -> Any:
    """Return result of first completed task, cancel others."""
    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    
    # Cancel pending tasks
    for task in pending:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
    
    # Return first result
    if done:
        return await next(iter(done))
    return None

class AsyncBatchProcessor:
    """Process items in batches with async support."""
    
    def __init__(self, batch_size: int = 10, concurrency: int = 5):
        self.batch_size = batch_size
        self.concurrency = concurrency
    
    async def process(
        self,
        items: List[Any],
        processor: Callable,
        *args,
        **kwargs
    ) -> List[Any]:
        """Process items in batches."""
        if not items:
            return []
        
        # Create batches
        batches = [
            items[i:i + self.batch_size]
            for i in range(0, len(items), self.batch_size)
        ]
        
        # Process batches with concurrency limit
        results = await gather_with_concurrency(
            self.concurrency,
            *[processor(batch, *args, **kwargs) for batch in batches],
            return_exceptions=True
        )
        
        # Flatten results
        all_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Batch processing error: {result}")
            elif isinstance(result, list):
                all_results.extend(result)
            else:
                all_results.append(result)
        
        return all_results

# Global instances
async_pool = AsyncPool(size=10)
async_throttle = AsyncThrottle(rate=10.0, per=1.0)
































