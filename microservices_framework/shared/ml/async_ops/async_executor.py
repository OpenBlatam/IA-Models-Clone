"""
Async Executor
Async execution utilities for blocking operations.
"""

import asyncio
from typing import Callable, Any, Optional, TypeVar, Coroutine
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import logging

logger = logging.getLogger(__name__)

T = TypeVar("T")


class AsyncExecutor:
    """Execute blocking operations asynchronously."""
    
    def __init__(
        self,
        max_workers: int = 4,
        executor_type: str = "thread",
    ):
        self.max_workers = max_workers
        self.executor_type = executor_type
        
        if executor_type == "thread":
            self.executor = ThreadPoolExecutor(max_workers=max_workers)
        elif executor_type == "process":
            self.executor = ProcessPoolExecutor(max_workers=max_workers)
        else:
            raise ValueError(f"Unknown executor type: {executor_type}")
    
    async def run(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Run function asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            lambda: func(*args, **kwargs)
        )
    
    async def run_batch(
        self,
        func: Callable[..., T],
        items: list,
        batch_size: int = 10,
    ) -> list:
        """Run function on batch of items asynchronously."""
        batches = [
            items[i:i + batch_size]
            for i in range(0, len(items), batch_size)
        ]
        
        tasks = [
            self.run(func, batch)
            for batch in batches
        ]
        
        results = await asyncio.gather(*tasks)
        
        # Flatten results
        flattened = []
        for batch_results in results:
            if isinstance(batch_results, list):
                flattened.extend(batch_results)
            else:
                flattened.append(batch_results)
        
        return flattened
    
    def shutdown(self, wait: bool = True):
        """Shutdown executor."""
        self.executor.shutdown(wait=wait)


class AsyncModelInference:
    """Async model inference wrapper."""
    
    def __init__(self, model: Any, executor: Optional[AsyncExecutor] = None):
        self.model = model
        self.executor = executor or AsyncExecutor()
    
    async def predict(self, input_data: Any) -> Any:
        """Async prediction."""
        return await self.executor.run(self.model, input_data)
    
    async def predict_batch(self, input_batch: list) -> list:
        """Async batch prediction."""
        return await self.executor.run_batch(
            lambda batch: [self.model(item) for item in batch],
            input_batch,
        )


def asyncify(func: Callable) -> Callable:
    """Decorator to make function async."""
    async def wrapper(*args, **kwargs):
        executor = AsyncExecutor()
        try:
            return await executor.run(func, *args, **kwargs)
        finally:
            executor.shutdown()
    return wrapper



