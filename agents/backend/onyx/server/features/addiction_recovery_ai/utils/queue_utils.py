"""
Queue utilities
Advanced queue patterns
"""

from typing import TypeVar, Optional, Callable, Any
from asyncio import Queue, QueueEmpty
import asyncio

T = TypeVar('T')


class AsyncQueue:
    """
    Async queue with additional utilities
    """
    
    def __init__(self, maxsize: int = 0):
        self._queue: Queue[T] = Queue(maxsize=maxsize)
    
    async def put(self, item: T) -> None:
        """Put item in queue"""
        await self._queue.put(item)
    
    async def get(self) -> T:
        """Get item from queue"""
        return await self._queue.get()
    
    async def get_nowait(self) -> Optional[T]:
        """Get item from queue without waiting"""
        try:
            return self._queue.get_nowait()
        except QueueEmpty:
            return None
    
    def put_nowait(self, item: T) -> bool:
        """Put item in queue without waiting"""
        try:
            self._queue.put_nowait(item)
            return True
        except Exception:
            return False
    
    def qsize(self) -> int:
        """Get queue size"""
        return self._queue.qsize()
    
    def empty(self) -> bool:
        """Check if queue is empty"""
        return self._queue.empty()
    
    def full(self) -> bool:
        """Check if queue is full"""
        return self._queue.full()
    
    async def drain(self) -> List[T]:
        """Drain all items from queue"""
        items = []
        while not self._queue.empty():
            try:
                items.append(self._queue.get_nowait())
            except QueueEmpty:
                break
        return items


def create_queue(maxsize: int = 0) -> AsyncQueue:
    """Create new async queue"""
    return AsyncQueue(maxsize=maxsize)


async def process_queue(
    queue: AsyncQueue,
    processor: Callable[[T], Any],
    workers: int = 1
) -> None:
    """
    Process queue with multiple workers
    
    Args:
        queue: Queue to process
        processor: Function to process items
        workers: Number of worker tasks
    """
    async def worker():
        while True:
            item = await queue.get()
            try:
                if asyncio.iscoroutinefunction(processor):
                    await processor(item)
                else:
                    processor(item)
            except Exception:
                pass
            finally:
                queue._queue.task_done()
    
    tasks = [asyncio.create_task(worker()) for _ in range(workers)]
    await asyncio.gather(*tasks)

