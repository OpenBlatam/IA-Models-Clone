"""
Queue Utilities for Piel Mejorador AI SAM3
==========================================

Unified queue and priority queue utilities.
"""

import asyncio
import logging
import heapq
from typing import TypeVar, Generic, Optional, Callable, Any, List
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class PriorityItem:
    """Item with priority."""
    priority: int
    item: Any
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __lt__(self, other):
        """Compare by priority (lower priority = higher priority)."""
        if self.priority != other.priority:
            return self.priority < other.priority
        return self.timestamp < other.timestamp


class PriorityQueue(Generic[T]):
    """Priority queue implementation."""
    
    def __init__(self):
        """Initialize priority queue."""
        self._heap: List[PriorityItem] = []
        self._lock = asyncio.Lock()
    
    async def put(self, item: T, priority: int = 0):
        """
        Put item in queue.
        
        Args:
            item: Item to add
            priority: Priority (lower = higher priority)
        """
        async with self._lock:
            heapq.heappush(self._heap, PriorityItem(priority=priority, item=item))
            logger.debug(f"Added item to priority queue with priority {priority}")
    
    async def get(self) -> Optional[T]:
        """
        Get highest priority item.
        
        Returns:
            Item or None if empty
        """
        async with self._lock:
            if not self._heap:
                return None
            return heapq.heappop(self._heap).item
    
    async def peek(self) -> Optional[T]:
        """
        Peek at highest priority item without removing.
        
        Returns:
            Item or None if empty
        """
        async with self._lock:
            if not self._heap:
                return None
            return self._heap[0].item
    
    async def size(self) -> int:
        """Get queue size."""
        async with self._lock:
            return len(self._heap)
    
    async def empty(self) -> bool:
        """Check if queue is empty."""
        async with self._lock:
            return len(self._heap) == 0
    
    async def clear(self):
        """Clear queue."""
        async with self._lock:
            self._heap.clear()


class AsyncQueue(Generic[T]):
    """Thread-safe async queue."""
    
    def __init__(self, maxsize: int = 0):
        """
        Initialize async queue.
        
        Args:
            maxsize: Maximum size (0 = unlimited)
        """
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
    
    async def put(self, item: T):
        """
        Put item in queue.
        
        Args:
            item: Item to add
        """
        await self._queue.put(item)
        logger.debug("Added item to async queue")
    
    async def get(self) -> T:
        """
        Get item from queue.
        
        Returns:
            Item
        """
        return await self._queue.get()
    
    async def get_nowait(self) -> T:
        """
        Get item without waiting.
        
        Returns:
            Item
            
        Raises:
            asyncio.QueueEmpty: If queue is empty
        """
        return self._queue.get_nowait()
    
    def put_nowait(self, item: T):
        """
        Put item without waiting.
        
        Args:
            item: Item to add
            
        Raises:
            asyncio.QueueFull: If queue is full
        """
        self._queue.put_nowait(item)
    
    async def size(self) -> int:
        """Get queue size."""
        return self._queue.qsize()
    
    async def empty(self) -> bool:
        """Check if queue is empty."""
        return self._queue.empty()
    
    async def full(self) -> bool:
        """Check if queue is full."""
        return self._queue.full()
    
    async def clear(self):
        """Clear queue."""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break


class QueueUtils:
    """Unified queue utilities."""
    
    @staticmethod
    def create_priority_queue() -> PriorityQueue:
        """
        Create priority queue.
        
        Returns:
            PriorityQueue
        """
        return PriorityQueue()
    
    @staticmethod
    def create_async_queue(maxsize: int = 0) -> AsyncQueue:
        """
        Create async queue.
        
        Args:
            maxsize: Maximum size (0 = unlimited)
            
        Returns:
            AsyncQueue
        """
        return AsyncQueue(maxsize=maxsize)
    
    @staticmethod
    def create_deque(maxlen: Optional[int] = None) -> deque:
        """
        Create deque.
        
        Args:
            maxlen: Optional maximum length
            
        Returns:
            Deque
        """
        return deque(maxlen=maxlen)


# Convenience functions
def create_priority_queue() -> PriorityQueue:
    """Create priority queue."""
    return QueueUtils.create_priority_queue()


def create_async_queue(maxsize: int = 0) -> AsyncQueue:
    """Create async queue."""
    return QueueUtils.create_async_queue(maxsize)




