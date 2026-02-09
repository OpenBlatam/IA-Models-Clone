"""Queue utilities."""

from typing import Any, Optional
from collections import deque
import asyncio


class AsyncQueue:
    """Async queue with size limit."""
    
    def __init__(self, maxsize: int = 0):
        """
        Initialize async queue.
        
        Args:
            maxsize: Maximum queue size (0 = unlimited)
        """
        self.queue = asyncio.Queue(maxsize=maxsize)
    
    async def put(self, item: Any) -> None:
        """Put item in queue."""
        await self.queue.put(item)
    
    async def get(self) -> Any:
        """Get item from queue."""
        return await self.queue.get()
    
    def qsize(self) -> int:
        """Get queue size."""
        return self.queue.qsize()
    
    def empty(self) -> bool:
        """Check if queue is empty."""
        return self.queue.empty()
    
    def full(self) -> bool:
        """Check if queue is full."""
        return self.queue.full()


class PriorityQueue:
    """Simple priority queue."""
    
    def __init__(self):
        self.items = []
    
    def put(self, item: Any, priority: float = 0.0) -> None:
        """
        Put item with priority.
        
        Args:
            item: Item to add
            priority: Priority (higher = more important)
        """
        self.items.append((priority, item))
        self.items.sort(reverse=True, key=lambda x: x[0])
    
    def get(self) -> Optional[Any]:
        """
        Get highest priority item.
        
        Returns:
            Item or None if empty
        """
        if not self.items:
            return None
        return self.items.pop(0)[1]
    
    def empty(self) -> bool:
        """Check if queue is empty."""
        return len(self.items) == 0
    
    def size(self) -> int:
        """Get queue size."""
        return len(self.items)


class CircularQueue:
    """Circular queue with fixed size."""
    
    def __init__(self, maxsize: int):
        """
        Initialize circular queue.
        
        Args:
            maxsize: Maximum size
        """
        self.maxsize = maxsize
        self.queue = deque(maxlen=maxsize)
    
    def put(self, item: Any) -> None:
        """Put item in queue (removes oldest if full)."""
        self.queue.append(item)
    
    def get(self) -> Optional[Any]:
        """
        Get item from queue.
        
        Returns:
            Item or None if empty
        """
        if not self.queue:
            return None
        return self.queue.popleft()
    
    def peek(self) -> Optional[Any]:
        """
        Peek at front item without removing.
        
        Returns:
            Front item or None
        """
        if not self.queue:
            return None
        return self.queue[0]
    
    def empty(self) -> bool:
        """Check if queue is empty."""
        return len(self.queue) == 0
    
    def full(self) -> bool:
        """Check if queue is full."""
        return len(self.queue) >= self.maxsize
    
    def size(self) -> int:
        """Get queue size."""
        return len(self.queue)

