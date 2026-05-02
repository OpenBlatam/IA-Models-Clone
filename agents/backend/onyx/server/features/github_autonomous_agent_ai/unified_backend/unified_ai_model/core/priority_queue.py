"""
Priority Queue
Heap-based priority queue for efficient task scheduling.
"""

import asyncio
import heapq
import logging
from enum import Enum
from typing import Any, List, Optional, Tuple

logger = logging.getLogger(__name__)


class PriorityLevel(int, Enum):
    """Priority levels for tasks."""
    CRITICAL = 10
    HIGH = 8
    NORMAL = 5
    LOW = 3
    BACKGROUND = 1


class PriorityTaskQueue:
    """
    Priority queue for tasks with advanced scheduling.
    Uses heap-based priority queue for O(log n) insert/remove.
    """
    
    def __init__(self, max_size: int = 10000):
        self._queue: List[Tuple[int, int, Any]] = []  # (priority, counter, task)
        self._counter = 0
        self._max_size = max_size
        self._lock = asyncio.Lock()
    
    async def push(self, task: Any, priority: int = 5) -> bool:
        """
        Push task with priority.
        
        Args:
            task: Task to add
            priority: Priority 1-10 (higher = more urgent)
            
        Returns:
            True if added, False if queue full
        """
        async with self._lock:
            if len(self._queue) >= self._max_size:
                return False
            
            # Negate priority for max-heap behavior (heapq is min-heap)
            heapq.heappush(self._queue, (-priority, self._counter, task))
            self._counter += 1
            return True
    
    async def pop(self) -> Optional[Any]:
        """Pop highest priority task."""
        async with self._lock:
            if not self._queue:
                return None
            _, _, task = heapq.heappop(self._queue)
            return task
    
    async def pop_batch(self, n: int) -> List[Any]:
        """Pop up to n highest priority tasks."""
        async with self._lock:
            tasks = []
            for _ in range(min(n, len(self._queue))):
                _, _, task = heapq.heappop(self._queue)
                tasks.append(task)
            return tasks
    
    async def peek(self) -> Optional[Any]:
        """Peek at highest priority task without removing."""
        async with self._lock:
            if not self._queue:
                return None
            return self._queue[0][2]
    
    def __len__(self) -> int:
        return len(self._queue)
    
    @property
    def is_empty(self) -> bool:
        return len(self._queue) == 0
