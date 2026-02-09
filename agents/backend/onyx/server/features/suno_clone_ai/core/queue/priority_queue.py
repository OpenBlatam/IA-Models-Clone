"""
Priority Queue

Utilities for priority queue management.
"""

import logging
import queue
from typing import Any, Optional, Callable
from core.queue.task_queue import Task

logger = logging.getLogger(__name__)


class PriorityQueue:
    """Priority queue manager."""
    
    def __init__(
        self,
        maxsize: int = 100
    ):
        """
        Initialize priority queue.
        
        Args:
            maxsize: Maximum queue size
        """
        self.queue = queue.PriorityQueue(maxsize=maxsize)
    
    def enqueue(
        self,
        task_id: str,
        func: Callable,
        *args,
        priority: int = 0,
        **kwargs
    ) -> bool:
        """
        Enqueue a task with priority.
        
        Args:
            task_id: Task identifier
            func: Function to execute
            *args: Function arguments
            priority: Task priority (lower = higher priority)
            **kwargs: Function keyword arguments
            
        Returns:
            True if enqueued successfully
        """
        task = Task(
            id=task_id,
            func=func,
            args=args,
            kwargs=kwargs,
            priority=priority
        )
        
        try:
            # Priority queue uses (priority, item) tuples
            # Lower priority number = higher priority
            self.queue.put((priority, task))
            logger.debug(f"Enqueued priority task: {task_id} (priority: {priority})")
            return True
        except queue.Full:
            logger.warning(f"Priority queue full, cannot enqueue task: {task_id}")
            return False
    
    def dequeue(self) -> Optional[Task]:
        """
        Dequeue highest priority task.
        
        Returns:
            Task or None
        """
        try:
            priority, task = self.queue.get_nowait()
            return task
        except queue.Empty:
            return None
    
    def size(self) -> int:
        """Get queue size."""
        return self.queue.qsize()
    
    def empty(self) -> bool:
        """Check if queue is empty."""
        return self.queue.empty()


def enqueue_priority(
    queue: PriorityQueue,
    task_id: str,
    func: Callable,
    priority: int = 0,
    *args,
    **kwargs
) -> bool:
    """Enqueue priority task."""
    return queue.enqueue(task_id, func, *args, priority=priority, **kwargs)


def dequeue_priority(queue: PriorityQueue) -> Optional[Task]:
    """Dequeue priority task."""
    return queue.dequeue()



