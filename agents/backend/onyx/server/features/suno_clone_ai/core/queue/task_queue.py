"""
Task Queue

Utilities for task queue management.
"""

import logging
import queue
import threading
from typing import Any, Optional, Callable
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class Task:
    """Task data class."""
    id: str
    func: Callable
    args: tuple
    kwargs: dict
    priority: int = 0
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


class TaskQueue:
    """Task queue manager."""
    
    def __init__(
        self,
        maxsize: int = 100,
        timeout: Optional[float] = None
    ):
        """
        Initialize task queue.
        
        Args:
            maxsize: Maximum queue size
            timeout: Timeout for operations
        """
        self.queue = queue.Queue(maxsize=maxsize)
        self.timeout = timeout
        self.processed = 0
        self.failed = 0
    
    def enqueue(
        self,
        task_id: str,
        func: Callable,
        *args,
        priority: int = 0,
        **kwargs
    ) -> bool:
        """
        Enqueue a task.
        
        Args:
            task_id: Task identifier
            func: Function to execute
            *args: Function arguments
            priority: Task priority
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
            self.queue.put(task, timeout=self.timeout)
            logger.debug(f"Enqueued task: {task_id}")
            return True
        except queue.Full:
            logger.warning(f"Queue full, cannot enqueue task: {task_id}")
            return False
    
    def dequeue(self) -> Optional[Task]:
        """
        Dequeue a task.
        
        Returns:
            Task or None
        """
        try:
            task = self.queue.get(timeout=self.timeout)
            return task
        except queue.Empty:
            return None
    
    def execute(self, task: Task) -> Any:
        """
        Execute a task.
        
        Args:
            task: Task to execute
            
        Returns:
            Task result
        """
        try:
            result = task.func(*task.args, **task.kwargs)
            self.processed += 1
            logger.debug(f"Executed task: {task.id}")
            return result
        except Exception as e:
            self.failed += 1
            logger.error(f"Task {task.id} failed: {e}")
            raise
    
    def size(self) -> int:
        """Get queue size."""
        return self.queue.qsize()
    
    def empty(self) -> bool:
        """Check if queue is empty."""
        return self.queue.empty()
    
    def full(self) -> bool:
        """Check if queue is full."""
        return self.queue.full()
    
    def get_stats(self) -> dict:
        """Get queue statistics."""
        return {
            'size': self.size(),
            'processed': self.processed,
            'failed': self.failed,
            'empty': self.empty(),
            'full': self.full()
        }


def enqueue_task(
    queue: TaskQueue,
    task_id: str,
    func: Callable,
    *args,
    **kwargs
) -> bool:
    """Enqueue task."""
    return queue.enqueue(task_id, func, *args, **kwargs)


def dequeue_task(queue: TaskQueue) -> Optional[Task]:
    """Dequeue task."""
    return queue.dequeue()


def get_queue_size(queue: TaskQueue) -> int:
    """Get queue size."""
    return queue.size()



