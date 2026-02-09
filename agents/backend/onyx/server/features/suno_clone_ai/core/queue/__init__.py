"""
Queue Module

Provides:
- Task queue management
- Priority queues
- Queue utilities
"""

from .task_queue import (
    TaskQueue,
    enqueue_task,
    dequeue_task,
    get_queue_size
)

from .priority_queue import (
    PriorityQueue,
    enqueue_priority,
    dequeue_priority
)

__all__ = [
    # Task queue
    "TaskQueue",
    "enqueue_task",
    "dequeue_task",
    "get_queue_size",
    # Priority queue
    "PriorityQueue",
    "enqueue_priority",
    "dequeue_priority"
]



