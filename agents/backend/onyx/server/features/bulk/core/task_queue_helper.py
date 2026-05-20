"""
Task Queue Helper - Common task queue manipulation utilities
===========================================================

Provides common utilities for manipulating and filtering task queues.
"""

import logging
from typing import List, Any, Optional, Callable

logger = logging.getLogger(__name__)


class TaskQueueHelper:
    """Common utilities for task queue manipulation."""
    
    @staticmethod
    def filter_tasks_by_request_id(
        task_queue: List[Any],
        request_id: str
    ) -> List[Any]:
        """
        Filter tasks in queue by request ID.
        
        Args:
            task_queue: List of tasks
            request_id: Request ID to filter by
            
        Returns:
            List of tasks matching the request ID
        """
        return [task for task in task_queue if task.request_id == request_id]
    
    @staticmethod
    def remove_tasks_by_request_id(
        task_queue: List[Any],
        request_id: str
    ) -> List[Any]:
        """
        Remove all tasks for a specific request from the queue.
        
        Args:
            task_queue: List of tasks to filter
            request_id: Request ID to remove tasks for
            
        Returns:
            New list with tasks for the request removed
        """
        return [task for task in task_queue if task.request_id != request_id]
    
    @staticmethod
    def get_batch_from_queue(
        task_queue: List[Any],
        batch_size: int,
        sort_key: Optional[Callable[[Any], Any]] = None
    ) -> tuple[List[Any], List[Any]]:
        """
        Get a batch of tasks from the queue.
        
        Args:
            task_queue: List of tasks
            batch_size: Maximum number of tasks to get
            sort_key: Optional function to sort tasks before batching
            
        Returns:
            Tuple of (tasks_to_process, remaining_queue)
        """
        if not task_queue:
            return [], []
        
        queue_copy = list(task_queue)
        
        if sort_key:
            queue_copy.sort(key=sort_key)
        
        batch_size = min(batch_size, len(queue_copy))
        tasks_to_process = queue_copy[:batch_size]
        remaining_queue = queue_copy[batch_size:]
        
        return tasks_to_process, remaining_queue
    
    @staticmethod
    def count_tasks_by_request_id(
        task_queue: List[Any],
        request_id: str
    ) -> int:
        """
        Count tasks in queue for a specific request.
        
        Args:
            task_queue: List of tasks
            request_id: Request ID to count tasks for
            
        Returns:
            Number of tasks for the request
        """
        return sum(1 for task in task_queue if task.request_id == request_id)
    
    @staticmethod
    def filter_tasks_by_status(
        task_queue: List[Any],
        status: str
    ) -> List[Any]:
        """
        Filter tasks in queue by status.
        
        Args:
            task_queue: List of tasks
            status: Status to filter by
            
        Returns:
            List of tasks with the specified status
        """
        return [task for task in task_queue if hasattr(task, 'status') and task.status == status]
    
    @staticmethod
    def sort_tasks_by_priority(
        task_queue: List[Any],
        reverse: bool = False
    ) -> List[Any]:
        """
        Sort tasks by priority.
        
        Args:
            task_queue: List of tasks
            reverse: Whether to sort in reverse order (highest priority first)
            
        Returns:
            Sorted list of tasks
        """
        return sorted(
            task_queue,
            key=lambda x: getattr(x, 'priority', 0),
            reverse=reverse
        )






