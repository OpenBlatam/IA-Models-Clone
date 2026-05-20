"""
Task validation utilities for TaskManager.

Refactored to consolidate task validation patterns.
"""

from typing import Dict, Any
from .task_manager import Task, TaskManager


class TaskValidator:
    """
    Validates tasks and provides consistent error handling.
    
    Responsibilities:
    - Validate task existence
    - Provide consistent error messages
    
    Single Responsibility: Handle all task validation operations.
    """
    
    @staticmethod
    def validate_task_exists(task_manager: TaskManager, task_id: str) -> Task:
        """
        Validate that a task exists and return it.
        
        Args:
            task_manager: Task manager instance
            task_id: Task identifier
            
        Returns:
            Task instance
            
        Raises:
            ValueError: If task not found
        """
        if task_id not in task_manager._tasks:
            raise ValueError(f"Task {task_id} not found")
        return task_manager._tasks[task_id]

