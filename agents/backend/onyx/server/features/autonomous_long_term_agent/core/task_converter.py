"""
Task Converter
Converts Task objects to API response formats
Provides type-safe conversion between internal Task objects and external API schemas
"""

from typing import List, Optional
from ..core.task_queue import Task, TaskStatus
from ..api.v1.schemas.responses import TaskResponse


class TaskConverter:
    """
    Converts Task objects to response schemas
    
    Responsibilities:
    - Convert internal Task objects to TaskResponse schemas
    - Parse status strings to TaskStatus enums
    - Handle conversion errors gracefully
    """
    
    @staticmethod
    def to_response(task: Task) -> TaskResponse:
        """
        Convert Task to TaskResponse
        
        Args:
            task: Task instance
        
        Returns:
            TaskResponse schema
        """
        return TaskResponse(
            task_id=task.id,
            status=task.status.value,
            instruction=task.instruction,
            created_at=task.created_at.isoformat(),
            started_at=task.started_at.isoformat() if task.started_at else None,
            completed_at=task.completed_at.isoformat() if task.completed_at else None,
            result=task.result,
            error=task.error
        )
    
    @staticmethod
    def to_response_list(tasks: List[Task]) -> List[TaskResponse]:
        """
        Convert list of Tasks to list of TaskResponse
        
        Args:
            tasks: List of Task instances
        
        Returns:
            List of TaskResponse schemas
        """
        return [TaskConverter.to_response(task) for task in tasks]
    
    @staticmethod
    def parse_status(status_str: Optional[str]) -> Optional[TaskStatus]:
        """
        Parse status string to TaskStatus enum
        
        Args:
            status_str: Status string (e.g., "pending", "completed", "failed")
        
        Returns:
            TaskStatus enum or None if status_str is None/empty
        
        Raises:
            ValueError: If status string is invalid (not a valid TaskStatus value)
        """
        if not status_str:
            return None
        
        try:
            return TaskStatus(status_str)
        except ValueError:
            valid_values = [status.value for status in TaskStatus]
            raise ValueError(
                f"Invalid status: '{status_str}'. "
                f"Valid values: {valid_values}"
            )




