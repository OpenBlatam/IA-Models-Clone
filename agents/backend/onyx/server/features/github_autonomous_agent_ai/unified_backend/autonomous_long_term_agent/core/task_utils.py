"""
Task Utilities
Shared utilities for task operations
"""

from typing import Dict, Any, List
from .task_queue import Task, TaskStatus


def task_to_dict(task: Task) -> Dict[str, Any]:
    """
    Convert Task to dictionary format.
    Single source of truth for task serialization.
    
    Args:
        task: Task instance
        
    Returns:
        Dictionary representation of task
    """
    return {
        "id": task.id,
        "instruction": task.instruction,
        "status": task.status.value if hasattr(task.status, 'value') else str(task.status),
        "started_at": task.started_at.isoformat() if task.started_at else None,
        "completed_at": task.completed_at.isoformat() if task.completed_at else None,
        "outcome": _determine_task_outcome(task)
    }


def tasks_to_dict_list(tasks: List[Task]) -> List[Dict[str, Any]]:
    """
    Convert list of tasks to list of dictionaries.
    
    Args:
        tasks: List of Task instances
        
    Returns:
        List of task dictionaries
    """
    return [task_to_dict(task) for task in tasks]


def _determine_task_outcome(task: Task) -> str:
    """Determine task outcome based on status"""
    if task.status == TaskStatus.COMPLETED:
        return "success"
    elif task.status == TaskStatus.FAILED:
        return "failure"
    else:
        return "pending"

