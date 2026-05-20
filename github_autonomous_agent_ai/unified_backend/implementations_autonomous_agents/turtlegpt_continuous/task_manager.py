"""
Task Manager

Manages task queue, processing, and lifecycle.
"""

from typing import Dict, List, Optional
import logging
import uuid
from datetime import datetime

from .models import AgentTask, TaskStatus

logger = logging.getLogger(__name__)


class TaskManager:
    """Manages agent tasks."""
    
    def __init__(self, max_concurrent_tasks: int = 3):
        """
        Initialize task manager.
        
        Args:
            max_concurrent_tasks: Maximum concurrent tasks
        """
        self.max_concurrent_tasks = max_concurrent_tasks
        self.task_queue: List[AgentTask] = []
        self.active_tasks: Dict[str, AgentTask] = {}
        self.completed_tasks: List[AgentTask] = []
    
    def submit_task(
        self,
        description: str,
        priority: int = 5,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Submit a task for processing.
        
        Args:
            description: Task description
            priority: Priority (1-10, higher = more priority)
            metadata: Additional metadata
            
        Returns:
            Task ID
        """
        task_id = str(uuid.uuid4())
        
        task = AgentTask(
            task_id=task_id,
            description=description,
            priority=priority,
            metadata=metadata or {}
        )
        
        self.task_queue.append(task)
        logger.info(f"Task submitted: {task_id} - {description}")
        
        return task_id
    
    def get_next_tasks(self) -> List[AgentTask]:
        """
        Get next tasks to process (up to max concurrent).
        
        Returns:
            List of tasks to process
        """
        # Sort by priority
        self.task_queue.sort(key=lambda t: t.priority, reverse=True)
        
        # Get tasks up to limit
        available_slots = self.max_concurrent_tasks - len(self.active_tasks)
        tasks_to_process = []
        
        while available_slots > 0 and len(self.task_queue) > 0:
            task = self.task_queue.pop(0)
            task.status = TaskStatus.PROCESSING
            self.active_tasks[task.task_id] = task
            tasks_to_process.append(task)
            available_slots -= 1
        
        return tasks_to_process
    
    def mark_task_completed(self, task_id: str, result: Optional[Dict] = None) -> None:
        """
        Mark task as completed.
        
        Args:
            task_id: Task ID
            result: Task result
        """
        if task_id in self.active_tasks:
            task = self.active_tasks.pop(task_id)
            task.status = TaskStatus.COMPLETED
            task.result = result
            self.completed_tasks.append(task)
            logger.info(f"Task completed: {task_id}")
    
    def mark_task_failed(self, task_id: str, error: str) -> None:
        """
        Mark task as failed.
        
        Args:
            task_id: Task ID
            error: Error message
        """
        if task_id in self.active_tasks:
            task = self.active_tasks.pop(task_id)
            task.status = TaskStatus.FAILED
            task.error = error
            self.completed_tasks.append(task)
            logger.error(f"Task failed: {task_id} - {error}")
    
    def get_completed_tasks_for_cleanup(self) -> List[AgentTask]:
        """
        Get completed tasks that need cleanup.
        
        Returns:
            List of completed tasks
        """
        completed = [
            task for task_id, task in self.active_tasks.items()
            if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]
        ]
        return completed
    
    def cleanup_completed_tasks(self, max_keep: int = 100) -> None:
        """
        Cleanup old completed tasks.
        
        Args:
            max_keep: Maximum tasks to keep
        """
        if len(self.completed_tasks) > max_keep:
            self.completed_tasks = self.completed_tasks[-max_keep:]
    
    def should_enter_idle_mode(self, enable_idle_mode: bool) -> bool:
        """
        Determine if should enter idle mode.
        
        Args:
            enable_idle_mode: Whether idle mode is enabled
            
        Returns:
            True if should enter idle mode
        """
        if not enable_idle_mode:
            return False
        
        return (
            len(self.task_queue) == 0 and
            len(self.active_tasks) == 0
        )
    
    def get_recent_tasks(self, count: int = 5) -> List[AgentTask]:
        """
        Get recent tasks (from queue and active).
        
        Args:
            count: Number of recent tasks to return
            
        Returns:
            List of recent tasks
        """
        recent = []
        # Add from queue (most recent first)
        recent.extend(self.task_queue[-count:])
        # Add from active tasks
        recent.extend(list(self.active_tasks.values())[-count:])
        # Return up to count
        return recent[:count]
    
    def get_stats(self) -> Dict:
        """Get task statistics."""
        return {
            "queue_size": len(self.task_queue),
            "active_tasks": len(self.active_tasks),
            "completed_tasks": len(self.completed_tasks)
        }



