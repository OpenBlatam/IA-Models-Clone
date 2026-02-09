"""
Task Manager
============

Task management for workflows.
"""

import logging
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Task status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"


@dataclass
class Task:
    """Task definition."""
    id: str
    name: str
    handler: Callable
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Any = None
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


class TaskManager:
    """Task manager."""
    
    def __init__(self):
        self._tasks: Dict[str, Task] = {}
        self._task_queue: List[str] = []
    
    def create_task(
        self,
        task_id: str,
        name: str,
        handler: Callable,
        max_retries: int = 3
    ) -> Task:
        """Create task."""
        task = Task(
            id=task_id,
            name=name,
            handler=handler,
            max_retries=max_retries
        )
        
        self._tasks[task_id] = task
        self._task_queue.append(task_id)
        logger.info(f"Created task: {task_id}")
        return task
    
    async def execute_task(self, task_id: str, *args, **kwargs) -> Any:
        """Execute task."""
        if task_id not in self._tasks:
            raise ValueError(f"Task {task_id} not found")
        
        task = self._tasks[task_id]
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.now()
        
        try:
            if asyncio.iscoroutinefunction(task.handler):
                result = await task.handler(*args, **kwargs)
            else:
                result = await asyncio.to_thread(task.handler, *args, **kwargs)
            
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()
            task.result = result
            
            return result
        
        except Exception as e:
            task.error = str(e)
            task.retry_count += 1
            
            if task.retry_count <= task.max_retries:
                task.status = TaskStatus.RETRYING
                logger.warning(f"Task {task_id} failed, retrying: {e}")
                # Retry logic would go here
            else:
                task.status = TaskStatus.FAILED
                task.completed_at = datetime.now()
                logger.error(f"Task {task_id} failed after {task.retry_count} attempts: {e}")
            
            raise
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """Get task by ID."""
        return self._tasks.get(task_id)
    
    def get_task_stats(self) -> Dict[str, Any]:
        """Get task statistics."""
        return {
            "total_tasks": len(self._tasks),
            "by_status": {
                status.value: sum(1 for t in self._tasks.values() if t.status == status)
                for status in TaskStatus
            }
        }


# Import asyncio
import asyncio















