"""
Task Manager for Flux2 Clothing Changer
=======================================

Advanced task management and scheduling system.
"""

import time
import uuid
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import logging
from collections import deque

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Task status."""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


class TaskPriority(Enum):
    """Task priority."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4


@dataclass
class Task:
    """Task information."""
    task_id: str
    task_type: str
    status: TaskStatus = TaskStatus.PENDING
    priority: TaskPriority = TaskPriority.NORMAL
    data: Dict[str, Any] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    created_at: float = time.time()
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    retry_count: int = 0
    max_retries: int = 3
    
    def __post_init__(self):
        if self.data is None:
            self.data = {}


class TaskManager:
    """Advanced task management system."""
    
    def __init__(
        self,
        max_concurrent_tasks: int = 10,
        enable_priority_queue: bool = True,
    ):
        """
        Initialize task manager.
        
        Args:
            max_concurrent_tasks: Maximum concurrent tasks
            enable_priority_queue: Enable priority-based queueing
        """
        self.max_concurrent_tasks = max_concurrent_tasks
        self.enable_priority_queue = enable_priority_queue
        
        self.tasks: Dict[str, Task] = {}
        self.task_handlers: Dict[str, Callable] = {}
        self.running_tasks: set = set()
        self.task_queue: deque = deque()
        self.task_history: deque = deque(maxlen=10000)
        
        # Statistics
        self.stats = {
            "total_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "cancelled_tasks": 0,
        }
    
    def register_handler(
        self,
        task_type: str,
        handler: Callable[[Dict[str, Any]], Any],
    ) -> None:
        """
        Register task handler.
        
        Args:
            task_type: Task type
            handler: Handler function
        """
        self.task_handlers[task_type] = handler
        logger.info(f"Registered handler for task type: {task_type}")
    
    def create_task(
        self,
        task_type: str,
        data: Dict[str, Any],
        priority: TaskPriority = TaskPriority.NORMAL,
        max_retries: int = 3,
    ) -> Task:
        """
        Create task.
        
        Args:
            task_type: Task type
            data: Task data
            priority: Task priority
            max_retries: Maximum retry attempts
            
        Returns:
            Created task
        """
        task_id = str(uuid.uuid4())
        
        task = Task(
            task_id=task_id,
            task_type=task_type,
            priority=priority,
            data=data,
            max_retries=max_retries,
        )
        
        self.tasks[task_id] = task
        self.task_queue.append(task_id)
        self.stats["total_tasks"] += 1
        
        logger.info(f"Created task: {task_id} ({task_type})")
        return task
    
    def execute_task(self, task_id: str) -> bool:
        """
        Execute task.
        
        Args:
            task_id: Task identifier
            
        Returns:
            True if executed
        """
        if task_id not in self.tasks:
            return False
        
        task = self.tasks[task_id]
        
        if task.task_type not in self.task_handlers:
            task.status = TaskStatus.FAILED
            task.error = f"Handler not found for task type: {task.task_type}"
            self.stats["failed_tasks"] += 1
            return False
        
        task.status = TaskStatus.RUNNING
        task.started_at = time.time()
        self.running_tasks.add(task_id)
        
        try:
            handler = self.task_handlers[task.task_type]
            result = handler(task.data)
            
            task.status = TaskStatus.COMPLETED
            task.result = result
            task.completed_at = time.time()
            self.stats["completed_tasks"] += 1
            
            logger.info(f"Task {task_id} completed")
        except Exception as e:
            task.error = str(e)
            
            if task.retry_count < task.max_retries:
                task.status = TaskStatus.RETRYING
                task.retry_count += 1
                logger.warning(f"Task {task_id} failed, retrying ({task.retry_count}/{task.max_retries})")
                # Re-queue for retry
                self.task_queue.append(task_id)
            else:
                task.status = TaskStatus.FAILED
                self.stats["failed_tasks"] += 1
                logger.error(f"Task {task_id} failed after {task.max_retries} retries")
        finally:
            self.running_tasks.discard(task_id)
            self.task_history.append(task)
        
        return task.status == TaskStatus.COMPLETED
    
    def cancel_task(self, task_id: str) -> bool:
        """
        Cancel task.
        
        Args:
            task_id: Task identifier
            
        Returns:
            True if cancelled
        """
        if task_id not in self.tasks:
            return False
        
        task = self.tasks[task_id]
        
        if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
            return False
        
        task.status = TaskStatus.CANCELLED
        self.running_tasks.discard(task_id)
        self.stats["cancelled_tasks"] += 1
        
        logger.info(f"Task {task_id} cancelled")
        return True
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """Get task by ID."""
        return self.tasks.get(task_id)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get task manager statistics."""
        return {
            **self.stats,
            "pending_tasks": len([t for t in self.tasks.values() if t.status == TaskStatus.PENDING]),
            "running_tasks": len(self.running_tasks),
            "queued_tasks": len(self.task_queue),
        }


