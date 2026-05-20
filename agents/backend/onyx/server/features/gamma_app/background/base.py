"""
Background Tasks Base Classes and Interfaces
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Any, Optional, Callable
from datetime import datetime
from uuid import uuid4
from dataclasses import dataclass


class TaskStatus(str, Enum):
    """Task status"""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    CANCELLED = "cancelled"


@dataclass
class Task:
    """Task definition"""
    id: str
    task_type: str
    payload: Dict[str, Any]
    status: TaskStatus
    priority: int = 0
    max_retries: int = 3
    retry_count: int = 0
    created_at: datetime = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()


class TaskQueue:
    """Task queue definition"""
    
    def __init__(self, name: str, max_size: Optional[int] = None):
        self.name = name
        self.max_size = max_size
        self.tasks: list = []


class Worker:
    """Worker definition"""
    
    def __init__(self, worker_id: str, queue_name: str):
        self.worker_id = worker_id
        self.queue_name = queue_name
        self.is_running = False
        self.current_task: Optional[Task] = None


class BackgroundTaskBase(ABC):
    """Base interface for background tasks"""
    
    @abstractmethod
    async def enqueue_task(
        self,
        task_type: str,
        payload: Dict[str, Any],
        priority: int = 0
    ) -> str:
        """Enqueue a task"""
        pass
    
    @abstractmethod
    async def get_task_status(self, task_id: str) -> Optional[Task]:
        """Get task status"""
        pass
    
    @abstractmethod
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a task"""
        pass

