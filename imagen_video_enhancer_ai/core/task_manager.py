"""
Task Manager for Imagen Video Enhancer AI
==========================================
"""

import asyncio
import logging
import uuid
from typing import Dict, Any, Optional, List, Callable, Awaitable
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict, field
from enum import Enum
from abc import ABC, abstractmethod

from .helpers import ensure_directory_exists, load_json_file, save_json_file
from .base_models import BaseModel, TimestampedModel, IdentifiedModel, StatusModel

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Task status enumeration."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskEvent(Enum):
    """Task lifecycle events."""
    CREATED = "task_created"
    STARTED = "task_started"
    COMPLETED = "task_completed"
    FAILED = "task_failed"
    CANCELLED = "task_cancelled"
    STATUS_CHANGED = "task_status_changed"


@dataclass
class Task(BaseModel, TimestampedModel, IdentifiedModel):
    """Task data structure."""
    service_type: str
    parameters: Dict[str, Any]
    priority: int
    status: TaskStatus
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self, exclude_none: bool = False) -> Dict[str, Any]:
        """Convert task to dictionary."""
        data = super().to_dict(exclude_none=exclude_none)
        # Ensure status is string value
        if isinstance(data.get("status"), TaskStatus):
            data["status"] = data["status"].value
        return data
    
    def is_pending(self) -> bool:
        """Check if status is pending."""
        return self.status == TaskStatus.PENDING
    
    def is_completed(self) -> bool:
        """Check if status is completed."""
        return self.status == TaskStatus.COMPLETED
    
    def is_failed(self) -> bool:
        """Check if status is failed."""
        return self.status == TaskStatus.FAILED
    
    def is_processing(self) -> bool:
        """Check if status is processing."""
        return self.status == TaskStatus.PROCESSING
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Task":
        """Create task from dictionary."""
        # Convert status string to enum if needed
        status = data.get("status")
        if isinstance(status, str):
            status = TaskStatus(status)
        
        return cls(
            id=data["id"],
            service_type=data["service_type"],
            parameters=data["parameters"],
            priority=data.get("priority", 0),
            status=status,
            created_at=datetime.fromisoformat(data["created_at"]) if isinstance(data.get("created_at"), str) else data.get("created_at", datetime.now()),
            started_at=datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None,
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
            result=data.get("result"),
            error=data.get("error"),
            metadata=data.get("metadata", {}),
        )


class TaskRepository(ABC):
    """Abstract base class for task storage."""
    
    @abstractmethod
    async def save(self, task: Task) -> None:
        """Save task."""
        pass
    
    @abstractmethod
    async def get(self, task_id: str) -> Optional[Task]:
        """Get task by ID."""
        pass
    
    @abstractmethod
    async def get_all(self) -> List[Task]:
        """Get all tasks."""
        pass
    
    @abstractmethod
    async def delete(self, task_id: str) -> None:
        """Delete task."""
        pass


class FileTaskRepository(TaskRepository):
    """File-based task repository."""
    
    def __init__(self, storage_dir: str):
        self.storage_dir = ensure_directory_exists(storage_dir)
    
    async def save(self, task: Task) -> None:
        """Save task to file."""
        file_path = self.storage_dir / f"{task.id}.json"
        save_json_file(task.to_dict(), str(file_path))
    
    async def get(self, task_id: str) -> Optional[Task]:
        """Get task from file."""
        file_path = self.storage_dir / f"{task_id}.json"
        if not file_path.exists():
            return None
        
        try:
            data = load_json_file(str(file_path))
            return Task.from_dict(data)
        except Exception as e:
            logger.error(f"Error loading task {task_id}: {e}")
            return None
    
    async def get_all(self) -> List[Task]:
        """Get all tasks from files."""
        tasks = []
        for file_path in self.storage_dir.glob("*.json"):
            try:
                data = load_json_file(str(file_path))
                tasks.append(Task.from_dict(data))
            except Exception as e:
                logger.error(f"Error loading task from {file_path}: {e}")
        return tasks
    
    async def delete(self, task_id: str) -> None:
        """Delete task file."""
        file_path = self.storage_dir / f"{task_id}.json"
        if file_path.exists():
            file_path.unlink()


class EventRegistry:
    """Registry for task events."""
    
    def __init__(self):
        self._listeners: Dict[TaskEvent, List[Callable[[Task], Awaitable[None]]]] = {
            event: [] for event in TaskEvent
        }
    
    def subscribe(self, event: TaskEvent, callback: Callable[[Task], Awaitable[None]]):
        """Subscribe to an event."""
        self._listeners[event].append(callback)
    
    async def emit(self, event: TaskEvent, task: Task):
        """Emit an event."""
        for callback in self._listeners[event]:
            try:
                await callback(task)
            except Exception as e:
                logger.error(f"Error in event listener for {event}: {e}")


class TaskManager:
    """Manages tasks for the enhancer agent."""
    
    def __init__(self, repository: Optional[TaskRepository] = None, storage_dir: str = "task_storage"):
        self.repository = repository or FileTaskRepository(storage_dir)
        self.events = EventRegistry()
        self._lock = asyncio.Lock()
        
        # In-memory cache for active tasks and queue
        self._tasks: Dict[str, Task] = {}
        self._pending_queue: List[str] = []
    
    async def initialize(self):
        """Initialize by loading tasks from repository."""
        tasks = await self.repository.get_all()
        for task in tasks:
            self._tasks[task.id] = task
            if task.status == TaskStatus.PENDING:
                self._pending_queue.append(task.id)
        
        # Sort queue
        self._pending_queue.sort(
            key=lambda tid: self._tasks[tid].priority,
            reverse=True
        )
        logger.info(f"TaskManager initialized with {len(self._tasks)} tasks")
    
    async def create_task(
        self,
        service_type: str,
        parameters: Dict[str, Any],
        priority: int = 0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a new task."""
        task_id = str(uuid.uuid4())
        
        task = Task(
            id=task_id,
            service_type=service_type,
            parameters=parameters,
            priority=priority,
            status=TaskStatus.PENDING,
            created_at=datetime.now(),
            metadata=metadata or {},
        )
        
        async with self._lock:
            self._tasks[task_id] = task
            self._pending_queue.append(task_id)
            self._pending_queue.sort(
                key=lambda tid: self._tasks[tid].priority,
                reverse=True
            )
        
        await self.repository.save(task)
        await self.events.emit(TaskEvent.CREATED, task)
        
        logger.info(f"Created task {task_id} with priority {priority}")
        return task_id
    
    async def get_pending_tasks(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get pending tasks."""
        async with self._lock:
            pending = []
            for tid in self._pending_queue:
                if len(pending) >= limit:
                    break
                task = self._tasks.get(tid)
                if task and task.status == TaskStatus.PENDING:
                    pending.append(task.to_dict())
            return pending
    
    async def update_task_status(self, task_id: str, status: str):
        """Update task status."""
        async with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                raise ValueError(f"Task {task_id} not found")
            
            old_status = task.status
            new_status = TaskStatus(status)
            task.status = new_status
            
            if new_status == TaskStatus.PROCESSING:
                task.started_at = datetime.now()
                if task_id in self._pending_queue:
                    self._pending_queue.remove(task_id)
                await self.events.emit(TaskEvent.STARTED, task)
            
            elif new_status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                task.completed_at = datetime.now()
            
            await self.events.emit(TaskEvent.STATUS_CHANGED, task)
        
        await self.repository.save(task)
    
    async def complete_task(self, task_id: str, result: Dict[str, Any]):
        """Mark task as completed."""
        async with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                raise ValueError(f"Task {task_id} not found")
            
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()
            task.result = result
            
            await self.events.emit(TaskEvent.COMPLETED, task)
        
        await self.repository.save(task)
        logger.info(f"Completed task {task_id}")
    
    async def fail_task(self, task_id: str, error: str):
        """Mark task as failed."""
        async with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                raise ValueError(f"Task {task_id} not found")
            
            task.status = TaskStatus.FAILED
            task.completed_at = datetime.now()
            task.error = error
            
            await self.events.emit(TaskEvent.FAILED, task)
        
        await self.repository.save(task)
        logger.warning(f"Failed task {task_id}: {error}")
    
    async def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get task status."""
        task = self._tasks.get(task_id)
        if not task:
            task = await self.repository.get(task_id)
            if task:
                self._tasks[task_id] = task
        
        if not task:
            raise ValueError(f"Task {task_id} not found")
        
        return task.to_dict()
    
    async def get_task_result(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task result."""
        task = self._tasks.get(task_id)
        if not task:
            task = await self.repository.get(task_id)
        
        if task and task.status == TaskStatus.COMPLETED:
            return task.result
        return None

