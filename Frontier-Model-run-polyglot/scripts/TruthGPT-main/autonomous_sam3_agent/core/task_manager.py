"""
Task Manager
============

Manages tasks for the autonomous agent with priority queue and status tracking.

Refactored with:
- TaskRepository for storage abstraction
- Hook system for extensibility
- Enhanced state management
- Event-driven architecture
"""

import asyncio
import logging
import json
from typing import Dict, Any, Optional, List, Callable, Awaitable
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict, field
from enum import Enum
from abc import ABC, abstractmethod
import uuid

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
    CREATED = "created"
    STARTED = "started"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    STATUS_CHANGED = "status_changed"


@dataclass
class Task:
    """Task data structure."""
    id: str
    image_path: str
    text_prompt: str
    priority: int
    status: TaskStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary."""
        data = asdict(self)
        data["status"] = self.status.value
        data["created_at"] = self.created_at.isoformat()
        if self.started_at:
            data["started_at"] = self.started_at.isoformat()
        if self.completed_at:
            data["completed_at"] = self.completed_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Task":
        """Create task from dictionary."""
        return cls(
            id=data["id"],
            image_path=data["image_path"],
            text_prompt=data["text_prompt"],
            priority=data["priority"],
            status=TaskStatus(data["status"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            started_at=datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None,
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
            result=data.get("result"),
            error=data.get("error"),
            metadata=data.get("metadata", {}),
        )
    
    @property
    def duration(self) -> Optional[float]:
        """Get task duration in seconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None


class TaskRepository(ABC):
    """Abstract base class for task storage."""
    
    @abstractmethod
    async def save(self, task: Task) -> None:
        """Save task to storage."""
        pass
    
    @abstractmethod
    async def load(self, task_id: str) -> Optional[Task]:
        """Load task from storage."""
        pass
    
    @abstractmethod
    async def delete(self, task_id: str) -> None:
        """Delete task from storage."""
        pass
    
    @abstractmethod
    async def load_all(self) -> List[Task]:
        """Load all tasks from storage."""
        pass


class FileTaskRepository(TaskRepository):
    """File-based task repository."""
    
    def __init__(self, storage_dir: Path):
        self.storage_dir = storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)
    
    async def save(self, task: Task) -> None:
        """Save task to JSON file."""
        task_file = self.storage_dir / f"{task.id}.json"
        with open(task_file, "w", encoding="utf-8") as f:
            json.dump(task.to_dict(), f, indent=2)
    
    async def load(self, task_id: str) -> Optional[Task]:
        """Load task from JSON file."""
        task_file = self.storage_dir / f"{task_id}.json"
        if not task_file.exists():
            return None
        
        with open(task_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return Task.from_dict(data)
    
    async def delete(self, task_id: str) -> None:
        """Delete task JSON file."""
        task_file = self.storage_dir / f"{task_id}.json"
        if task_file.exists():
            task_file.unlink()
    
    async def load_all(self) -> List[Task]:
        """Load all tasks from directory."""
        tasks = []
        for task_file in self.storage_dir.glob("*.json"):
            try:
                with open(task_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                tasks.append(Task.from_dict(data))
            except Exception as e:
                logger.error(f"Error loading task from {task_file}: {e}")
        return tasks


class EventRegistry:
    """Registry for event handlers."""
    
    def __init__(self):
        self._handlers: Dict[TaskEvent, List[Callable]] = {
            event: [] for event in TaskEvent
        }
    
    def register(self, event: TaskEvent, handler: Callable):
        """Register handler for event."""
        self._handlers[event].append(handler)
    
    def unregister(self, event: TaskEvent, handler: Callable):
        """Unregister handler from event."""
        if handler in self._handlers[event]:
            self._handlers[event].remove(handler)
    
    async def emit(self, event: TaskEvent, task: Task, **kwargs):
        """Emit event to all handlers."""
        for handler in self._handlers[event]:
            try:
                result = handler(task, **kwargs)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error(f"Event handler error: {e}")


class TaskManager:
    """
    Manages tasks for the autonomous agent.
    
    Features:
    - Priority queue
    - Status tracking
    - Result storage
    - Task persistence
    - Event-driven architecture
    - Repository pattern for storage
    
    Refactored with:
    - Repository pattern for storage abstraction
    - Event registry for hooks
    - Factory method for task creation
    """
    
    def __init__(
        self,
        storage_dir: str = "task_storage",
        repository: Optional[TaskRepository] = None,
    ):
        """
        Initialize task manager.
        
        Args:
            storage_dir: Directory for task storage
            repository: Custom task repository (optional)
        """
        self.storage_dir = Path(storage_dir)
        
        # Use provided repository or create default file-based one
        self.repository = repository or FileTaskRepository(self.storage_dir)
        
        # In-memory task cache
        self._tasks: Dict[str, Task] = {}
        self._lock = asyncio.Lock()
        self._pending_queue: List[str] = []
        
        # Event registry for hooks
        self._events = EventRegistry()
        
        # Statistics
        self._stats = {
            "total_created": 0,
            "total_completed": 0,
            "total_failed": 0,
            "total_cancelled": 0,
        }
        
        logger.info(f"Initialized TaskManager with storage: {self.storage_dir}")
    
    # === Event Hook Registration ===
    
    def on_task_created(self, handler: Callable):
        """Register handler for task created event."""
        self._events.register(TaskEvent.CREATED, handler)
    
    def on_task_started(self, handler: Callable):
        """Register handler for task started event."""
        self._events.register(TaskEvent.STARTED, handler)
    
    def on_task_completed(self, handler: Callable):
        """Register handler for task completed event."""
        self._events.register(TaskEvent.COMPLETED, handler)
    
    def on_task_failed(self, handler: Callable):
        """Register handler for task failed event."""
        self._events.register(TaskEvent.FAILED, handler)
    
    def on_status_changed(self, handler: Callable):
        """Register handler for any status change."""
        self._events.register(TaskEvent.STATUS_CHANGED, handler)
    
    # === Task Factory Method ===
    
    def _create_task_instance(
        self,
        image_path: str,
        text_prompt: str,
        priority: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Task:
        """Factory method to create task instance."""
        return Task(
            id=str(uuid.uuid4()),
            image_path=image_path,
            text_prompt=text_prompt,
            priority=priority,
            status=TaskStatus.PENDING,
            created_at=datetime.now(),
            metadata=metadata or {},
        )
    
    # === Core Operations ===
    
    async def create_task(
        self,
        image_path: str,
        text_prompt: str,
        priority: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Create a new task.
        
        Args:
            image_path: Path to input image
            text_prompt: Text prompt for segmentation
            priority: Task priority (higher = more important)
            metadata: Optional task metadata
            
        Returns:
            Task ID
        """
        task = self._create_task_instance(
            image_path=image_path,
            text_prompt=text_prompt,
            priority=priority,
            metadata=metadata,
        )
        
        async with self._lock:
            self._tasks[task.id] = task
            self._pending_queue.append(task.id)
            self._sort_pending_queue()
            self._stats["total_created"] += 1
        
        # Persist task
        await self.repository.save(task)
        
        # Emit event
        await self._events.emit(TaskEvent.CREATED, task)
        
        logger.info(f"Created task {task.id} with priority {priority}")
        return task.id
    
    def _sort_pending_queue(self):
        """Sort pending queue by priority (descending)."""
        self._pending_queue.sort(
            key=lambda tid: self._tasks[tid].priority,
            reverse=True
        )
    
    async def get_pending_tasks(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get pending tasks sorted by priority."""
        async with self._lock:
            pending = [
                self._tasks[tid].to_dict()
                for tid in self._pending_queue
                if tid in self._tasks and self._tasks[tid].status == TaskStatus.PENDING
            ][:limit]
        return pending
    
    async def get_next_task(self) -> Optional[Task]:
        """Get the next pending task (highest priority)."""
        async with self._lock:
            for tid in self._pending_queue:
                if tid in self._tasks and self._tasks[tid].status == TaskStatus.PENDING:
                    return self._tasks[tid]
        return None
    
    async def update_task_status(self, task_id: str, status: str):
        """Update task status."""
        async with self._lock:
            if task_id not in self._tasks:
                raise ValueError(f"Task {task_id} not found")
            
            task = self._tasks[task_id]
            old_status = task.status
            task.status = TaskStatus(status)
            
            if status == "processing":
                task.started_at = datetime.now()
                if task_id in self._pending_queue:
                    self._pending_queue.remove(task_id)
            elif status in ["completed", "failed", "cancelled"]:
                task.completed_at = datetime.now()
        
        await self.repository.save(task)
        
        # Emit status changed event
        await self._events.emit(
            TaskEvent.STATUS_CHANGED, 
            task, 
            old_status=old_status.value,
            new_status=status
        )
        
        # Emit specific event
        if status == "processing":
            await self._events.emit(TaskEvent.STARTED, task)
        
        logger.debug(f"Updated task {task_id} status to {status}")
    
    async def complete_task(self, task_id: str, result: Dict[str, Any]):
        """Mark task as completed with result."""
        async with self._lock:
            if task_id not in self._tasks:
                raise ValueError(f"Task {task_id} not found")
            
            task = self._tasks[task_id]
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()
            task.result = result
            self._stats["total_completed"] += 1
        
        await self.repository.save(task)
        await self._events.emit(TaskEvent.COMPLETED, task, result=result)
        
        logger.info(f"Completed task {task_id}")
    
    async def fail_task(self, task_id: str, error: str):
        """Mark task as failed with error."""
        async with self._lock:
            if task_id not in self._tasks:
                raise ValueError(f"Task {task_id} not found")
            
            task = self._tasks[task_id]
            task.status = TaskStatus.FAILED
            task.completed_at = datetime.now()
            task.error = error
            self._stats["total_failed"] += 1
        
        await self.repository.save(task)
        await self._events.emit(TaskEvent.FAILED, task, error=error)
        
        logger.warning(f"Failed task {task_id}: {error}")
    
    async def cancel_task(self, task_id: str):
        """Cancel a pending task."""
        async with self._lock:
            if task_id not in self._tasks:
                raise ValueError(f"Task {task_id} not found")
            
            task = self._tasks[task_id]
            if task.status != TaskStatus.PENDING:
                raise ValueError(f"Cannot cancel task {task_id} with status {task.status}")
            
            task.status = TaskStatus.CANCELLED
            task.completed_at = datetime.now()
            self._stats["total_cancelled"] += 1
            
            if task_id in self._pending_queue:
                self._pending_queue.remove(task_id)
        
        await self.repository.save(task)
        await self._events.emit(TaskEvent.CANCELLED, task)
        
        logger.info(f"Cancelled task {task_id}")
    
    async def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get task status."""
        async with self._lock:
            if task_id not in self._tasks:
                raise ValueError(f"Task {task_id} not found")
            return self._tasks[task_id].to_dict()
    
    async def get_task_result(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task result if completed."""
        async with self._lock:
            if task_id not in self._tasks:
                raise ValueError(f"Task {task_id} not found")
            
            task = self._tasks[task_id]
            if task.status == TaskStatus.COMPLETED:
                return task.result
            return None
    
    async def load_tasks(self):
        """Load tasks from storage."""
        tasks = await self.repository.load_all()
        
        async with self._lock:
            for task in tasks:
                self._tasks[task.id] = task
                if task.status == TaskStatus.PENDING:
                    self._pending_queue.append(task.id)
            
            self._sort_pending_queue()
        
        logger.info(f"Loaded {len(tasks)} tasks from storage")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get task manager statistics."""
        return {
            **self._stats,
            "pending_count": len(self._pending_queue),
            "total_in_memory": len(self._tasks),
        }
