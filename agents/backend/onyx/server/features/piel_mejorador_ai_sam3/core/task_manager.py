"""
Task Manager for Piel Mejorador AI SAM3
========================================
"""

import asyncio
import logging
import uuid
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict, field
from enum import Enum

from .helpers import ensure_directory_exists, load_json_file, save_json_file

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Task status enumeration."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Task:
    """Task data structure."""
    id: str
    service_type: str
    parameters: Dict[str, Any]
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
            service_type=data["service_type"],
            parameters=data["parameters"],
            priority=data.get("priority", 0),
            status=TaskStatus(data["status"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            started_at=datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None,
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
            result=data.get("result"),
            error=data.get("error"),
            metadata=data.get("metadata", {}),
        )


class FileTaskRepository:
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


class TaskManager:
    """Manages tasks for the skin enhancement agent."""
    
    def __init__(self, repository: Optional[FileTaskRepository] = None, storage_dir: str = "task_storage"):
        self.repository = repository or FileTaskRepository(storage_dir)
        self._lock = asyncio.Lock()
        self._tasks: Dict[str, Task] = {}
        self._pending_queue: List[str] = []
    
    async def initialize(self):
        """Initialize by loading tasks from repository."""
        tasks = await self.repository.get_all()
        for task in tasks:
            self._tasks[task.id] = task
            if task.status == TaskStatus.PENDING:
                self._pending_queue.append(task.id)
        logger.info(f"Loaded {len(tasks)} tasks from repository")
    
    async def create_task(
        self,
        service_type: str,
        parameters: Dict[str, Any],
        priority: int = 0
    ) -> str:
        """Create a new task."""
        task_id = str(uuid.uuid4())
        task = Task(
            id=task_id,
            service_type=service_type,
            parameters=parameters,
            priority=priority,
            status=TaskStatus.PENDING,
            created_at=datetime.now()
        )
        
        async with self._lock:
            self._tasks[task_id] = task
            self._pending_queue.append(task_id)
            await self.repository.save(task)
        
        logger.info(f"Created task {task_id}: {service_type}")
        return task_id
    
    async def get_pending_tasks(self, limit: int = 10) -> List[Task]:
        """Get pending tasks sorted by priority."""
        async with self._lock:
            pending = [
                self._tasks[task_id]
                for task_id in self._pending_queue
                if task_id in self._tasks and self._tasks[task_id].status == TaskStatus.PENDING
            ]
            pending.sort(key=lambda t: t.priority, reverse=True)
            return pending[:limit]
    
    async def update_task_status(self, task_id: str, status: str):
        """Update task status."""
        async with self._lock:
            if task_id not in self._tasks:
                return
            
            task = self._tasks[task_id]
            task.status = TaskStatus(status)
            
            if status == "processing":
                task.started_at = datetime.now()
                if task_id in self._pending_queue:
                    self._pending_queue.remove(task_id)
            elif status == "completed":
                task.completed_at = datetime.now()
            elif status == "failed":
                task.completed_at = datetime.now()
            
            await self.repository.save(task)
    
    async def complete_task(self, task_id: str, result: Dict[str, Any]):
        """Mark task as completed with result."""
        async with self._lock:
            if task_id not in self._tasks:
                return
            
            task = self._tasks[task_id]
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()
            task.result = result
            await self.repository.save(task)
    
    async def fail_task(self, task_id: str, error: str):
        """Mark task as failed with error."""
        async with self._lock:
            if task_id not in self._tasks:
                return
            
            task = self._tasks[task_id]
            task.status = TaskStatus.FAILED
            task.completed_at = datetime.now()
            task.error = error
            await self.repository.save(task)
    
    async def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get task status."""
        async with self._lock:
            if task_id not in self._tasks:
                return {"status": "not_found"}
            
            task = self._tasks[task_id]
            return {
                "id": task.id,
                "status": task.status.value,
                "service_type": task.service_type,
                "created_at": task.created_at.isoformat(),
                "started_at": task.started_at.isoformat() if task.started_at else None,
                "completed_at": task.completed_at.isoformat() if task.completed_at else None,
                "error": task.error,
            }
    
    async def get_task_result(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task result if completed."""
        async with self._lock:
            if task_id not in self._tasks:
                return None
            
            task = self._tasks[task_id]
            if task.status != TaskStatus.COMPLETED:
                return None
            
            return task.result




