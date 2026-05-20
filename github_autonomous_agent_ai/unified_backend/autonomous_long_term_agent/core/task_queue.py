"""
Task Queue for Autonomous Agent
Implements task management for long-horizon reasoning
"""

import asyncio
import logging
from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ════════════════════════════════════════════════════════════════════════════

DEFAULT_MAX_RETRIES = 3


class TaskStatus(str, Enum):
    """Task status enumeration"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Task:
    """Task representation"""
    id: str
    instruction: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = DEFAULT_MAX_RETRIES


class TaskQueue:
    """Task queue for managing agent tasks"""
    
    def __init__(self):
        self._queue: asyncio.Queue = asyncio.Queue()
        self._tasks: Dict[str, Task] = {}
        self._lock = asyncio.Lock()
        self._task_counter = 0
    
    async def add_task(
        self,
        instruction: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Add a new task to the queue"""
        async with self._lock:
            self._task_counter += 1
            task_id = f"task_{self._task_counter}_{int(datetime.utcnow().timestamp())}"
            
            task = Task(
                id=task_id,
                instruction=instruction,
                metadata=metadata or {}
            )
            
            self._tasks[task_id] = task
            await self._queue.put(task_id)
            
            logger.info(f"Added task {task_id}: {instruction[:50]}...")
            return task_id
    
    async def get_next_task(self) -> Optional[Task]:
        """Get next task from queue (non-blocking)"""
        try:
            task_id = await asyncio.wait_for(self._queue.get(), timeout=0.1)
            task = self._tasks.get(task_id)
            if task and task.status == TaskStatus.PENDING:
                task.status = TaskStatus.RUNNING
                task.started_at = datetime.utcnow()
                return task
        except asyncio.TimeoutError:
            return None
        return None
    
    async def complete_task(
        self,
        task_id: str,
        result: Optional[Dict[str, Any]] = None
    ) -> None:
        """Mark task as completed"""
        async with self._lock:
            task = self._tasks.get(task_id)
            if task:
                task.status = TaskStatus.COMPLETED
                task.completed_at = datetime.utcnow()
                task.result = result
                logger.info(f"Task {task_id} completed")
    
    async def fail_task(
        self,
        task_id: str,
        error: str
    ) -> None:
        """Mark task as failed"""
        async with self._lock:
            task = self._tasks.get(task_id)
            if task:
                task.retry_count += 1
                if task.retry_count >= task.max_retries:
                    task.status = TaskStatus.FAILED
                    task.error = error
                    logger.error(f"Task {task_id} failed after {task.retry_count} retries: {error}")
                else:
                    task.status = TaskStatus.PENDING
                    await self._queue.put(task_id)
                    logger.warning(f"Task {task_id} retry {task.retry_count}/{task.max_retries}")
    
    async def cancel_task(self, task_id: str) -> None:
        """Cancel a task"""
        async with self._lock:
            task = self._tasks.get(task_id)
            if task:
                task.status = TaskStatus.CANCELLED
                logger.info(f"Task {task_id} cancelled")
    
    async def get_task(self, task_id: str) -> Optional[Task]:
        """Get task by ID"""
        async with self._lock:
            return self._tasks.get(task_id)
    
    async def list_tasks(
        self,
        status: Optional[TaskStatus] = None,
        limit: int = 100
    ) -> List[Task]:
        """List tasks, optionally filtered by status"""
        async with self._lock:
            tasks = list(self._tasks.values())
            if status:
                tasks = [t for t in tasks if t.status == status]
            return tasks[:limit]
    
    async def get_queue_size(self) -> int:
        """Get current queue size"""
        return self._queue.qsize()
    
    async def get_recent_tasks(self, limit: int = 10) -> List[Task]:
        """Get recent tasks sorted by creation time (most recent first)"""
        async with self._lock:
            tasks = list(self._tasks.values())
            # Sort by created_at descending (most recent first)
            tasks.sort(key=lambda t: t.created_at, reverse=True)
            return tasks[:limit]
    
    async def clear_completed(self) -> int:
        """Clear completed tasks older than 1 hour"""
        cutoff = datetime.utcnow().timestamp() - 3600
        removed = 0
        
        async with self._lock:
            to_remove = []
            for task_id, task in self._tasks.items():
                if (task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED] and
                    task.completed_at and
                    task.completed_at.timestamp() < cutoff):
                    to_remove.append(task_id)
            
            for task_id in to_remove:
                del self._tasks[task_id]
                removed += 1
        
        if removed > 0:
            logger.info(f"Cleared {removed} old completed tasks")
        
        return removed




