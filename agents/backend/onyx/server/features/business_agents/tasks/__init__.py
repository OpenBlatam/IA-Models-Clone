"""
Async task system for background job processing.
Supports both Celery and RQ backends via settings.
"""
from enum import Enum
from typing import Optional, Dict, Any, Callable
from uuid import uuid4
from datetime import datetime
import json


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILURE = "failure"


class TaskResult:
    def __init__(self, task_id: str, status: TaskStatus, result: Optional[Any] = None, error: Optional[str] = None):
        self.task_id = task_id
        self.status = status
        self.result = result
        self.error = error
        self.created_at = datetime.now()
        self.updated_at = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


# In-memory task store (replace with Redis/DB in production)
_task_store: Dict[str, TaskResult] = {}


def create_task(func: Callable, *args, **kwargs) -> str:
    """Create a new async task."""
    task_id = str(uuid4())
    task_result = TaskResult(task_id, TaskStatus.PENDING)
    _task_store[task_id] = task_result
    
    # In a real implementation, this would enqueue to Celery/RQ
    # For now, we'll use a simple async executor
    import asyncio
    asyncio.create_task(_execute_task(task_id, func, *args, **kwargs))
    
    return task_id


async def _execute_task(task_id: str, func: Callable, *args, **kwargs):
    """Execute task and update status."""
    import asyncio
    task = _task_store.get(task_id)
    if not task:
        return
    
    task.status = TaskStatus.RUNNING
    task.updated_at = datetime.now()
    
    # Publish event (if event bus available)
    try:
        from ..event_system import event_bus, EventType
        await event_bus.publish(EventType.TASK_CREATED, {"task_id": task_id})
    except Exception:
        pass
    
    try:
        if asyncio.iscoroutinefunction(func):
            result = await func(*args, **kwargs)
        else:
            result = func(*args, **kwargs)
        
        task.status = TaskStatus.SUCCESS
        task.result = result
        
        # Publish success event
        try:
            from ..event_system import event_bus, EventType
            await event_bus.publish(EventType.TASK_COMPLETED, {"task_id": task_id, "result": result})
        except Exception:
            pass
    except Exception as e:
        task.status = TaskStatus.FAILURE
        task.error = str(e)
        
        # Publish failure event
        try:
            from ..event_system import event_bus, EventType
            await event_bus.publish(EventType.TASK_FAILED, {"task_id": task_id, "error": str(e)})
        except Exception:
            pass
    finally:
        task.updated_at = datetime.now()


def get_task(task_id: str) -> Optional[TaskResult]:
    """Get task status and result."""
    return _task_store.get(task_id)


def list_tasks(status: Optional[TaskStatus] = None, limit: int = 100) -> list[TaskResult]:
    """List tasks, optionally filtered by status."""
    tasks = list(_task_store.values())
    if status:
        tasks = [t for t in tasks if t.status == status]
    tasks.sort(key=lambda x: x.created_at, reverse=True)
    return tasks[:limit]
