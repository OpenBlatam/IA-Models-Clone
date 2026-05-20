"""Task management API endpoints."""
from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Optional, Dict, Any
from pydantic import BaseModel
from . import create_task, get_task, list_tasks, TaskStatus

router = APIRouter(prefix="/tasks", tags=["Tasks"])


class TaskRequest(BaseModel):
    func_name: str
    args: list = []
    kwargs: Dict[str, Any] = {}


@router.post("/", response_model=Dict[str, str])
async def create_background_task(request: TaskRequest):
    """Create a new background task."""
    # Note: This is a simplified example. In production, you'd use Celery/RQ
    # and pass the actual function name or a task ID to the worker
    
    # For demo purposes, create a simple task
    async def demo_task():
        import asyncio
        await asyncio.sleep(2)  # Simulate work
        return {"completed": True, "func": request.func_name}
    
    task_id = create_task(demo_task)
    return {"task_id": task_id, "status": "pending"}


@router.get("/{task_id}", response_model=Dict[str, Any])
async def get_task_status(task_id: str):
    """Get task status and result."""
    task = get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task.to_dict()


@router.get("/", response_model=list[Dict[str, Any]])
async def list_all_tasks(
    status: Optional[TaskStatus] = None,
    limit: int = 100
):
    """List all tasks, optionally filtered by status."""
    tasks = list_tasks(status=status, limit=limit)
    return [task.to_dict() for task in tasks]


