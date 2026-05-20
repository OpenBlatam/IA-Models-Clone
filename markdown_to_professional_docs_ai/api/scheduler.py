"""Task scheduler endpoints"""
from fastapi import APIRouter, HTTPException, Form
from typing import Optional
from utils.scheduler import get_scheduler
import json

router = APIRouter(prefix="/scheduler", tags=["Scheduler"])


@router.get("/tasks")
async def list_scheduled_tasks():
    """List all scheduled tasks"""
    scheduler = get_scheduler()
    tasks = scheduler.list_tasks()
    
    return {
        "tasks": [
            {
                "id": task["id"],
                "name": task["name"],
                "schedule": task["schedule"],
                "status": task["status"],
                "next_run": task.get("next_run")
            }
            for task in tasks
        ],
        "total": len(tasks)
    }


@router.get("/task/{task_id}")
async def get_task_status(task_id: str):
    """Get task status"""
    scheduler = get_scheduler()
    task = scheduler.get_task(task_id)
    
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return task


@router.post("/task/{task_id}/cancel")
async def cancel_task(task_id: str):
    """Cancel scheduled task"""
    scheduler = get_scheduler()
    success = scheduler.cancel_task(task_id)
    
    if success:
        return {
            "status": "success",
            "task_id": task_id
        }
    else:
        raise HTTPException(status_code=400, detail="Failed to cancel task")

