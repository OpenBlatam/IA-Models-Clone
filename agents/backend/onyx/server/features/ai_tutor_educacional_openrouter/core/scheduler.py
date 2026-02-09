"""
Task scheduling system for background jobs.
"""

import asyncio
import logging
from typing import Callable, Dict, Any, Optional, List
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


class TaskPriority(Enum):
    """Task priority levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TaskScheduler:
    """
    Task scheduler for background jobs and periodic tasks.
    """
    
    def __init__(self):
        self.tasks: Dict[str, Dict[str, Any]] = {}
        self.running = False
        self.background_tasks: List[asyncio.Task] = []
    
    async def schedule_task(
        self,
        task_id: str,
        task_func: Callable,
        schedule: str,
        priority: TaskPriority = TaskPriority.MEDIUM,
        max_retries: int = 3
    ):
        """
        Schedule a periodic task.
        
        Args:
            task_id: Unique identifier for the task
            task_func: Async function to execute
            schedule: Schedule string (e.g., "daily", "hourly", "weekly")
            priority: Task priority level
            max_retries: Maximum retry attempts
        """
        self.tasks[task_id] = {
            "func": task_func,
            "schedule": schedule,
            "priority": priority,
            "max_retries": max_retries,
            "last_run": None,
            "next_run": self._calculate_next_run(schedule),
            "run_count": 0,
            "error_count": 0
        }
        
        logger.info(f"Scheduled task {task_id} with schedule {schedule}")
    
    def _calculate_next_run(self, schedule: str) -> datetime:
        """Calculate next run time based on schedule."""
        now = datetime.now()
        
        if schedule == "hourly":
            return now + timedelta(hours=1)
        elif schedule == "daily":
            return now + timedelta(days=1)
        elif schedule == "weekly":
            return now + timedelta(weeks=1)
        elif schedule == "monthly":
            return now + timedelta(days=30)
        else:
            # Default to daily
            return now + timedelta(days=1)
    
    async def run_task(self, task_id: str) -> Dict[str, Any]:
        """Run a specific task."""
        if task_id not in self.tasks:
            raise ValueError(f"Task {task_id} not found")
        
        task = self.tasks[task_id]
        
        try:
            logger.info(f"Running task {task_id}")
            result = await task["func"]()
            
            task["last_run"] = datetime.now()
            task["next_run"] = self._calculate_next_run(task["schedule"])
            task["run_count"] += 1
            
            return {
                "success": True,
                "task_id": task_id,
                "result": result,
                "run_count": task["run_count"]
            }
        except Exception as e:
            task["error_count"] += 1
            logger.error(f"Task {task_id} failed: {e}")
            
            if task["error_count"] < task["max_retries"]:
                logger.info(f"Retrying task {task_id} ({task['error_count']}/{task['max_retries']})")
                return await self.run_task(task_id)
            
            return {
                "success": False,
                "task_id": task_id,
                "error": str(e),
                "error_count": task["error_count"]
            }
    
    async def start(self):
        """Start the scheduler."""
        self.running = True
        logger.info("Task scheduler started")
        
        while self.running:
            await self._process_tasks()
            await asyncio.sleep(60)  # Check every minute
    
    async def _process_tasks(self):
        """Process due tasks."""
        now = datetime.now()
        
        for task_id, task in self.tasks.items():
            if task["next_run"] and task["next_run"] <= now:
                # Run task in background
                asyncio.create_task(self.run_task(task_id))
    
    def stop(self):
        """Stop the scheduler."""
        self.running = False
        logger.info("Task scheduler stopped")
    
    def get_task_stats(self) -> Dict[str, Any]:
        """Get statistics about scheduled tasks."""
        return {
            "total_tasks": len(self.tasks),
            "tasks": {
                task_id: {
                    "schedule": task["schedule"],
                    "priority": task["priority"].value,
                    "run_count": task["run_count"],
                    "error_count": task["error_count"],
                    "last_run": task["last_run"].isoformat() if task["last_run"] else None,
                    "next_run": task["next_run"].isoformat() if task["next_run"] else None
                }
                for task_id, task in self.tasks.items()
            }
        }




