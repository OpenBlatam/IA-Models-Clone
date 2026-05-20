"""Task scheduling system"""
from typing import Dict, Any, Optional, Callable
from datetime import datetime, timedelta
import asyncio
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class TaskScheduler:
    """Schedule and manage tasks"""
    
    def __init__(self, tasks_dir: Optional[str] = None):
        """
        Initialize task scheduler
        
        Args:
            tasks_dir: Directory for storing tasks
        """
        if tasks_dir is None:
            from config import settings
            tasks_dir = settings.temp_dir + "/scheduled_tasks"
        
        self.tasks_dir = Path(tasks_dir)
        self.tasks_dir.mkdir(parents=True, exist_ok=True)
        self._scheduled_tasks: Dict[str, asyncio.Task] = {}
        self._task_definitions: Dict[str, Dict[str, Any]] = {}
    
    def schedule_task(
        self,
        task_id: str,
        task_func: Callable,
        schedule: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Schedule a task
        
        Args:
            task_id: Unique task ID
            task_func: Async function to execute
            schedule: Schedule configuration
            metadata: Optional metadata
            
        Returns:
            Task ID
        """
        task_def = {
            "id": task_id,
            "function": task_func.__name__,
            "schedule": schedule,
            "metadata": metadata or {},
            "created_at": datetime.now().isoformat(),
            "status": "scheduled"
        }
        
        # Save task definition
        task_file = self.tasks_dir / f"{task_id}.json"
        with open(task_file, 'w') as f:
            json.dump(task_def, f, indent=2)
        
        self._task_definitions[task_id] = task_def
        
        # Schedule task
        if schedule.get("type") == "interval":
            interval = schedule.get("seconds", 60)
            task = asyncio.create_task(self._run_interval_task(task_id, task_func, interval))
        elif schedule.get("type") == "cron":
            # Cron-like scheduling (simplified)
            task = asyncio.create_task(self._run_cron_task(task_id, task_func, schedule))
        elif schedule.get("type") == "once":
            delay = schedule.get("delay_seconds", 0)
            task = asyncio.create_task(self._run_once_task(task_id, task_func, delay))
        else:
            raise ValueError(f"Unknown schedule type: {schedule.get('type')}")
        
        self._scheduled_tasks[task_id] = task
        
        return task_id
    
    async def _run_interval_task(
        self,
        task_id: str,
        task_func: Callable,
        interval: int
    ) -> None:
        """Run task at intervals"""
        while True:
            try:
                await asyncio.sleep(interval)
                await task_func()
                self._update_task_status(task_id, "completed")
            except Exception as e:
                logger.error(f"Error in scheduled task {task_id}: {e}")
                self._update_task_status(task_id, "error", str(e))
                await asyncio.sleep(interval)
    
    async def _run_cron_task(
        self,
        task_id: str,
        task_func: Callable,
        schedule: Dict[str, Any]
    ) -> None:
        """Run task on cron schedule"""
        # Simplified cron - check every minute
        while True:
            try:
                now = datetime.now()
                minute = now.minute
                hour = now.hour
                day = now.day
                
                cron_minute = schedule.get("minute", "*")
                cron_hour = schedule.get("hour", "*")
                cron_day = schedule.get("day", "*")
                
                should_run = (
                    (cron_minute == "*" or minute == cron_minute) and
                    (cron_hour == "*" or hour == cron_hour) and
                    (cron_day == "*" or day == cron_day)
                )
                
                if should_run:
                    await task_func()
                    self._update_task_status(task_id, "completed")
                
                await asyncio.sleep(60)  # Check every minute
            except Exception as e:
                logger.error(f"Error in cron task {task_id}: {e}")
                self._update_task_status(task_id, "error", str(e))
                await asyncio.sleep(60)
    
    async def _run_once_task(
        self,
        task_id: str,
        task_func: Callable,
        delay: int
    ) -> None:
        """Run task once after delay"""
        await asyncio.sleep(delay)
        try:
            await task_func()
            self._update_task_status(task_id, "completed")
        except Exception as e:
            logger.error(f"Error in one-time task {task_id}: {e}")
            self._update_task_status(task_id, "error", str(e))
    
    def cancel_task(self, task_id: str) -> bool:
        """
        Cancel scheduled task
        
        Args:
            task_id: Task ID
            
        Returns:
            True if cancelled, False otherwise
        """
        if task_id in self._scheduled_tasks:
            task = self._scheduled_tasks[task_id]
            task.cancel()
            del self._scheduled_tasks[task_id]
            self._update_task_status(task_id, "cancelled")
            return True
        return False
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task status"""
        return self._task_definitions.get(task_id)
    
    def list_tasks(self) -> List[Dict[str, Any]]:
        """List all scheduled tasks"""
        return list(self._task_definitions.values())
    
    def _update_task_status(
        self,
        task_id: str,
        status: str,
        error: Optional[str] = None
    ) -> None:
        """Update task status"""
        if task_id in self._task_definitions:
            self._task_definitions[task_id]["status"] = status
            self._task_definitions[task_id]["last_run"] = datetime.now().isoformat()
            if error:
                self._task_definitions[task_id]["last_error"] = error
            
            # Save to file
            task_file = self.tasks_dir / f"{task_id}.json"
            if task_file.exists():
                with open(task_file, 'w') as f:
                    json.dump(self._task_definitions[task_id], f, indent=2)


# Global scheduler
_scheduler: Optional[TaskScheduler] = None


def get_scheduler() -> TaskScheduler:
    """Get global scheduler"""
    global _scheduler
    if _scheduler is None:
        _scheduler = TaskScheduler()
    return _scheduler

