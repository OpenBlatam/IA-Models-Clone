"""
Scheduler
=========

System for scheduling tasks and operations.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Callable, Awaitable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import croniter

logger = logging.getLogger(__name__)


class ScheduleType(Enum):
    """Schedule type."""
    ONCE = "once"
    INTERVAL = "interval"
    CRON = "cron"


@dataclass
class ScheduledTask:
    """Scheduled task definition."""
    id: str
    name: str
    task: Callable[[], Awaitable[Any]]
    schedule_type: ScheduleType
    schedule_value: str  # ISO datetime, interval seconds, or cron expression
    enabled: bool = True
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    run_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


class Scheduler:
    """Task scheduler."""
    
    def __init__(self):
        """Initialize scheduler."""
        self.tasks: Dict[str, ScheduledTask] = {}
        self.running = False
        self._task: Optional[asyncio.Task] = None
    
    def schedule(
        self,
        task_id: str,
        name: str,
        task: Callable[[], Awaitable[Any]],
        schedule_type: ScheduleType,
        schedule_value: str,
        enabled: bool = True,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Schedule a task.
        
        Args:
            task_id: Task ID
            name: Task name
            task: Task function
            schedule_type: Schedule type
            schedule_value: Schedule value
            enabled: Whether task is enabled
            metadata: Optional metadata
        """
        scheduled_task = ScheduledTask(
            id=task_id,
            name=name,
            task=task,
            schedule_type=schedule_type,
            schedule_value=schedule_value,
            enabled=enabled,
            metadata=metadata or {}
        )
        
        # Calculate next run
        scheduled_task.next_run = self._calculate_next_run(scheduled_task)
        
        self.tasks[task_id] = scheduled_task
        logger.info(f"Scheduled task {name} (ID: {task_id})")
    
    def _calculate_next_run(self, task: ScheduledTask) -> Optional[datetime]:
        """Calculate next run time."""
        now = datetime.now()
        
        if task.schedule_type == ScheduleType.ONCE:
            try:
                next_run = datetime.fromisoformat(task.schedule_value)
                return next_run if next_run > now else None
            except Exception:
                return None
        
        elif task.schedule_type == ScheduleType.INTERVAL:
            try:
                interval = float(task.schedule_value)
                if task.last_run:
                    return task.last_run + timedelta(seconds=interval)
                else:
                    return now + timedelta(seconds=interval)
            except Exception:
                return None
        
        elif task.schedule_type == ScheduleType.CRON:
            try:
                cron = croniter.croniter(task.schedule_value, now)
                return cron.get_next(datetime)
            except Exception:
                return None
        
        return None
    
    async def start(self):
        """Start scheduler."""
        if self.running:
            return
        
        self.running = True
        self._task = asyncio.create_task(self._scheduler_loop())
        logger.info("Scheduler started")
    
    async def stop(self):
        """Stop scheduler."""
        self.running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Scheduler stopped")
    
    async def _scheduler_loop(self):
        """Scheduler main loop."""
        while self.running:
            try:
                now = datetime.now()
                
                # Check all tasks
                for task_id, task in list(self.tasks.items()):
                    if not task.enabled:
                        continue
                    
                    if task.next_run and now >= task.next_run:
                        # Execute task
                        asyncio.create_task(self._execute_task(task))
                        
                        # Update next run
                        task.last_run = now
                        task.next_run = self._calculate_next_run(task)
                        task.run_count += 1
                
                # Sleep for 1 second
                await asyncio.sleep(1.0)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")
                await asyncio.sleep(1.0)
    
    async def _execute_task(self, task: ScheduledTask):
        """Execute scheduled task."""
        try:
            logger.info(f"Executing scheduled task: {task.name}")
            await task.task()
            logger.info(f"Completed scheduled task: {task.name}")
        except Exception as e:
            logger.error(f"Error executing scheduled task {task.name}: {e}")
    
    def enable(self, task_id: str):
        """Enable a task."""
        if task_id in self.tasks:
            self.tasks[task_id].enabled = True
            self.tasks[task_id].next_run = self._calculate_next_run(self.tasks[task_id])
    
    def disable(self, task_id: str):
        """Disable a task."""
        if task_id in self.tasks:
            self.tasks[task_id].enabled = False
    
    def remove(self, task_id: str):
        """Remove a task."""
        self.tasks.pop(task_id, None)
    
    def get_task(self, task_id: str) -> Optional[ScheduledTask]:
        """Get task by ID."""
        return self.tasks.get(task_id)
    
    def get_all_tasks(self) -> List[ScheduledTask]:
        """Get all tasks."""
        return list(self.tasks.values())
    
    def get_stats(self) -> Dict[str, Any]:
        """Get scheduler statistics."""
        return {
            "running": self.running,
            "total_tasks": len(self.tasks),
            "enabled_tasks": len([t for t in self.tasks.values() if t.enabled]),
            "tasks": [
                {
                    "id": t.id,
                    "name": t.name,
                    "enabled": t.enabled,
                    "run_count": t.run_count,
                    "next_run": t.next_run.isoformat() if t.next_run else None
                }
                for t in self.tasks.values()
            ]
        }




