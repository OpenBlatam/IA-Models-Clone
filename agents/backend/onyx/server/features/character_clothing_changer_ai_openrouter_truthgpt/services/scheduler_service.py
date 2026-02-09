"""
Scheduler Service
=================
Service for scheduling tasks to run at specific times or intervals
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Callable, Awaitable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import uuid

logger = logging.getLogger(__name__)


class ScheduleType(Enum):
    """Schedule types"""
    ONCE = "once"  # Run once at specific time
    INTERVAL = "interval"  # Run at fixed intervals
    CRON = "cron"  # Run on cron schedule
    DAILY = "daily"  # Run daily at specific time
    WEEKLY = "weekly"  # Run weekly on specific day


@dataclass
class ScheduledTask:
    """Scheduled task definition"""
    id: str
    name: str
    task: Callable[[], Awaitable[Any]]
    schedule_type: ScheduleType
    schedule_config: Dict[str, Any]
    enabled: bool = True
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    run_count: int = 0
    error_count: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class SchedulerService:
    """
    Service for scheduling tasks.
    
    Features:
    - One-time tasks
    - Interval-based tasks
    - Daily/weekly schedules
    - Task status tracking
    - Error handling
    """
    
    def __init__(self):
        """Initialize scheduler service"""
        self._tasks: Dict[str, ScheduledTask] = {}
        self._running = False
        self._scheduler_task: Optional[asyncio.Task] = None
    
    def schedule_once(
        self,
        name: str,
        task: Callable[[], Awaitable[Any]],
        run_at: datetime,
        task_id: Optional[str] = None
    ) -> ScheduledTask:
        """
        Schedule task to run once at specific time.
        
        Args:
            name: Task name
            task: Async task function
            run_at: When to run the task
            task_id: Optional task ID
        
        Returns:
            ScheduledTask
        """
        if task_id is None:
            task_id = f"task_{uuid.uuid4().hex[:12]}"
        
        scheduled_task = ScheduledTask(
            id=task_id,
            name=name,
            task=task,
            schedule_type=ScheduleType.ONCE,
            schedule_config={'run_at': run_at.isoformat()},
            next_run=run_at
        )
        
        self._tasks[task_id] = scheduled_task
        logger.info(f"Scheduled one-time task '{name}' (ID: {task_id}) to run at {run_at}")
        
        return scheduled_task
    
    def schedule_interval(
        self,
        name: str,
        task: Callable[[], Awaitable[Any]],
        interval_seconds: float,
        task_id: Optional[str] = None,
        start_immediately: bool = False
    ) -> ScheduledTask:
        """
        Schedule task to run at fixed intervals.
        
        Args:
            name: Task name
            task: Async task function
            interval_seconds: Interval in seconds
            task_id: Optional task ID
            start_immediately: Whether to run immediately
        
        Returns:
            ScheduledTask
        """
        if task_id is None:
            task_id = f"task_{uuid.uuid4().hex[:12]}"
        
        next_run = datetime.now() if start_immediately else datetime.now() + timedelta(seconds=interval_seconds)
        
        scheduled_task = ScheduledTask(
            id=task_id,
            name=name,
            task=task,
            schedule_type=ScheduleType.INTERVAL,
            schedule_config={'interval_seconds': interval_seconds},
            next_run=next_run
        )
        
        self._tasks[task_id] = scheduled_task
        logger.info(f"Scheduled interval task '{name}' (ID: {task_id}) every {interval_seconds}s")
        
        return scheduled_task
    
    def schedule_daily(
        self,
        name: str,
        task: Callable[[], Awaitable[Any]],
        time: str,  # Format: "HH:MM"
        task_id: Optional[str] = None
    ) -> ScheduledTask:
        """
        Schedule task to run daily at specific time.
        
        Args:
            name: Task name
            task: Async task function
            time: Time in "HH:MM" format
            task_id: Optional task ID
        
        Returns:
            ScheduledTask
        """
        if task_id is None:
            task_id = f"task_{uuid.uuid4().hex[:12]}"
        
        hour, minute = map(int, time.split(':'))
        now = datetime.now()
        next_run = datetime(now.year, now.month, now.day, hour, minute)
        
        # If time has passed today, schedule for tomorrow
        if next_run < now:
            next_run += timedelta(days=1)
        
        scheduled_task = ScheduledTask(
            id=task_id,
            name=name,
            task=task,
            schedule_type=ScheduleType.DAILY,
            schedule_config={'time': time},
            next_run=next_run
        )
        
        self._tasks[task_id] = scheduled_task
        logger.info(f"Scheduled daily task '{name}' (ID: {task_id}) at {time}")
        
        return scheduled_task
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a scheduled task"""
        if task_id in self._tasks:
            del self._tasks[task_id]
            logger.info(f"Cancelled scheduled task {task_id}")
            return True
        return False
    
    async def start(self):
        """Start scheduler"""
        if self._running:
            return
        
        self._running = True
        self._scheduler_task = asyncio.create_task(self._scheduler_loop())
        logger.info("Scheduler service started")
    
    async def stop(self):
        """Stop scheduler"""
        if not self._running:
            return
        
        self._running = False
        
        if self._scheduler_task:
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Scheduler service stopped")
    
    async def _scheduler_loop(self):
        """Main scheduler loop"""
        while self._running:
            try:
                now = datetime.now()
                tasks_to_run = []
                
                # Find tasks that should run
                for task in self._tasks.values():
                    if not task.enabled:
                        continue
                    
                    if task.next_run and task.next_run <= now:
                        tasks_to_run.append(task)
                
                # Run tasks
                for task in tasks_to_run:
                    asyncio.create_task(self._run_task(task))
                
                # Sleep for a short time
                await asyncio.sleep(1)
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")
                await asyncio.sleep(1)
    
    async def _run_task(self, task: ScheduledTask):
        """Run a scheduled task"""
        try:
            task.last_run = datetime.now()
            task.run_count += 1
            
            await task.task()
            
            # Calculate next run time
            if task.schedule_type == ScheduleType.ONCE:
                # Remove one-time tasks after running
                if task.id in self._tasks:
                    del self._tasks[task.id]
            elif task.schedule_type == ScheduleType.INTERVAL:
                interval = task.schedule_config['interval_seconds']
                task.next_run = datetime.now() + timedelta(seconds=interval)
            elif task.schedule_type == ScheduleType.DAILY:
                time_str = task.schedule_config['time']
                hour, minute = map(int, time_str.split(':'))
                now = datetime.now()
                next_run = datetime(now.year, now.month, now.day, hour, minute)
                if next_run <= now:
                    next_run += timedelta(days=1)
                task.next_run = next_run
            
            logger.info(f"Scheduled task '{task.name}' (ID: {task.id}) completed")
        
        except Exception as e:
            task.error_count += 1
            logger.error(f"Scheduled task '{task.name}' (ID: {task.id}) failed: {e}")
    
    def get_task(self, task_id: str) -> Optional[ScheduledTask]:
        """Get scheduled task by ID"""
        return self._tasks.get(task_id)
    
    def list_tasks(self) -> List[ScheduledTask]:
        """List all scheduled tasks"""
        return list(self._tasks.values())
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get scheduler statistics"""
        return {
            'total_tasks': len(self._tasks),
            'enabled_tasks': len([t for t in self._tasks.values() if t.enabled]),
            'running': self._running,
            'tasks': [
                {
                    'id': t.id,
                    'name': t.name,
                    'schedule_type': t.schedule_type.value,
                    'enabled': t.enabled,
                    'run_count': t.run_count,
                    'error_count': t.error_count,
                    'next_run': t.next_run.isoformat() if t.next_run else None
                }
                for t in self._tasks.values()
            ]
        }


# Global scheduler service instance
_scheduler_service: Optional[SchedulerService] = None


def get_scheduler_service() -> SchedulerService:
    """Get or create scheduler service instance"""
    global _scheduler_service
    if _scheduler_service is None:
        _scheduler_service = SchedulerService()
    return _scheduler_service

