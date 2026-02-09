"""
Task Scheduler
==============

Advanced task scheduling with cron-like expressions and async execution.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import re

logger = logging.getLogger(__name__)

@dataclass
class ScheduledTask:
    """Scheduled task definition."""
    task_id: str
    func: Callable
    schedule: str  # cron expression or interval
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    enabled: bool = True
    args: tuple = ()
    kwargs: dict = None
    max_runs: Optional[int] = None
    run_count: int = 0

class TaskScheduler:
    """Advanced task scheduler."""
    
    def __init__(self):
        self.tasks: Dict[str, ScheduledTask] = {}
        self.is_running = False
        self.scheduler_task = None
        self.check_interval = 1.0  # seconds
    
    def schedule_task(
        self,
        task_id: str,
        func: Callable,
        schedule: str,
        args: tuple = (),
        kwargs: Optional[dict] = None,
        max_runs: Optional[int] = None
    ) -> ScheduledTask:
        """Schedule a task."""
        task = ScheduledTask(
            task_id=task_id,
            func=func,
            schedule=schedule,
            args=args,
            kwargs=kwargs or {},
            max_runs=max_runs
        )
        
        # Calculate next run
        task.next_run = self._calculate_next_run(schedule)
        
        self.tasks[task_id] = task
        logger.info(f"Task scheduled: {task_id} (next run: {task.next_run})")
        
        return task
    
    def _calculate_next_run(self, schedule: str) -> datetime:
        """Calculate next run time from schedule."""
        now = datetime.now()
        
        # Check if it's an interval (e.g., "5m", "1h", "30s")
        interval_match = re.match(r'^(\d+)([smhd])$', schedule.lower())
        if interval_match:
            value = int(interval_match.group(1))
            unit = interval_match.group(2)
            
            if unit == 's':
                delta = timedelta(seconds=value)
            elif unit == 'm':
                delta = timedelta(minutes=value)
            elif unit == 'h':
                delta = timedelta(hours=value)
            elif unit == 'd':
                delta = timedelta(days=value)
            else:
                delta = timedelta(minutes=1)
            
            return now + delta
        
        # Simple cron: "minute hour day month weekday"
        # For now, treat as interval
        return now + timedelta(minutes=1)
    
    async def _execute_task(self, task: ScheduledTask):
        """Execute a scheduled task."""
        try:
            logger.info(f"Executing task: {task.task_id}")
            
            if asyncio.iscoroutinefunction(task.func):
                await task.func(*task.args, **task.kwargs)
            else:
                # Run in executor for sync functions
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, task.func, *task.args, **task.kwargs)
            
            task.run_count += 1
            task.last_run = datetime.now()
            
            # Calculate next run
            task.next_run = self._calculate_next_run(task.schedule)
            
            logger.info(f"Task completed: {task.task_id} (run #{task.run_count})")
            
        except Exception as e:
            logger.error(f"Task execution failed: {task.task_id} - {e}")
    
    async def _scheduler_loop(self):
        """Main scheduler loop."""
        while self.is_running:
            try:
                now = datetime.now()
                
                for task in self.tasks.values():
                    if not task.enabled:
                        continue
                    
                    # Check max runs
                    if task.max_runs and task.run_count >= task.max_runs:
                        continue
                    
                    # Check if it's time to run
                    if task.next_run and now >= task.next_run:
                        asyncio.create_task(self._execute_task(task))
                
                await asyncio.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"Scheduler loop error: {e}")
                await asyncio.sleep(self.check_interval)
    
    async def start(self):
        """Start scheduler."""
        if self.is_running:
            return
        
        self.is_running = True
        self.scheduler_task = asyncio.create_task(self._scheduler_loop())
        logger.info("Task scheduler started")
    
    async def stop(self):
        """Stop scheduler."""
        self.is_running = False
        if self.scheduler_task:
            self.scheduler_task.cancel()
            try:
                await self.scheduler_task
            except asyncio.CancelledError:
                pass
        logger.info("Task scheduler stopped")
    
    def enable_task(self, task_id: str):
        """Enable a task."""
        if task_id in self.tasks:
            self.tasks[task_id].enabled = True
            logger.info(f"Task enabled: {task_id}")
    
    def disable_task(self, task_id: str):
        """Disable a task."""
        if task_id in self.tasks:
            self.tasks[task_id].enabled = False
            logger.info(f"Task disabled: {task_id}")
    
    def remove_task(self, task_id: str):
        """Remove a task."""
        if task_id in self.tasks:
            del self.tasks[task_id]
            logger.info(f"Task removed: {task_id}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get scheduler statistics."""
        return {
            "is_running": self.is_running,
            "total_tasks": len(self.tasks),
            "enabled_tasks": sum(1 for t in self.tasks.values() if t.enabled),
            "tasks": {
                task_id: {
                    "enabled": task.enabled,
                    "run_count": task.run_count,
                    "last_run": task.last_run.isoformat() if task.last_run else None,
                    "next_run": task.next_run.isoformat() if task.next_run else None,
                    "max_runs": task.max_runs
                }
                for task_id, task in self.tasks.items()
            }
        }

# Global instance
task_scheduler = TaskScheduler()
































