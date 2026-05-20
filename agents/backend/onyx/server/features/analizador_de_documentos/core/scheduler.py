"""
Scheduler for Document Analyzer
=================================

Advanced task scheduling with cron-like expressions.
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import re

logger = logging.getLogger(__name__)

@dataclass
class ScheduledTask:
    """Scheduled task definition"""
    task_id: str
    func: Callable
    schedule: str  # Cron expression
    enabled: bool = True
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    run_count: int = 0

class Scheduler:
    """Advanced scheduler with cron-like expressions"""
    
    def __init__(self):
        self.tasks: Dict[str, ScheduledTask] = {}
        self.scheduler_task: Optional[asyncio.Task] = None
        self.is_running = False
        logger.info("Scheduler initialized")
    
    def schedule_task(
        self,
        task_id: str,
        func: Callable,
        schedule: str,
        enabled: bool = True
    ):
        """Schedule a task"""
        task = ScheduledTask(
            task_id=task_id,
            func=func,
            schedule=schedule,
            enabled=enabled
        )
        
        task.next_run = self._calculate_next_run(schedule)
        self.tasks[task_id] = task
        logger.info(f"Scheduled task: {task_id} with schedule: {schedule}")
    
    def _calculate_next_run(self, schedule: str) -> datetime:
        """Calculate next run time from cron expression"""
        # Simple implementation - supports: "every N seconds", "every N minutes", cron-like
        now = datetime.now()
        
        if schedule.startswith("every "):
            # Parse "every N seconds/minutes/hours"
            match = re.match(r"every (\d+) (second|minute|hour|day)s?", schedule)
            if match:
                value = int(match.group(1))
                unit = match.group(2)
                
                if unit == "second":
                    return now + timedelta(seconds=value)
                elif unit == "minute":
                    return now + timedelta(minutes=value)
                elif unit == "hour":
                    return now + timedelta(hours=value)
                elif unit == "day":
                    return now + timedelta(days=value)
        
        # Default: every minute
        return now + timedelta(minutes=1)
    
    async def start(self):
        """Start the scheduler"""
        if self.is_running:
            return
        
        self.is_running = True
        
        async def scheduler_loop():
            while self.is_running:
                try:
                    now = datetime.now()
                    
                    for task_id, task in self.tasks.items():
                        if not task.enabled:
                            continue
                        
                        if task.next_run and now >= task.next_run:
                            # Execute task
                            try:
                                logger.info(f"Executing scheduled task: {task_id}")
                                if asyncio.iscoroutinefunction(task.func):
                                    await task.func()
                                else:
                                    await asyncio.to_thread(task.func)
                                
                                task.last_run = now
                                task.run_count += 1
                            except Exception as e:
                                logger.error(f"Error executing task {task_id}: {e}")
                            
                            # Calculate next run
                            task.next_run = self._calculate_next_run(task.schedule)
                    
                    await asyncio.sleep(1)  # Check every second
                
                except Exception as e:
                    logger.error(f"Error in scheduler loop: {e}")
                    await asyncio.sleep(1)
        
        self.scheduler_task = asyncio.create_task(scheduler_loop())
        logger.info("Scheduler started")
    
    def stop(self):
        """Stop the scheduler"""
        self.is_running = False
        if self.scheduler_task:
            self.scheduler_task.cancel()
        logger.info("Scheduler stopped")
    
    def enable_task(self, task_id: str):
        """Enable a task"""
        if task_id in self.tasks:
            self.tasks[task_id].enabled = True
            logger.info(f"Enabled task: {task_id}")
    
    def disable_task(self, task_id: str):
        """Disable a task"""
        if task_id in self.tasks:
            self.tasks[task_id].enabled = False
            logger.info(f"Disabled task: {task_id}")
    
    def get_task_status(self) -> Dict[str, Any]:
        """Get task status"""
        return {
            task_id: {
                "enabled": task.enabled,
                "schedule": task.schedule,
                "last_run": task.last_run.isoformat() if task.last_run else None,
                "next_run": task.next_run.isoformat() if task.next_run else None,
                "run_count": task.run_count
            }
            for task_id, task in self.tasks.items()
        }

# Global instance
scheduler = Scheduler()
















