"""
Advanced Scheduling for Recovery AI
"""

import schedule
import time
import threading
from typing import Dict, List, Optional, Any, Callable
import logging

logger = logging.getLogger(__name__)


class TaskScheduler:
    """Advanced task scheduler"""
    
    def __init__(self):
        """Initialize scheduler"""
        self.tasks = {}
        self.running = False
        self.thread = None
        
        logger.info("TaskScheduler initialized")
    
    def schedule_task(
        self,
        task_id: str,
        task_func: Callable,
        schedule_type: str = "interval",
        **schedule_kwargs
    ):
        """
        Schedule task
        
        Args:
            task_id: Task identifier
            task_func: Task function
            schedule_type: Schedule type (interval, daily, weekly, cron)
            **schedule_kwargs: Schedule parameters
        """
        if schedule_type == "interval":
            seconds = schedule_kwargs.get("seconds", 60)
            schedule.every(seconds).seconds.do(task_func).tag(task_id)
        
        elif schedule_type == "daily":
            time_str = schedule_kwargs.get("time", "00:00")
            schedule.every().day.at(time_str).do(task_func).tag(task_id)
        
        elif schedule_type == "weekly":
            day = schedule_kwargs.get("day", "monday")
            time_str = schedule_kwargs.get("time", "00:00")
            getattr(schedule.every(), day.lower()).at(time_str).do(task_func).tag(task_id)
        
        self.tasks[task_id] = {
            "func": task_func,
            "type": schedule_type,
            "kwargs": schedule_kwargs
        }
        
        logger.info(f"Task scheduled: {task_id} ({schedule_type})")
    
    def start(self):
        """Start scheduler"""
        if self.running:
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        logger.info("TaskScheduler started")
    
    def _run(self):
        """Run scheduler loop"""
        while self.running:
            schedule.run_pending()
            time.sleep(1)
    
    def stop(self):
        """Stop scheduler"""
        self.running = False
        if self.thread:
            self.thread.join()
        logger.info("TaskScheduler stopped")
    
    def cancel_task(self, task_id: str):
        """Cancel task"""
        schedule.clear(task_id)
        if task_id in self.tasks:
            del self.tasks[task_id]
        logger.info(f"Task cancelled: {task_id}")


class ModelUpdateScheduler:
    """Schedule model updates"""
    
    def __init__(self, scheduler: TaskScheduler):
        """
        Initialize model update scheduler
        
        Args:
            scheduler: Task scheduler
        """
        self.scheduler = scheduler
    
    def schedule_model_update(
        self,
        model_id: str,
        update_func: Callable,
        interval_hours: int = 24
    ):
        """
        Schedule model update
        
        Args:
            model_id: Model identifier
            update_func: Update function
            interval_hours: Update interval in hours
        """
        self.scheduler.schedule_task(
            task_id=f"model_update_{model_id}",
            task_func=update_func,
            schedule_type="interval",
            seconds=interval_hours * 3600
        )
        logger.info(f"Model update scheduled: {model_id} (every {interval_hours}h)")

