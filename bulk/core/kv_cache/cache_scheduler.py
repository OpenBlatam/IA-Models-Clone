"""
Cache scheduling utilities.

Provides scheduling capabilities for cache operations.
"""
from __future__ import annotations

import logging
import time
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from threading import Timer

logger = logging.getLogger(__name__)


class CacheScheduler:
    """
    Cache scheduler for scheduled operations.
    
    Provides scheduling capabilities for cache tasks.
    """
    
    def __init__(self, cache: Any):
        """
        Initialize cache scheduler.
        
        Args:
            cache: Cache instance
        """
        self.cache = cache
        self.scheduled_tasks: List[Dict[str, Any]] = []
        self.timers: List[Timer] = []
    
    def schedule_at(
        self,
        task_name: str,
        task_fn: Callable,
        scheduled_time: datetime,
        **kwargs
    ) -> None:
        """
        Schedule task at specific time.
        
        Args:
            task_name: Name of task
            task_fn: Task function
            scheduled_time: When to execute
            **kwargs: Arguments for task function
        """
        now = datetime.now()
        delay = (scheduled_time - now).total_seconds()
        
        if delay <= 0:
            logger.warning(f"Task {task_name} scheduled time is in the past")
            return
        
        def execute_task():
            try:
                task_fn(**kwargs)
            except Exception as e:
                logger.error(f"Scheduled task {task_name} failed: {e}")
        
        timer = Timer(delay, execute_task)
        timer.start()
        
        self.scheduled_tasks.append({
            "name": task_name,
            "function": task_fn,
            "scheduled_time": scheduled_time,
            "kwargs": kwargs,
            "timer": timer
        })
        
        self.timers.append(timer)
        logger.info(f"Scheduled task '{task_name}' for {scheduled_time}")
    
    def schedule_interval(
        self,
        task_name: str,
        task_fn: Callable,
        interval: float,
        **kwargs
    ) -> None:
        """
        Schedule recurring task.
        
        Args:
            task_name: Name of task
            task_fn: Task function
            interval: Interval in seconds
            **kwargs: Arguments for task function
        """
        def execute_recurring():
            try:
                task_fn(**kwargs)
            except Exception as e:
                logger.error(f"Recurring task {task_name} failed: {e}")
            
            # Schedule next execution
            timer = Timer(interval, execute_recurring)
            timer.start()
            self.timers.append(timer)
        
        # Start first execution
        timer = Timer(interval, execute_recurring)
        timer.start()
        
        self.scheduled_tasks.append({
            "name": task_name,
            "function": task_fn,
            "interval": interval,
            "kwargs": kwargs,
            "recurring": True
        })
        
        self.timers.append(timer)
        logger.info(f"Scheduled recurring task '{task_name}' every {interval}s")
    
    def cancel_task(self, task_name: str) -> bool:
        """
        Cancel scheduled task.
        
        Args:
            task_name: Name of task to cancel
            
        Returns:
            True if cancelled
        """
        for task in self.scheduled_tasks:
            if task["name"] == task_name:
                if "timer" in task:
                    task["timer"].cancel()
                self.scheduled_tasks.remove(task)
                logger.info(f"Cancelled task '{task_name}'")
                return True
        return False
    
    def cancel_all(self) -> None:
        """Cancel all scheduled tasks."""
        for timer in self.timers:
            timer.cancel()
        self.timers.clear()
        self.scheduled_tasks.clear()
        logger.info("Cancelled all scheduled tasks")
    
    def list_tasks(self) -> List[Dict[str, Any]]:
        """
        List all scheduled tasks.
        
        Returns:
            List of task information
        """
        return [
            {
                "name": task["name"],
                "scheduled_time": task.get("scheduled_time"),
                "interval": task.get("interval"),
                "recurring": task.get("recurring", False)
            }
            for task in self.scheduled_tasks
        ]


class CacheMaintenanceScheduler:
    """
    Maintenance scheduler for cache.
    
    Provides scheduled maintenance tasks.
    """
    
    def __init__(self, cache: Any):
        """
        Initialize maintenance scheduler.
        
        Args:
            cache: Cache instance
        """
        self.cache = cache
        self.scheduler = CacheScheduler(cache)
    
    def schedule_daily_backup(self, hour: int = 2) -> None:
        """
        Schedule daily backup.
        
        Args:
            hour: Hour of day (0-23)
        """
        from kv_cache import CacheBackupManager
        
        backup_manager = CacheBackupManager(self.cache)
        
        def daily_backup():
            backup_manager.create_backup(f"daily_backup_{datetime.now().strftime('%Y%m%d')}")
        
        # Schedule for today at specified hour
        now = datetime.now()
        scheduled_time = datetime(now.year, now.month, now.day, hour)
        
        if scheduled_time < now:
            scheduled_time += timedelta(days=1)
        
        self.scheduler.schedule_at("daily_backup", daily_backup, scheduled_time)
    
    def schedule_weekly_cleanup(self, day: int = 0, hour: int = 3) -> None:
        """
        Schedule weekly cleanup.
        
        Args:
            day: Day of week (0=Monday, 6=Sunday)
            hour: Hour of day
        """
        from kv_cache import CacheRepair
        
        repair = CacheRepair(self.cache)
        
        def weekly_cleanup():
            repair.cleanup_orphaned_entries()
            repair.repair_invalid_entries()
        
        # Calculate next scheduled time
        now = datetime.now()
        days_ahead = (day - now.weekday()) % 7
        if days_ahead == 0 and now.hour >= hour:
            days_ahead = 7
        
        scheduled_time = now.replace(hour=hour, minute=0, second=0) + timedelta(days=days_ahead)
        
        self.scheduler.schedule_at("weekly_cleanup", weekly_cleanup, scheduled_time)

