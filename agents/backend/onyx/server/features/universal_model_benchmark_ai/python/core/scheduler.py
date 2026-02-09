"""
Scheduler Module - Task scheduling system.

Provides:
- Cron-like scheduling
- One-time tasks
- Recurring tasks
- Task persistence
"""

import logging
import time
import threading
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import re

logger = logging.getLogger(__name__)


class ScheduleType(str, Enum):
    """Schedule type."""
    ONCE = "once"
    INTERVAL = "interval"
    CRON = "cron"


@dataclass
class ScheduledTask:
    """Scheduled task definition."""
    id: str
    name: str
    task_type: str
    payload: Dict[str, Any]
    schedule_type: ScheduleType
    schedule: str  # cron expression, interval seconds, or datetime
    enabled: bool = True
    last_run: Optional[str] = None
    next_run: Optional[str] = None
    run_count: int = 0
    max_runs: Optional[int] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "task_type": self.task_type,
            "payload": self.payload,
            "schedule_type": self.schedule_type.value,
            "schedule": self.schedule,
            "enabled": self.enabled,
            "last_run": self.last_run,
            "next_run": self.next_run,
            "run_count": self.run_count,
            "max_runs": self.max_runs,
            "created_at": self.created_at,
        }


class CronParser:
    """Cron expression parser."""
    
    @staticmethod
    def parse(cron_expr: str) -> Dict[str, Any]:
        """
        Parse cron expression.
        
        Format: minute hour day month weekday
        Example: "0 9 * * *" (9 AM daily)
        
        Args:
            cron_expr: Cron expression
            
        Returns:
            Parsed cron parts
        """
        parts = cron_expr.split()
        if len(parts) != 5:
            raise ValueError("Cron expression must have 5 parts")
        
        return {
            "minute": parts[0],
            "hour": parts[1],
            "day": parts[2],
            "month": parts[3],
            "weekday": parts[4],
        }
    
    @staticmethod
    def matches(cron_expr: str, dt: datetime) -> bool:
        """
        Check if datetime matches cron expression.
        
        Args:
            cron_expr: Cron expression
            dt: Datetime to check
            
        Returns:
            True if matches
        """
        parts = CronParser.parse(cron_expr)
        
        def matches_field(value: int, field_expr: str) -> bool:
            if field_expr == "*":
                return True
            if "/" in field_expr:
                step = int(field_expr.split("/")[1])
                return value % step == 0
            if "-" in field_expr:
                start, end = map(int, field_expr.split("-"))
                return start <= value <= end
            if "," in field_expr:
                return value in [int(x) for x in field_expr.split(",")]
            return value == int(field_expr)
        
        return (
            matches_field(dt.minute, parts["minute"]) and
            matches_field(dt.hour, parts["hour"]) and
            matches_field(dt.day, parts["day"]) and
            matches_field(dt.month, parts["month"]) and
            matches_field(dt.weekday(), parts["weekday"])
        )
    
    @staticmethod
    def next_run(cron_expr: str, from_dt: Optional[datetime] = None) -> datetime:
        """
        Calculate next run time from cron expression.
        
        Args:
            cron_expr: Cron expression
            from_dt: Starting datetime (defaults to now)
            
        Returns:
            Next run datetime
        """
        if from_dt is None:
            from_dt = datetime.now()
        
        # Simple implementation - check next 24 hours
        for i in range(1440):  # 24 hours in minutes
            check_dt = from_dt + timedelta(minutes=i)
            if CronParser.matches(cron_expr, check_dt):
                return check_dt
        
        # Fallback: next day same time
        return from_dt + timedelta(days=1)


class TaskScheduler:
    """Task scheduler."""
    
    def __init__(self, task_queue=None):
        """
        Initialize scheduler.
        
        Args:
            task_queue: Optional task queue for enqueuing tasks
        """
        self.scheduled_tasks: Dict[str, ScheduledTask] = {}
        self.task_queue = task_queue
        self.running = False
        self.worker_thread: Optional[threading.Thread] = None
        self.handlers: Dict[str, Callable] = {}
    
    def register_handler(self, task_type: str, handler: Callable) -> None:
        """
        Register task handler.
        
        Args:
            task_type: Task type
            handler: Handler function
        """
        self.handlers[task_type] = handler
    
    def schedule_task(
        self,
        name: str,
        task_type: str,
        payload: Dict[str, Any],
        schedule_type: ScheduleType,
        schedule: str,
        max_runs: Optional[int] = None,
    ) -> ScheduledTask:
        """
        Schedule a task.
        
        Args:
            name: Task name
            task_type: Task type
            payload: Task payload
            schedule_type: Schedule type
            schedule: Schedule expression
            max_runs: Maximum number of runs
            
        Returns:
            Scheduled task
        """
        task_id = f"scheduled_{int(time.time() * 1000)}"
        
        # Calculate next run
        next_run = None
        if schedule_type == ScheduleType.ONCE:
            next_run = datetime.fromisoformat(schedule).isoformat()
        elif schedule_type == ScheduleType.CRON:
            next_run = CronParser.next_run(schedule).isoformat()
        elif schedule_type == ScheduleType.INTERVAL:
            next_run = (datetime.now() + timedelta(seconds=int(schedule))).isoformat()
        
        task = ScheduledTask(
            id=task_id,
            name=name,
            task_type=task_type,
            payload=payload,
            schedule_type=schedule_type,
            schedule=schedule,
            next_run=next_run,
            max_runs=max_runs,
        )
        
        self.scheduled_tasks[task_id] = task
        logger.info(f"Scheduled task: {name} (id: {task_id})")
        
        return task
    
    def _should_run(self, task: ScheduledTask) -> bool:
        """Check if task should run now."""
        if not task.enabled:
            return False
        
        if task.max_runs and task.run_count >= task.max_runs:
            return False
        
        if not task.next_run:
            return False
        
        next_run_dt = datetime.fromisoformat(task.next_run)
        return datetime.now() >= next_run_dt
    
    def _execute_task(self, task: ScheduledTask) -> None:
        """Execute scheduled task."""
        if task.task_type in self.handlers:
            handler = self.handlers[task.task_type]
            try:
                handler(task.payload)
                task.run_count += 1
                task.last_run = datetime.now().isoformat()
                logger.info(f"Executed scheduled task: {task.name}")
            except Exception as e:
                logger.error(f"Error executing task {task.name}: {e}")
        elif self.task_queue:
            # Enqueue to task queue
            self.task_queue.enqueue(
                task_type=task.task_type,
                payload=task.payload,
                priority=5,
            )
            task.run_count += 1
            task.last_run = datetime.now().isoformat()
            logger.info(f"Enqueued scheduled task: {task.name}")
        
        # Calculate next run
        self._calculate_next_run(task)
    
    def _calculate_next_run(self, task: ScheduledTask) -> None:
        """Calculate next run time for task."""
        if task.schedule_type == ScheduleType.ONCE:
            task.enabled = False
            task.next_run = None
        elif task.schedule_type == ScheduleType.INTERVAL:
            interval = int(task.schedule)
            task.next_run = (datetime.now() + timedelta(seconds=interval)).isoformat()
        elif task.schedule_type == ScheduleType.CRON:
            task.next_run = CronParser.next_run(task.schedule).isoformat()
    
    def scheduler_loop(self) -> None:
        """Scheduler loop."""
        while self.running:
            for task in list(self.scheduled_tasks.values()):
                if self._should_run(task):
                    self._execute_task(task)
            time.sleep(1)  # Check every second
    
    def start(self) -> None:
        """Start scheduler."""
        if self.running:
            return
        
        self.running = True
        self.worker_thread = threading.Thread(target=self.scheduler_loop, daemon=True)
        self.worker_thread.start()
        logger.info("Scheduler started")
    
    def stop(self) -> None:
        """Stop scheduler."""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5)
        logger.info("Scheduler stopped")
    
    def get_task(self, task_id: str) -> Optional[ScheduledTask]:
        """Get scheduled task by ID."""
        return self.scheduled_tasks.get(task_id)
    
    def list_tasks(self) -> List[ScheduledTask]:
        """List all scheduled tasks."""
        return list(self.scheduled_tasks.values())
    
    def enable_task(self, task_id: str) -> None:
        """Enable a scheduled task."""
        task = self.scheduled_tasks.get(task_id)
        if task:
            task.enabled = True
            logger.info(f"Enabled task: {task.name}")
    
    def disable_task(self, task_id: str) -> None:
        """Disable a scheduled task."""
        task = self.scheduled_tasks.get(task_id)
        if task:
            task.enabled = False
            logger.info(f"Disabled task: {task.name}")












