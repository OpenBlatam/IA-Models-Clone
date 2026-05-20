"""
Task Scheduler
==============

Persistent task scheduling for autonomous 24/7 agent.
"""

import asyncio
import logging
import json
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict
import re

logger = logging.getLogger(__name__)


@dataclass
class ScheduledTask:
    """A scheduled task configuration."""
    id: str
    name: str
    image_path: str
    text_prompt: str
    cron_expression: str
    priority: int = 0
    enabled: bool = True
    last_run: Optional[str] = None
    next_run: Optional[str] = None
    created_at: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class CronParser:
    """Simple cron expression parser."""
    
    @staticmethod
    def parse(expression: str) -> Dict[str, Any]:
        """
        Parse a cron expression.
        
        Supports: minute hour day month weekday
        Example: "0 */2 * * *" = every 2 hours
        """
        parts = expression.split()
        if len(parts) != 5:
            raise ValueError(f"Invalid cron expression: {expression}")
        
        return {
            "minute": parts[0],
            "hour": parts[1],
            "day": parts[2],
            "month": parts[3],
            "weekday": parts[4],
        }
    
    @staticmethod
    def matches(expression: str, dt: datetime) -> bool:
        """Check if datetime matches cron expression."""
        parsed = CronParser.parse(expression)
        
        checks = [
            CronParser._field_matches(parsed["minute"], dt.minute, 0, 59),
            CronParser._field_matches(parsed["hour"], dt.hour, 0, 23),
            CronParser._field_matches(parsed["day"], dt.day, 1, 31),
            CronParser._field_matches(parsed["month"], dt.month, 1, 12),
            CronParser._field_matches(parsed["weekday"], dt.weekday(), 0, 6),
        ]
        
        return all(checks)
    
    @staticmethod
    def _field_matches(field: str, value: int, min_val: int, max_val: int) -> bool:
        """Check if a value matches a cron field."""
        if field == "*":
            return True
        
        # Handle */n (every n)
        if field.startswith("*/"):
            step = int(field[2:])
            return value % step == 0
        
        # Handle comma-separated values
        if "," in field:
            values = [int(v) for v in field.split(",")]
            return value in values
        
        # Handle range (e.g., 1-5)
        if "-" in field:
            start, end = field.split("-")
            return int(start) <= value <= int(end)
        
        # Handle single value
        return int(field) == value
    
    @staticmethod
    def get_next_run(expression: str, from_time: Optional[datetime] = None) -> datetime:
        """Calculate next run time from cron expression."""
        if from_time is None:
            from_time = datetime.now()
        
        # Start checking from next minute
        check_time = from_time.replace(second=0, microsecond=0) + timedelta(minutes=1)
        
        # Check up to 1 year ahead
        max_iterations = 525600  # minutes in a year
        
        for _ in range(max_iterations):
            if CronParser.matches(expression, check_time):
                return check_time
            check_time += timedelta(minutes=1)
        
        raise ValueError(f"Could not find next run time for: {expression}")


class TaskScheduler:
    """
    Manages scheduled tasks with persistence.
    
    Features:
    - Cron-style scheduling
    - Task persistence (survives restarts)
    - Enable/disable schedules
    - Schedule history
    """
    
    def __init__(
        self,
        storage_path: str = "scheduled_tasks.json",
        task_callback: Optional[Callable] = None
    ):
        """
        Initialize task scheduler.
        
        Args:
            storage_path: Path to persist schedules
            task_callback: Async callback to submit tasks
        """
        self.storage_path = Path(storage_path)
        self.task_callback = task_callback
        self._schedules: Dict[str, ScheduledTask] = {}
        self._running = False
        self._scheduler_task: Optional[asyncio.Task] = None
        
        logger.info(f"Initialized TaskScheduler (storage: {self.storage_path})")
    
    async def start(self):
        """Start the scheduler."""
        if self._running:
            logger.warning("TaskScheduler is already running")
            return
        
        # Load persisted schedules
        await self.load_schedules()
        
        self._running = True
        self._scheduler_task = asyncio.create_task(self._scheduler_loop())
        logger.info(f"TaskScheduler started with {len(self._schedules)} schedules")
    
    async def stop(self):
        """Stop the scheduler."""
        if not self._running:
            return
        
        self._running = False
        if self._scheduler_task:
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass
        
        # Save schedules before stopping
        await self.save_schedules()
        logger.info("TaskScheduler stopped")
    
    async def schedule(
        self,
        name: str,
        image_path: str,
        text_prompt: str,
        cron_expression: str,
        priority: int = 0,
    ) -> str:
        """
        Schedule a new recurring task.
        
        Args:
            name: Task name
            image_path: Path to input image
            text_prompt: Text prompt for segmentation
            cron_expression: Cron expression (e.g., "0 */2 * * *")
            priority: Task priority
            
        Returns:
            Schedule ID
        """
        import uuid
        schedule_id = str(uuid.uuid4())
        
        # Validate cron expression
        CronParser.parse(cron_expression)
        next_run = CronParser.get_next_run(cron_expression)
        
        schedule = ScheduledTask(
            id=schedule_id,
            name=name,
            image_path=image_path,
            text_prompt=text_prompt,
            cron_expression=cron_expression,
            priority=priority,
            enabled=True,
            next_run=next_run.isoformat(),
            created_at=datetime.now().isoformat(),
        )
        
        self._schedules[schedule_id] = schedule
        await self.save_schedules()
        
        logger.info(f"Created schedule {schedule_id}: {name} (next: {next_run})")
        return schedule_id
    
    async def remove_schedule(self, schedule_id: str):
        """Remove a schedule."""
        if schedule_id not in self._schedules:
            raise ValueError(f"Schedule {schedule_id} not found")
        
        del self._schedules[schedule_id]
        await self.save_schedules()
        logger.info(f"Removed schedule {schedule_id}")
    
    async def enable_schedule(self, schedule_id: str):
        """Enable a schedule."""
        if schedule_id not in self._schedules:
            raise ValueError(f"Schedule {schedule_id} not found")
        
        self._schedules[schedule_id].enabled = True
        await self.save_schedules()
        logger.info(f"Enabled schedule {schedule_id}")
    
    async def disable_schedule(self, schedule_id: str):
        """Disable a schedule."""
        if schedule_id not in self._schedules:
            raise ValueError(f"Schedule {schedule_id} not found")
        
        self._schedules[schedule_id].enabled = False
        await self.save_schedules()
        logger.info(f"Disabled schedule {schedule_id}")
    
    def get_schedules(self) -> List[Dict[str, Any]]:
        """Get all schedules."""
        return [s.to_dict() for s in self._schedules.values()]
    
    def get_schedule(self, schedule_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific schedule."""
        if schedule_id in self._schedules:
            return self._schedules[schedule_id].to_dict()
        return None
    
    async def _scheduler_loop(self):
        """Main scheduler loop."""
        while self._running:
            try:
                now = datetime.now()
                
                for schedule in self._schedules.values():
                    if not schedule.enabled:
                        continue
                    
                    if schedule.next_run:
                        next_run = datetime.fromisoformat(schedule.next_run)
                        
                        if now >= next_run:
                            # Time to run this task
                            await self._execute_schedule(schedule)
                            
                            # Calculate next run
                            schedule.last_run = now.isoformat()
                            schedule.next_run = CronParser.get_next_run(
                                schedule.cron_expression
                            ).isoformat()
                
                await asyncio.sleep(60)  # Check every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Scheduler error: {e}", exc_info=True)
                await asyncio.sleep(60)
    
    async def _execute_schedule(self, schedule: ScheduledTask):
        """Execute a scheduled task."""
        logger.info(f"Executing scheduled task: {schedule.name}")
        
        if self.task_callback:
            try:
                await self.task_callback(
                    image_path=schedule.image_path,
                    text_prompt=schedule.text_prompt,
                    priority=schedule.priority,
                )
            except Exception as e:
                logger.error(f"Failed to execute schedule {schedule.id}: {e}")
    
    async def save_schedules(self):
        """Save schedules to disk."""
        data = {
            "schedules": [s.to_dict() for s in self._schedules.values()],
            "saved_at": datetime.now().isoformat(),
        }
        
        with open(self.storage_path, "w") as f:
            json.dump(data, f, indent=2)
    
    async def load_schedules(self):
        """Load schedules from disk."""
        if not self.storage_path.exists():
            return
        
        try:
            with open(self.storage_path, "r") as f:
                data = json.load(f)
            
            for schedule_data in data.get("schedules", []):
                schedule = ScheduledTask(**schedule_data)
                self._schedules[schedule.id] = schedule
            
            logger.info(f"Loaded {len(self._schedules)} schedules from disk")
        except Exception as e:
            logger.error(f"Error loading schedules: {e}")
