"""
Event Scheduler for Color Grading AI
=====================================

Advanced event scheduling and task management.
"""

import logging
import asyncio
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import uuid

logger = logging.getLogger(__name__)


class ScheduleType(Enum):
    """Schedule types."""
    ONCE = "once"  # Execute once
    INTERVAL = "interval"  # Execute at intervals
    CRON = "cron"  # Cron-like schedule
    DAILY = "daily"  # Daily at specific time
    WEEKLY = "weekly"  # Weekly on specific day


@dataclass
class ScheduledEvent:
    """Scheduled event."""
    event_id: str
    name: str
    schedule_type: ScheduleType
    handler: Callable
    parameters: Dict[str, Any] = field(default_factory=dict)
    next_run: Optional[datetime] = None
    interval: Optional[timedelta] = None
    cron_expression: Optional[str] = None
    enabled: bool = True
    max_runs: Optional[int] = None
    run_count: int = 0
    last_run: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class EventScheduler:
    """
    Event scheduler.
    
    Features:
    - One-time scheduling
    - Interval scheduling
    - Cron-like scheduling
    - Daily/weekly schedules
    - Event management
    - Execution tracking
    """
    
    def __init__(self):
        """Initialize event scheduler."""
        self._events: Dict[str, ScheduledEvent] = {}
        self._running = False
        self._scheduler_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()
    
    async def schedule_once(
        self,
        name: str,
        handler: Callable,
        run_at: datetime,
        parameters: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Schedule one-time event.
        
        Args:
            name: Event name
            handler: Handler function
            run_at: When to run
            parameters: Optional parameters
            
        Returns:
            Event ID
        """
        event_id = str(uuid.uuid4())
        event = ScheduledEvent(
            event_id=event_id,
            name=name,
            schedule_type=ScheduleType.ONCE,
            handler=handler,
            parameters=parameters or {},
            next_run=run_at
        )
        
        async with self._lock:
            self._events[event_id] = event
        
        logger.info(f"Scheduled one-time event: {name} at {run_at}")
        return event_id
    
    async def schedule_interval(
        self,
        name: str,
        handler: Callable,
        interval: timedelta,
        parameters: Optional[Dict[str, Any]] = None,
        max_runs: Optional[int] = None
    ) -> str:
        """
        Schedule interval event.
        
        Args:
            name: Event name
            handler: Handler function
            interval: Interval between runs
            parameters: Optional parameters
            max_runs: Optional maximum runs
            
        Returns:
            Event ID
        """
        event_id = str(uuid.uuid4())
        event = ScheduledEvent(
            event_id=event_id,
            name=name,
            schedule_type=ScheduleType.INTERVAL,
            handler=handler,
            parameters=parameters or {},
            interval=interval,
            next_run=datetime.now() + interval,
            max_runs=max_runs
        )
        
        async with self._lock:
            self._events[event_id] = event
        
        logger.info(f"Scheduled interval event: {name} (interval: {interval})")
        return event_id
    
    async def schedule_daily(
        self,
        name: str,
        handler: Callable,
        time: str,  # HH:MM format
        parameters: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Schedule daily event.
        
        Args:
            name: Event name
            handler: Handler function
            time: Time in HH:MM format
            parameters: Optional parameters
            
        Returns:
            Event ID
        """
        event_id = str(uuid.uuid4())
        
        # Calculate next run time
        hour, minute = map(int, time.split(":"))
        now = datetime.now()
        next_run = datetime(now.year, now.month, now.day, hour, minute)
        if next_run <= now:
            next_run += timedelta(days=1)
        
        event = ScheduledEvent(
            event_id=event_id,
            name=name,
            schedule_type=ScheduleType.DAILY,
            handler=handler,
            parameters=parameters or {},
            next_run=next_run
        )
        
        async with self._lock:
            self._events[event_id] = event
        
        logger.info(f"Scheduled daily event: {name} at {time}")
        return event_id
    
    async def cancel(self, event_id: str) -> bool:
        """
        Cancel scheduled event.
        
        Args:
            event_id: Event ID
            
        Returns:
            True if cancelled
        """
        async with self._lock:
            if event_id in self._events:
                del self._events[event_id]
                logger.info(f"Cancelled event: {event_id}")
                return True
            return False
    
    async def enable(self, event_id: str) -> bool:
        """Enable event."""
        async with self._lock:
            if event_id in self._events:
                self._events[event_id].enabled = True
                return True
            return False
    
    async def disable(self, event_id: str) -> bool:
        """Disable event."""
        async with self._lock:
            if event_id in self._events:
                self._events[event_id].enabled = False
                return True
            return False
    
    async def start(self):
        """Start scheduler."""
        if self._running:
            return
        
        self._running = True
        self._scheduler_task = asyncio.create_task(self._scheduler_loop())
        logger.info("Event scheduler started")
    
    async def stop(self):
        """Stop scheduler."""
        self._running = False
        if self._scheduler_task:
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass
        logger.info("Event scheduler stopped")
    
    async def _scheduler_loop(self):
        """Scheduler main loop."""
        while self._running:
            try:
                await asyncio.sleep(1.0)  # Check every second
                await self._check_and_execute()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")
    
    async def _check_and_execute(self):
        """Check and execute due events."""
        now = datetime.now()
        events_to_run = []
        
        async with self._lock:
            for event in self._events.values():
                if not event.enabled:
                    continue
                
                if event.next_run and event.next_run <= now:
                    # Check max runs
                    if event.max_runs and event.run_count >= event.max_runs:
                        continue
                    
                    events_to_run.append(event)
        
        # Execute events
        for event in events_to_run:
            try:
                await self._execute_event(event)
            except Exception as e:
                logger.error(f"Error executing event {event.name}: {e}")
    
    async def _execute_event(self, event: ScheduledEvent):
        """Execute a scheduled event."""
        logger.info(f"Executing scheduled event: {event.name}")
        
        # Execute handler
        if asyncio.iscoroutinefunction(event.handler):
            await event.handler(**event.parameters)
        else:
            event.handler(**event.parameters)
        
        # Update event
        async with self._lock:
            event.run_count += 1
            event.last_run = datetime.now()
            
            # Calculate next run
            if event.schedule_type == ScheduleType.ONCE:
                # Remove one-time events
                if event.event_id in self._events:
                    del self._events[event.event_id]
            elif event.schedule_type == ScheduleType.INTERVAL:
                if event.interval:
                    event.next_run = datetime.now() + event.interval
            elif event.schedule_type == ScheduleType.DAILY:
                event.next_run = event.next_run + timedelta(days=1)
    
    def get_events(self, enabled_only: bool = False) -> List[ScheduledEvent]:
        """Get all events."""
        events = list(self._events.values())
        if enabled_only:
            events = [e for e in events if e.enabled]
        return events
    
    def get_event(self, event_id: str) -> Optional[ScheduledEvent]:
        """Get event by ID."""
        return self._events.get(event_id)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get scheduler statistics."""
        total_events = len(self._events)
        enabled_events = sum(1 for e in self._events.values() if e.enabled)
        total_runs = sum(e.run_count for e in self._events.values())
        
        return {
            "total_events": total_events,
            "enabled_events": enabled_events,
            "disabled_events": total_events - enabled_events,
            "total_runs": total_runs,
            "running": self._running,
        }


