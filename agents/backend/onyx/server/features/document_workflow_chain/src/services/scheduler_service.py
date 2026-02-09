"""
Scheduler Service - Advanced Implementation
==========================================

Advanced scheduler service with cron-like scheduling and task management.
"""

from __future__ import annotations
import logging
import asyncio
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from enum import Enum
import uuid
from dataclasses import dataclass
import croniter

logger = logging.getLogger(__name__)


class ScheduleStatus(str, Enum):
    """Schedule status enumeration"""
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


class ScheduleType(str, Enum):
    """Schedule type enumeration"""
    CRON = "cron"
    INTERVAL = "interval"
    ONCE = "once"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


@dataclass
class Schedule:
    """Schedule data class"""
    id: str
    name: str
    schedule_type: ScheduleType
    cron_expression: Optional[str]
    interval_seconds: Optional[int]
    start_time: datetime
    end_time: Optional[datetime]
    function: Callable
    args: tuple
    kwargs: dict
    status: ScheduleStatus
    created_at: datetime
    last_run: Optional[datetime]
    next_run: Optional[datetime]
    run_count: int
    max_runs: Optional[int]
    timeout: Optional[int]
    metadata: Dict[str, Any]


class SchedulerService:
    """Advanced scheduler service with cron-like scheduling"""
    
    def __init__(self):
        self.schedules: Dict[str, Schedule] = {}
        self.running_tasks: Dict[str, asyncio.Task] = {}
        self.is_running = False
        self.scheduler_task: Optional[asyncio.Task] = None
        
        # Statistics
        self.stats = {
            "total_schedules": 0,
            "active_schedules": 0,
            "paused_schedules": 0,
            "completed_schedules": 0,
            "failed_schedules": 0,
            "total_runs": 0,
            "successful_runs": 0,
            "failed_runs": 0,
            "schedules_by_type": {schedule_type.value: 0 for schedule_type in ScheduleType}
        }
    
    async def start(self):
        """Start scheduler service"""
        try:
            if not self.is_running:
                self.is_running = True
                self.scheduler_task = asyncio.create_task(self._scheduler_loop())
                logger.info("Scheduler service started")
        
        except Exception as e:
            logger.error(f"Failed to start scheduler service: {e}")
            raise
    
    async def stop(self):
        """Stop scheduler service"""
        try:
            if self.is_running:
                self.is_running = False
                
                # Cancel scheduler task
                if self.scheduler_task:
                    self.scheduler_task.cancel()
                    try:
                        await self.scheduler_task
                    except asyncio.CancelledError:
                        pass
                
                # Cancel all running tasks
                for task_id, task in self.running_tasks.items():
                    task.cancel()
                
                # Wait for tasks to complete
                if self.running_tasks:
                    await asyncio.gather(*self.running_tasks.values(), return_exceptions=True)
                
                logger.info("Scheduler service stopped")
        
        except Exception as e:
            logger.error(f"Failed to stop scheduler service: {e}")
    
    async def add_cron_schedule(
        self,
        name: str,
        cron_expression: str,
        function: Callable,
        args: tuple = (),
        kwargs: dict = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        max_runs: Optional[int] = None,
        timeout: Optional[int] = None,
        metadata: Dict[str, Any] = None
    ) -> str:
        """Add cron-based schedule"""
        try:
            schedule_id = str(uuid.uuid4())
            
            # Validate cron expression
            try:
                croniter.croniter(cron_expression)
            except Exception as e:
                raise ValueError(f"Invalid cron expression: {e}")
            
            # Calculate next run time
            now = datetime.utcnow()
            start = start_time or now
            cron = croniter.croniter(cron_expression, start)
            next_run = cron.get_next(datetime)
            
            schedule = Schedule(
                id=schedule_id,
                name=name,
                schedule_type=ScheduleType.CRON,
                cron_expression=cron_expression,
                interval_seconds=None,
                start_time=start,
                end_time=end_time,
                function=function,
                args=args,
                kwargs=kwargs or {},
                status=ScheduleStatus.ACTIVE,
                created_at=now,
                last_run=None,
                next_run=next_run,
                run_count=0,
                max_runs=max_runs,
                timeout=timeout,
                metadata=metadata or {}
            )
            
            self.schedules[schedule_id] = schedule
            self._update_statistics()
            
            logger.info(f"Cron schedule added: {schedule_id} - {name}")
            return schedule_id
        
        except Exception as e:
            logger.error(f"Failed to add cron schedule: {e}")
            raise
    
    async def add_interval_schedule(
        self,
        name: str,
        interval_seconds: int,
        function: Callable,
        args: tuple = (),
        kwargs: dict = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        max_runs: Optional[int] = None,
        timeout: Optional[int] = None,
        metadata: Dict[str, Any] = None
    ) -> str:
        """Add interval-based schedule"""
        try:
            schedule_id = str(uuid.uuid4())
            
            now = datetime.utcnow()
            start = start_time or now
            next_run = start + timedelta(seconds=interval_seconds)
            
            schedule = Schedule(
                id=schedule_id,
                name=name,
                schedule_type=ScheduleType.INTERVAL,
                cron_expression=None,
                interval_seconds=interval_seconds,
                start_time=start,
                end_time=end_time,
                function=function,
                args=args,
                kwargs=kwargs or {},
                status=ScheduleStatus.ACTIVE,
                created_at=now,
                last_run=None,
                next_run=next_run,
                run_count=0,
                max_runs=max_runs,
                timeout=timeout,
                metadata=metadata or {}
            )
            
            self.schedules[schedule_id] = schedule
            self._update_statistics()
            
            logger.info(f"Interval schedule added: {schedule_id} - {name}")
            return schedule_id
        
        except Exception as e:
            logger.error(f"Failed to add interval schedule: {e}")
            raise
    
    async def add_daily_schedule(
        self,
        name: str,
        hour: int,
        minute: int,
        function: Callable,
        args: tuple = (),
        kwargs: dict = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        max_runs: Optional[int] = None,
        timeout: Optional[int] = None,
        metadata: Dict[str, Any] = None
    ) -> str:
        """Add daily schedule"""
        try:
            cron_expression = f"{minute} {hour} * * *"
            return await self.add_cron_schedule(
                name=name,
                cron_expression=cron_expression,
                function=function,
                args=args,
                kwargs=kwargs,
                start_time=start_time,
                end_time=end_time,
                max_runs=max_runs,
                timeout=timeout,
                metadata=metadata
            )
        
        except Exception as e:
            logger.error(f"Failed to add daily schedule: {e}")
            raise
    
    async def add_weekly_schedule(
        self,
        name: str,
        day_of_week: int,
        hour: int,
        minute: int,
        function: Callable,
        args: tuple = (),
        kwargs: dict = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        max_runs: Optional[int] = None,
        timeout: Optional[int] = None,
        metadata: Dict[str, Any] = None
    ) -> str:
        """Add weekly schedule"""
        try:
            cron_expression = f"{minute} {hour} * * {day_of_week}"
            return await self.add_cron_schedule(
                name=name,
                cron_expression=cron_expression,
                function=function,
                args=args,
                kwargs=kwargs,
                start_time=start_time,
                end_time=end_time,
                max_runs=max_runs,
                timeout=timeout,
                metadata=metadata
            )
        
        except Exception as e:
            logger.error(f"Failed to add weekly schedule: {e}")
            raise
    
    async def add_monthly_schedule(
        self,
        name: str,
        day_of_month: int,
        hour: int,
        minute: int,
        function: Callable,
        args: tuple = (),
        kwargs: dict = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        max_runs: Optional[int] = None,
        timeout: Optional[int] = None,
        metadata: Dict[str, Any] = None
    ) -> str:
        """Add monthly schedule"""
        try:
            cron_expression = f"{minute} {hour} {day_of_month} * *"
            return await self.add_cron_schedule(
                name=name,
                cron_expression=cron_expression,
                function=function,
                args=args,
                kwargs=kwargs,
                start_time=start_time,
                end_time=end_time,
                max_runs=max_runs,
                timeout=timeout,
                metadata=metadata
            )
        
        except Exception as e:
            logger.error(f"Failed to add monthly schedule: {e}")
            raise
    
    async def pause_schedule(self, schedule_id: str) -> bool:
        """Pause schedule"""
        try:
            if schedule_id in self.schedules:
                schedule = self.schedules[schedule_id]
                schedule.status = ScheduleStatus.PAUSED
                self._update_statistics()
                
                logger.info(f"Schedule paused: {schedule_id}")
                return True
            
            return False
        
        except Exception as e:
            logger.error(f"Failed to pause schedule: {e}")
            return False
    
    async def resume_schedule(self, schedule_id: str) -> bool:
        """Resume schedule"""
        try:
            if schedule_id in self.schedules:
                schedule = self.schedules[schedule_id]
                schedule.status = ScheduleStatus.ACTIVE
                self._update_statistics()
                
                logger.info(f"Schedule resumed: {schedule_id}")
                return True
            
            return False
        
        except Exception as e:
            logger.error(f"Failed to resume schedule: {e}")
            return False
    
    async def cancel_schedule(self, schedule_id: str) -> bool:
        """Cancel schedule"""
        try:
            if schedule_id in self.schedules:
                schedule = self.schedules[schedule_id]
                schedule.status = ScheduleStatus.CANCELLED
                self._update_statistics()
                
                # Cancel running task if exists
                if schedule_id in self.running_tasks:
                    self.running_tasks[schedule_id].cancel()
                    del self.running_tasks[schedule_id]
                
                logger.info(f"Schedule cancelled: {schedule_id}")
                return True
            
            return False
        
        except Exception as e:
            logger.error(f"Failed to cancel schedule: {e}")
            return False
    
    async def get_schedule(self, schedule_id: str) -> Optional[Dict[str, Any]]:
        """Get schedule information"""
        try:
            if schedule_id in self.schedules:
                schedule = self.schedules[schedule_id]
                return {
                    "id": schedule.id,
                    "name": schedule.name,
                    "type": schedule.schedule_type.value,
                    "cron_expression": schedule.cron_expression,
                    "interval_seconds": schedule.interval_seconds,
                    "status": schedule.status.value,
                    "created_at": schedule.created_at.isoformat(),
                    "last_run": schedule.last_run.isoformat() if schedule.last_run else None,
                    "next_run": schedule.next_run.isoformat() if schedule.next_run else None,
                    "run_count": schedule.run_count,
                    "max_runs": schedule.max_runs,
                    "timeout": schedule.timeout,
                    "metadata": schedule.metadata
                }
            
            return None
        
        except Exception as e:
            logger.error(f"Failed to get schedule: {e}")
            return None
    
    async def list_schedules(
        self,
        status: Optional[ScheduleStatus] = None,
        schedule_type: Optional[ScheduleType] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """List schedules with filtering"""
        try:
            filtered_schedules = []
            
            for schedule in self.schedules.values():
                if status and schedule.status != status:
                    continue
                if schedule_type and schedule.schedule_type != schedule_type:
                    continue
                
                filtered_schedules.append({
                    "id": schedule.id,
                    "name": schedule.name,
                    "type": schedule.schedule_type.value,
                    "status": schedule.status.value,
                    "created_at": schedule.created_at.isoformat(),
                    "last_run": schedule.last_run.isoformat() if schedule.last_run else None,
                    "next_run": schedule.next_run.isoformat() if schedule.next_run else None,
                    "run_count": schedule.run_count,
                    "metadata": schedule.metadata
                })
            
            # Sort by next_run (soonest first)
            filtered_schedules.sort(
                key=lambda x: x["next_run"] or "9999-12-31T23:59:59",
                reverse=False
            )
            
            return filtered_schedules[:limit]
        
        except Exception as e:
            logger.error(f"Failed to list schedules: {e}")
            return []
    
    async def get_scheduler_stats(self) -> Dict[str, Any]:
        """Get scheduler statistics"""
        try:
            return {
                "is_running": self.is_running,
                "total_schedules": self.stats["total_schedules"],
                "active_schedules": self.stats["active_schedules"],
                "paused_schedules": self.stats["paused_schedules"],
                "completed_schedules": self.stats["completed_schedules"],
                "failed_schedules": self.stats["failed_schedules"],
                "total_runs": self.stats["total_runs"],
                "successful_runs": self.stats["successful_runs"],
                "failed_runs": self.stats["failed_runs"],
                "schedules_by_type": self.stats["schedules_by_type"],
                "running_tasks": len(self.running_tasks),
                "timestamp": datetime.utcnow().isoformat()
            }
        
        except Exception as e:
            logger.error(f"Failed to get scheduler stats: {e}")
            return {"error": str(e)}
    
    async def _scheduler_loop(self):
        """Main scheduler loop"""
        try:
            while self.is_running:
                now = datetime.utcnow()
                
                # Check for schedules that need to run
                for schedule_id, schedule in self.schedules.items():
                    if (schedule.status == ScheduleStatus.ACTIVE and
                        schedule.next_run and
                        schedule.next_run <= now and
                        schedule_id not in self.running_tasks):
                        
                        # Check if schedule has reached max runs
                        if schedule.max_runs and schedule.run_count >= schedule.max_runs:
                            schedule.status = ScheduleStatus.COMPLETED
                            continue
                        
                        # Check if schedule has reached end time
                        if schedule.end_time and now >= schedule.end_time:
                            schedule.status = ScheduleStatus.COMPLETED
                            continue
                        
                        # Execute schedule
                        await self._execute_schedule(schedule_id)
                
                # Wait before next iteration
                await asyncio.sleep(1)
        
        except asyncio.CancelledError:
            logger.info("Scheduler loop cancelled")
        except Exception as e:
            logger.error(f"Scheduler loop error: {e}")
    
    async def _execute_schedule(self, schedule_id: str):
        """Execute scheduled task"""
        try:
            schedule = self.schedules[schedule_id]
            
            # Create task
            task = asyncio.create_task(
                self._run_scheduled_function(schedule)
            )
            self.running_tasks[schedule_id] = task
            
            # Update schedule
            schedule.last_run = datetime.utcnow()
            schedule.run_count += 1
            self.stats["total_runs"] += 1
            
            # Calculate next run time
            await self._calculate_next_run(schedule)
            
            logger.info(f"Executing schedule: {schedule_id} - {schedule.name}")
        
        except Exception as e:
            logger.error(f"Failed to execute schedule: {e}")
    
    async def _run_scheduled_function(self, schedule: Schedule):
        """Run scheduled function"""
        try:
            if asyncio.iscoroutinefunction(schedule.function):
                if schedule.timeout:
                    result = await asyncio.wait_for(
                        schedule.function(*schedule.args, **schedule.kwargs),
                        timeout=schedule.timeout
                    )
                else:
                    result = await schedule.function(*schedule.args, **schedule.kwargs)
            else:
                if schedule.timeout:
                    result = await asyncio.wait_for(
                        asyncio.get_event_loop().run_in_executor(
                            None, schedule.function, *schedule.args, **schedule.kwargs
                        ),
                        timeout=schedule.timeout
                    )
                else:
                    result = await asyncio.get_event_loop().run_in_executor(
                        None, schedule.function, *schedule.args, **schedule.kwargs
                    )
            
            self.stats["successful_runs"] += 1
            logger.info(f"Schedule completed successfully: {schedule.id} - {schedule.name}")
            
        except asyncio.TimeoutError:
            self.stats["failed_runs"] += 1
            logger.error(f"Schedule timed out: {schedule.id} - {schedule.name}")
        except Exception as e:
            self.stats["failed_runs"] += 1
            logger.error(f"Schedule failed: {schedule.id} - {schedule.name}: {e}")
        finally:
            # Remove from running tasks
            if schedule.id in self.running_tasks:
                del self.running_tasks[schedule.id]
    
    async def _calculate_next_run(self, schedule: Schedule):
        """Calculate next run time for schedule"""
        try:
            now = datetime.utcnow()
            
            if schedule.schedule_type == ScheduleType.CRON:
                if schedule.cron_expression:
                    cron = croniter.croniter(schedule.cron_expression, now)
                    schedule.next_run = cron.get_next(datetime)
            
            elif schedule.schedule_type == ScheduleType.INTERVAL:
                if schedule.interval_seconds:
                    schedule.next_run = now + timedelta(seconds=schedule.interval_seconds)
            
            elif schedule.schedule_type == ScheduleType.ONCE:
                schedule.next_run = None
                schedule.status = ScheduleStatus.COMPLETED
            
            # Check if next run exceeds end time
            if schedule.end_time and schedule.next_run and schedule.next_run > schedule.end_time:
                schedule.next_run = None
                schedule.status = ScheduleStatus.COMPLETED
        
        except Exception as e:
            logger.error(f"Failed to calculate next run: {e}")
    
    def _update_statistics(self):
        """Update scheduler statistics"""
        try:
            self.stats["total_schedules"] = len(self.schedules)
            self.stats["active_schedules"] = len([s for s in self.schedules.values() if s.status == ScheduleStatus.ACTIVE])
            self.stats["paused_schedules"] = len([s for s in self.schedules.values() if s.status == ScheduleStatus.PAUSED])
            self.stats["completed_schedules"] = len([s for s in self.schedules.values() if s.status == ScheduleStatus.COMPLETED])
            self.stats["failed_schedules"] = len([s for s in self.schedules.values() if s.status == ScheduleStatus.FAILED])
            
            # Update by type
            for schedule_type in ScheduleType:
                self.stats["schedules_by_type"][schedule_type.value] = len([
                    s for s in self.schedules.values() if s.schedule_type == schedule_type
                ])
        
        except Exception as e:
            logger.error(f"Failed to update statistics: {e}")


# Global scheduler service instance
scheduler_service = SchedulerService()

