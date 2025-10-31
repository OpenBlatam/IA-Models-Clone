"""
Gamma App - Scheduling Service
Advanced task scheduling and cron job management
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import time
from datetime import datetime, timedelta
import uuid
import threading
from croniter import croniter
import pytz
from pathlib import Path

logger = logging.getLogger(__name__)

class ScheduleType(Enum):
    """Schedule types"""
    ONCE = "once"
    INTERVAL = "interval"
    CRON = "cron"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"

class JobStatus(Enum):
    """Job status"""
    PENDING = "pending"
    SCHEDULED = "scheduled"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"

@dataclass
class JobDefinition:
    """Job definition"""
    id: str
    name: str
    description: str
    function: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    schedule_type: ScheduleType = ScheduleType.ONCE
    schedule_config: Dict[str, Any] = field(default_factory=dict)
    timezone: str = "UTC"
    enabled: bool = True
    max_retries: int = 3
    retry_delay: int = 60
    timeout: int = 300
    priority: int = 0
    tags: List[str] = field(default_factory=list)

@dataclass
class JobExecution:
    """Job execution"""
    id: str
    job_id: str
    status: JobStatus
    scheduled_time: datetime
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    result: Any = None
    error: Optional[str] = None
    retry_count: int = 0
    execution_time: float = 0.0

class SchedulingService:
    """Advanced scheduling service"""
    
    def __init__(self):
        self.jobs = {}
        self.executions = {}
        self.job_functions = {}
        self.scheduler_thread = None
        self.running = False
        self.lock = threading.Lock()
        self.job_queue = asyncio.Queue()
        self.executor = asyncio.get_event_loop()
    
    def register_job_function(self, name: str, function: Callable):
        """Register a job function"""
        self.job_functions[name] = function
        logger.info(f"Registered job function: {name}")
    
    def create_job(self, definition: JobDefinition) -> str:
        """Create a new scheduled job"""
        try:
            # Validate job
            self._validate_job(definition)
            
            # Store job
            self.jobs[definition.id] = definition
            
            # Schedule job if enabled
            if definition.enabled:
                self._schedule_job(definition)
            
            logger.info(f"Created job: {definition.name} ({definition.id})")
            return definition.id
            
        except Exception as e:
            logger.error(f"Error creating job: {e}")
            raise
    
    def _validate_job(self, definition: JobDefinition):
        """Validate job definition"""
        # Check function exists
        if definition.function not in self.job_functions:
            raise ValueError(f"Job function not found: {definition.function}")
        
        # Validate schedule config
        if definition.schedule_type == ScheduleType.INTERVAL:
            if 'interval' not in definition.schedule_config:
                raise ValueError("Interval schedule requires 'interval' config")
        
        elif definition.schedule_type == ScheduleType.CRON:
            if 'cron_expression' not in definition.schedule_config:
                raise ValueError("Cron schedule requires 'cron_expression' config")
            
            # Validate cron expression
            try:
                croniter(definition.schedule_config['cron_expression'])
            except Exception as e:
                raise ValueError(f"Invalid cron expression: {e}")
        
        # Validate timezone
        try:
            pytz.timezone(definition.timezone)
        except Exception as e:
            raise ValueError(f"Invalid timezone: {e}")
    
    def _schedule_job(self, job: JobDefinition):
        """Schedule a job for execution"""
        try:
            if job.schedule_type == ScheduleType.ONCE:
                # Schedule for immediate execution
                scheduled_time = datetime.now(pytz.timezone(job.timezone))
                self._add_to_queue(job, scheduled_time)
            
            elif job.schedule_type == ScheduleType.INTERVAL:
                # Schedule recurring interval
                interval_seconds = job.schedule_config['interval']
                scheduled_time = datetime.now(pytz.timezone(job.timezone)) + timedelta(seconds=interval_seconds)
                self._add_to_queue(job, scheduled_time)
            
            elif job.schedule_type == ScheduleType.CRON:
                # Schedule using cron expression
                cron_expr = job.schedule_config['cron_expression']
                cron = croniter(cron_expr, datetime.now(pytz.timezone(job.timezone)))
                scheduled_time = cron.get_next(datetime)
                self._add_to_queue(job, scheduled_time)
            
            elif job.schedule_type == ScheduleType.DAILY:
                # Schedule daily
                hour = job.schedule_config.get('hour', 0)
                minute = job.schedule_config.get('minute', 0)
                scheduled_time = self._get_next_daily_time(hour, minute, job.timezone)
                self._add_to_queue(job, scheduled_time)
            
            elif job.schedule_type == ScheduleType.WEEKLY:
                # Schedule weekly
                day_of_week = job.schedule_config.get('day_of_week', 0)  # Monday = 0
                hour = job.schedule_config.get('hour', 0)
                minute = job.schedule_config.get('minute', 0)
                scheduled_time = self._get_next_weekly_time(day_of_week, hour, minute, job.timezone)
                self._add_to_queue(job, scheduled_time)
            
            elif job.schedule_type == ScheduleType.MONTHLY:
                # Schedule monthly
                day_of_month = job.schedule_config.get('day_of_month', 1)
                hour = job.schedule_config.get('hour', 0)
                minute = job.schedule_config.get('minute', 0)
                scheduled_time = self._get_next_monthly_time(day_of_month, hour, minute, job.timezone)
                self._add_to_queue(job, scheduled_time)
            
        except Exception as e:
            logger.error(f"Error scheduling job: {e}")
    
    def _get_next_daily_time(self, hour: int, minute: int, timezone: str) -> datetime:
        """Get next daily execution time"""
        tz = pytz.timezone(timezone)
        now = datetime.now(tz)
        
        next_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if next_time <= now:
            next_time += timedelta(days=1)
        
        return next_time
    
    def _get_next_weekly_time(self, day_of_week: int, hour: int, minute: int, timezone: str) -> datetime:
        """Get next weekly execution time"""
        tz = pytz.timezone(timezone)
        now = datetime.now(tz)
        
        days_ahead = day_of_week - now.weekday()
        if days_ahead <= 0:  # Target day already happened this week
            days_ahead += 7
        
        next_time = now + timedelta(days=days_ahead)
        next_time = next_time.replace(hour=hour, minute=minute, second=0, microsecond=0)
        
        return next_time
    
    def _get_next_monthly_time(self, day_of_month: int, hour: int, minute: int, timezone: str) -> datetime:
        """Get next monthly execution time"""
        tz = pytz.timezone(timezone)
        now = datetime.now(tz)
        
        # Get next month
        if now.month == 12:
            next_month = now.replace(year=now.year + 1, month=1, day=day_of_month, hour=hour, minute=minute, second=0, microsecond=0)
        else:
            next_month = now.replace(month=now.month + 1, day=day_of_month, hour=hour, minute=minute, second=0, microsecond=0)
        
        # If day doesn't exist in next month, use last day of month
        try:
            return next_month
        except ValueError:
            # Get last day of next month
            if next_month.month == 12:
                last_day = now.replace(year=now.year + 1, month=1, day=1) - timedelta(days=1)
            else:
                last_day = now.replace(month=now.month + 2, day=1) - timedelta(days=1)
            
            return last_day.replace(hour=hour, minute=minute, second=0, microsecond=0)
    
    def _add_to_queue(self, job: JobDefinition, scheduled_time: datetime):
        """Add job to execution queue"""
        try:
            execution_id = str(uuid.uuid4())
            execution = JobExecution(
                id=execution_id,
                job_id=job.id,
                status=JobStatus.SCHEDULED,
                scheduled_time=scheduled_time
            )
            
            self.executions[execution_id] = execution
            
            # Schedule execution
            delay = (scheduled_time - datetime.now(pytz.timezone(job.timezone))).total_seconds()
            if delay > 0:
                asyncio.create_task(self._schedule_execution(execution, delay))
            
        except Exception as e:
            logger.error(f"Error adding job to queue: {e}")
    
    async def _schedule_execution(self, execution: JobExecution, delay: float):
        """Schedule job execution"""
        try:
            await asyncio.sleep(delay)
            
            # Check if job still exists and is enabled
            if execution.job_id not in self.jobs:
                execution.status = JobStatus.CANCELLED
                return
            
            job = self.jobs[execution.job_id]
            if not job.enabled:
                execution.status = JobStatus.CANCELLED
                return
            
            # Execute job
            await self._execute_job(execution, job)
            
            # Reschedule if recurring
            if job.schedule_type in [ScheduleType.INTERVAL, ScheduleType.CRON, ScheduleType.DAILY, ScheduleType.WEEKLY, ScheduleType.MONTHLY]:
                self._schedule_job(job)
            
        except Exception as e:
            logger.error(f"Error in scheduled execution: {e}")
            execution.status = JobStatus.FAILED
            execution.error = str(e)
    
    async def _execute_job(self, execution: JobExecution, job: JobDefinition):
        """Execute a job"""
        try:
            execution.status = JobStatus.RUNNING
            execution.start_time = datetime.now()
            
            # Get job function
            if job.function not in self.job_functions:
                raise ValueError(f"Job function not found: {job.function}")
            
            func = self.job_functions[job.function]
            
            # Execute with retries
            for attempt in range(job.max_retries + 1):
                try:
                    # Execute job
                    if asyncio.iscoroutinefunction(func):
                        result = await asyncio.wait_for(func(**job.parameters), timeout=job.timeout)
                    else:
                        result = await asyncio.get_event_loop().run_in_executor(
                            None, 
                            lambda: func(**job.parameters)
                        )
                    
                    # Job completed successfully
                    execution.status = JobStatus.COMPLETED
                    execution.result = result
                    execution.end_time = datetime.now()
                    execution.execution_time = (execution.end_time - execution.start_time).total_seconds()
                    
                    logger.info(f"Job completed: {job.name} ({job.id})")
                    return
                    
                except asyncio.TimeoutError:
                    error_msg = f"Job timeout: {job.name}"
                    logger.warning(error_msg)
                    if attempt < job.max_retries:
                        execution.retry_count += 1
                        await asyncio.sleep(job.retry_delay)
                        continue
                    else:
                        execution.status = JobStatus.FAILED
                        execution.error = error_msg
                        break
                        
                except Exception as e:
                    error_msg = f"Job error: {str(e)}"
                    logger.error(error_msg)
                    if attempt < job.max_retries:
                        execution.retry_count += 1
                        await asyncio.sleep(job.retry_delay)
                        continue
                    else:
                        execution.status = JobStatus.FAILED
                        execution.error = error_msg
                        break
            
            execution.end_time = datetime.now()
            execution.execution_time = (execution.end_time - execution.start_time).total_seconds()
            
        except Exception as e:
            execution.status = JobStatus.FAILED
            execution.error = str(e)
            execution.end_time = datetime.now()
            logger.error(f"Error executing job: {e}")
    
    def enable_job(self, job_id: str) -> bool:
        """Enable a job"""
        try:
            if job_id not in self.jobs:
                return False
            
            job = self.jobs[job_id]
            if not job.enabled:
                job.enabled = True
                self._schedule_job(job)
                logger.info(f"Enabled job: {job.name} ({job_id})")
            
            return True
            
        except Exception as e:
            logger.error(f"Error enabling job: {e}")
            return False
    
    def disable_job(self, job_id: str) -> bool:
        """Disable a job"""
        try:
            if job_id not in self.jobs:
                return False
            
            job = self.jobs[job_id]
            if job.enabled:
                job.enabled = False
                logger.info(f"Disabled job: {job.name} ({job_id})")
            
            return True
            
        except Exception as e:
            logger.error(f"Error disabling job: {e}")
            return False
    
    def trigger_job(self, job_id: str) -> str:
        """Manually trigger a job"""
        try:
            if job_id not in self.jobs:
                raise ValueError(f"Job not found: {job_id}")
            
            job = self.jobs[job_id]
            
            # Create immediate execution
            execution_id = str(uuid.uuid4())
            execution = JobExecution(
                id=execution_id,
                job_id=job_id,
                status=JobStatus.SCHEDULED,
                scheduled_time=datetime.now(pytz.timezone(job.timezone))
            )
            
            self.executions[execution_id] = execution
            
            # Execute immediately
            asyncio.create_task(self._execute_job(execution, job))
            
            logger.info(f"Triggered job: {job.name} ({job_id})")
            return execution_id
            
        except Exception as e:
            logger.error(f"Error triggering job: {e}")
            raise
    
    def get_job_executions(
        self,
        job_id: Optional[str] = None,
        status: Optional[JobStatus] = None,
        limit: int = 100
    ) -> List[JobExecution]:
        """Get job executions with filters"""
        try:
            executions = list(self.executions.values())
            
            if job_id:
                executions = [e for e in executions if e.job_id == job_id]
            
            if status:
                executions = [e for e in executions if e.status == status]
            
            # Sort by scheduled time (newest first)
            executions.sort(key=lambda x: x.scheduled_time, reverse=True)
            
            return executions[:limit]
            
        except Exception as e:
            logger.error(f"Error getting job executions: {e}")
            return []
    
    def get_scheduling_statistics(self) -> Dict[str, Any]:
        """Get scheduling statistics"""
        try:
            executions = list(self.executions.values())
            
            stats = {
                'total_jobs': len(self.jobs),
                'enabled_jobs': len([j for j in self.jobs.values() if j.enabled]),
                'disabled_jobs': len([j for j in self.jobs.values() if not j.enabled]),
                'total_executions': len(executions),
                'completed_executions': len([e for e in executions if e.status == JobStatus.COMPLETED]),
                'failed_executions': len([e for e in executions if e.status == JobStatus.FAILED]),
                'running_executions': len([e for e in executions if e.status == JobStatus.RUNNING]),
                'scheduled_executions': len([e for e in executions if e.status == JobStatus.SCHEDULED]),
                'average_execution_time': 0.0,
                'success_rate': 0.0
            }
            
            # Calculate average execution time
            completed_executions = [e for e in executions if e.status == JobStatus.COMPLETED and e.execution_time > 0]
            if completed_executions:
                stats['average_execution_time'] = sum(e.execution_time for e in completed_executions) / len(completed_executions)
            
            # Calculate success rate
            total_completed = len([e for e in executions if e.status in [JobStatus.COMPLETED, JobStatus.FAILED]])
            if total_completed > 0:
                stats['success_rate'] = len([e for e in executions if e.status == JobStatus.COMPLETED]) / total_completed
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting scheduling statistics: {e}")
            return {}
    
    def export_jobs(self, output_path: str) -> bool:
        """Export all jobs to file"""
        try:
            jobs_data = []
            for job in self.jobs.values():
                job_data = {
                    'id': job.id,
                    'name': job.name,
                    'description': job.description,
                    'function': job.function,
                    'parameters': job.parameters,
                    'schedule_type': job.schedule_type.value,
                    'schedule_config': job.schedule_config,
                    'timezone': job.timezone,
                    'enabled': job.enabled,
                    'max_retries': job.max_retries,
                    'retry_delay': job.retry_delay,
                    'timeout': job.timeout,
                    'priority': job.priority,
                    'tags': job.tags
                }
                jobs_data.append(job_data)
            
            with open(output_path, 'w') as f:
                json.dump(jobs_data, f, indent=2)
            
            logger.info(f"Exported {len(jobs_data)} jobs to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting jobs: {e}")
            return False
    
    def import_jobs(self, file_path: str) -> int:
        """Import jobs from file"""
        try:
            with open(file_path, 'r') as f:
                jobs_data = json.load(f)
            
            imported_count = 0
            for job_data in jobs_data:
                try:
                    job = JobDefinition(
                        id=job_data['id'],
                        name=job_data['name'],
                        description=job_data['description'],
                        function=job_data['function'],
                        parameters=job_data.get('parameters', {}),
                        schedule_type=ScheduleType(job_data['schedule_type']),
                        schedule_config=job_data.get('schedule_config', {}),
                        timezone=job_data.get('timezone', 'UTC'),
                        enabled=job_data.get('enabled', True),
                        max_retries=job_data.get('max_retries', 3),
                        retry_delay=job_data.get('retry_delay', 60),
                        timeout=job_data.get('timeout', 300),
                        priority=job_data.get('priority', 0),
                        tags=job_data.get('tags', [])
                    )
                    
                    self.create_job(job)
                    imported_count += 1
                    
                except Exception as e:
                    logger.error(f"Error importing job {job_data.get('name', 'unknown')}: {e}")
            
            logger.info(f"Imported {imported_count} jobs from {file_path}")
            return imported_count
            
        except Exception as e:
            logger.error(f"Error importing jobs: {e}")
            return 0
    
    def cleanup_old_executions(self, days: int = 30):
        """Cleanup old job executions"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            executions_to_remove = []
            
            for execution_id, execution in self.executions.items():
                if execution.scheduled_time < cutoff_date:
                    executions_to_remove.append(execution_id)
            
            for execution_id in executions_to_remove:
                del self.executions[execution_id]
            
            logger.info(f"Cleaned up {len(executions_to_remove)} old job executions")
            
        except Exception as e:
            logger.error(f"Error cleaning up old executions: {e}")

# Global scheduling service instance
scheduling_service = SchedulingService()

def register_scheduled_job(name: str, function: Callable):
    """Register scheduled job function"""
    scheduling_service.register_job_function(name, function)

def create_scheduled_job(definition: JobDefinition) -> str:
    """Create scheduled job using global service"""
    return scheduling_service.create_job(definition)

def enable_scheduled_job(job_id: str) -> bool:
    """Enable scheduled job using global service"""
    return scheduling_service.enable_job(job_id)

def disable_scheduled_job(job_id: str) -> bool:
    """Disable scheduled job using global service"""
    return scheduling_service.disable_job(job_id)

def trigger_scheduled_job(job_id: str) -> str:
    """Trigger scheduled job using global service"""
    return scheduling_service.trigger_job(job_id)

def get_job_executions(job_id: str = None, status: JobStatus = None, limit: int = 100) -> List[JobExecution]:
    """Get job executions using global service"""
    return scheduling_service.get_job_executions(job_id, status, limit)

def get_scheduling_statistics() -> Dict[str, Any]:
    """Get scheduling statistics using global service"""
    return scheduling_service.get_scheduling_statistics()

def export_scheduled_jobs(output_path: str) -> bool:
    """Export scheduled jobs using global service"""
    return scheduling_service.export_jobs(output_path)

def import_scheduled_jobs(file_path: str) -> int:
    """Import scheduled jobs using global service"""
    return scheduling_service.import_jobs(file_path)

























