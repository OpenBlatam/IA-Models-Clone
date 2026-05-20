"""
Workflow Scheduler - Advanced scheduling system for document workflow chains
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import uuid
from croniter import croniter
import pytz

logger = logging.getLogger(__name__)

class ScheduleType(Enum):
    """Types of schedule triggers"""
    ONCE = "once"
    INTERVAL = "interval"
    CRON = "cron"
    EVENT_BASED = "event_based"
    CONDITIONAL = "conditional"

class ScheduleStatus(Enum):
    """Status of scheduled workflows"""
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class ScheduleTrigger:
    """Represents a schedule trigger configuration"""
    trigger_id: str
    schedule_type: ScheduleType
    config: Dict[str, Any]
    timezone: str = "UTC"
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_triggered: Optional[datetime] = None
    next_trigger: Optional[datetime] = None

@dataclass
class WorkflowSchedule:
    """Represents a scheduled workflow"""
    schedule_id: str
    workflow_id: str
    trigger: ScheduleTrigger
    priority: int = 0
    max_retries: int = 3
    retry_delay: int = 300  # seconds
    timeout: int = 3600  # seconds
    status: ScheduleStatus = ScheduleStatus.ACTIVE
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    execution_history: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ScheduleExecution:
    """Represents a workflow execution instance"""
    execution_id: str
    schedule_id: str
    workflow_id: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    status: str = "running"
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    retry_count: int = 0

class WorkflowScheduler:
    """Advanced workflow scheduler with multiple trigger types and intelligent execution"""
    
    def __init__(self, workflow_engine=None):
        self.workflow_engine = workflow_engine
        self.schedules: Dict[str, WorkflowSchedule] = {}
        self.executions: Dict[str, ScheduleExecution] = {}
        self.running_tasks: Dict[str, asyncio.Task] = {}
        self.scheduler_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
        self._execution_lock = asyncio.Lock()
        
        # Performance metrics
        self.metrics = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "average_execution_time": 0.0,
            "active_schedules": 0,
            "queued_executions": 0
        }
        
        logger.info("WorkflowScheduler initialized")

    async def start(self):
        """Start the scheduler service"""
        if self.scheduler_task and not self.scheduler_task.done():
            logger.warning("Scheduler is already running")
            return
            
        self.scheduler_task = asyncio.create_task(self._scheduler_loop())
        logger.info("WorkflowScheduler started")

    async def stop(self):
        """Stop the scheduler service"""
        self._shutdown_event.set()
        
        if self.scheduler_task:
            await self.scheduler_task
            self.scheduler_task = None
            
        # Cancel all running tasks
        for task in self.running_tasks.values():
            if not task.done():
                task.cancel()
                
        await asyncio.gather(*self.running_tasks.values(), return_exceptions=True)
        self.running_tasks.clear()
        
        logger.info("WorkflowScheduler stopped")

    async def create_schedule(
        self,
        workflow_id: str,
        schedule_type: ScheduleType,
        config: Dict[str, Any],
        priority: int = 0,
        timezone: str = "UTC",
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a new workflow schedule"""
        schedule_id = str(uuid.uuid4())
        trigger_id = str(uuid.uuid4())
        
        trigger = ScheduleTrigger(
            trigger_id=trigger_id,
            schedule_type=schedule_type,
            config=config,
            timezone=timezone
        )
        
        # Calculate next trigger time
        trigger.next_trigger = await self._calculate_next_trigger(trigger)
        
        schedule = WorkflowSchedule(
            schedule_id=schedule_id,
            workflow_id=workflow_id,
            trigger=trigger,
            priority=priority,
            metadata=metadata or {}
        )
        
        self.schedules[schedule_id] = schedule
        self.metrics["active_schedules"] = len(self.schedules)
        
        logger.info(f"Created schedule {schedule_id} for workflow {workflow_id}")
        return schedule_id

    async def update_schedule(
        self,
        schedule_id: str,
        config: Optional[Dict[str, Any]] = None,
        priority: Optional[int] = None,
        status: Optional[ScheduleStatus] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Update an existing schedule"""
        if schedule_id not in self.schedules:
            return False
            
        schedule = self.schedules[schedule_id]
        
        if config is not None:
            schedule.trigger.config.update(config)
            schedule.trigger.next_trigger = await self._calculate_next_trigger(schedule.trigger)
            
        if priority is not None:
            schedule.priority = priority
            
        if status is not None:
            schedule.status = status
            
        if metadata is not None:
            schedule.metadata.update(metadata)
            
        schedule.updated_at = datetime.utcnow()
        
        logger.info(f"Updated schedule {schedule_id}")
        return True

    async def delete_schedule(self, schedule_id: str) -> bool:
        """Delete a schedule"""
        if schedule_id not in self.schedules:
            return False
            
        # Cancel any running executions for this schedule
        for execution_id, execution in list(self.executions.items()):
            if execution.schedule_id == schedule_id and execution.status == "running":
                await self._cancel_execution(execution_id)
                
        del self.schedules[schedule_id]
        self.metrics["active_schedules"] = len(self.schedules)
        
        logger.info(f"Deleted schedule {schedule_id}")
        return True

    async def pause_schedule(self, schedule_id: str) -> bool:
        """Pause a schedule"""
        return await self.update_schedule(schedule_id, status=ScheduleStatus.PAUSED)

    async def resume_schedule(self, schedule_id: str) -> bool:
        """Resume a paused schedule"""
        return await self.update_schedule(schedule_id, status=ScheduleStatus.ACTIVE)

    async def get_schedule(self, schedule_id: str) -> Optional[WorkflowSchedule]:
        """Get schedule details"""
        return self.schedules.get(schedule_id)

    async def list_schedules(
        self,
        workflow_id: Optional[str] = None,
        status: Optional[ScheduleStatus] = None,
        limit: int = 100
    ) -> List[WorkflowSchedule]:
        """List schedules with optional filtering"""
        schedules = list(self.schedules.values())
        
        if workflow_id:
            schedules = [s for s in schedules if s.workflow_id == workflow_id]
            
        if status:
            schedules = [s for s in schedules if s.status == status]
            
        # Sort by priority and creation time
        schedules.sort(key=lambda s: (-s.priority, s.created_at))
        
        return schedules[:limit]

    async def get_execution_history(
        self,
        schedule_id: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get execution history"""
        history = []
        
        for schedule in self.schedules.values():
            if schedule_id and schedule.schedule_id != schedule_id:
                continue
                
            history.extend(schedule.execution_history)
            
        # Sort by execution time
        history.sort(key=lambda h: h.get("started_at", ""), reverse=True)
        
        return history[:limit]

    async def get_metrics(self) -> Dict[str, Any]:
        """Get scheduler performance metrics"""
        return {
            **self.metrics,
            "uptime": datetime.utcnow() - (self.schedules[list(self.schedules.keys())[0]].created_at if self.schedules else datetime.utcnow()),
            "memory_usage": len(self.schedules) + len(self.executions),
            "active_executions": len([e for e in self.executions.values() if e.status == "running"])
        }

    async def _scheduler_loop(self):
        """Main scheduler loop"""
        logger.info("Scheduler loop started")
        
        while not self._shutdown_event.is_set():
            try:
                await self._check_triggers()
                await asyncio.sleep(1)  # Check every second
            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")
                await asyncio.sleep(5)  # Wait longer on error
                
        logger.info("Scheduler loop stopped")

    async def _check_triggers(self):
        """Check for triggers that need to be executed"""
        now = datetime.utcnow()
        
        for schedule in self.schedules.values():
            if schedule.status != ScheduleStatus.ACTIVE:
                continue
                
            if not schedule.trigger.enabled:
                continue
                
            if schedule.trigger.next_trigger and now >= schedule.trigger.next_trigger:
                await self._trigger_execution(schedule)
                schedule.trigger.last_triggered = now
                schedule.trigger.next_trigger = await self._calculate_next_trigger(schedule.trigger)

    async def _trigger_execution(self, schedule: WorkflowSchedule):
        """Trigger a workflow execution"""
        execution_id = str(uuid.uuid4())
        
        execution = ScheduleExecution(
            execution_id=execution_id,
            schedule_id=schedule.schedule_id,
            workflow_id=schedule.workflow_id,
            started_at=datetime.utcnow()
        )
        
        self.executions[execution_id] = execution
        self.metrics["total_executions"] += 1
        self.metrics["queued_executions"] += 1
        
        # Create and start execution task
        task = asyncio.create_task(
            self._execute_workflow(execution, schedule)
        )
        self.running_tasks[execution_id] = task
        
        logger.info(f"Triggered execution {execution_id} for schedule {schedule.schedule_id}")

    async def _execute_workflow(self, execution: ScheduleExecution, schedule: WorkflowSchedule):
        """Execute a workflow with retry logic"""
        try:
            # Execute the workflow
            if self.workflow_engine:
                result = await self.workflow_engine.execute_workflow(
                    execution.workflow_id,
                    timeout=schedule.timeout
                )
                execution.result = result
                execution.status = "completed"
                execution.completed_at = datetime.utcnow()
                self.metrics["successful_executions"] += 1
            else:
                raise Exception("No workflow engine available")
                
        except Exception as e:
            execution.error = str(e)
            execution.retry_count += 1
            
            if execution.retry_count < schedule.max_retries:
                # Schedule retry
                execution.status = "retrying"
                await asyncio.sleep(schedule.retry_delay)
                await self._execute_workflow(execution, schedule)
            else:
                execution.status = "failed"
                execution.completed_at = datetime.utcnow()
                self.metrics["failed_executions"] += 1
                
        finally:
            # Update execution history
            execution_data = {
                "execution_id": execution.execution_id,
                "started_at": execution.started_at.isoformat(),
                "completed_at": execution.completed_at.isoformat() if execution.completed_at else None,
                "status": execution.status,
                "retry_count": execution.retry_count,
                "error": execution.error
            }
            
            schedule.execution_history.append(execution_data)
            
            # Keep only last 100 executions
            if len(schedule.execution_history) > 100:
                schedule.execution_history = schedule.execution_history[-100:]
                
            # Clean up
            self.metrics["queued_executions"] = max(0, self.metrics["queued_executions"] - 1)
            if execution_id in self.running_tasks:
                del self.running_tasks[execution_id]

    async def _calculate_next_trigger(self, trigger: ScheduleTrigger) -> Optional[datetime]:
        """Calculate the next trigger time based on schedule type"""
        now = datetime.utcnow()
        tz = pytz.timezone(trigger.timezone)
        now_tz = now.replace(tzinfo=pytz.UTC).astimezone(tz)
        
        if trigger.schedule_type == ScheduleType.ONCE:
            # One-time execution
            if trigger.config.get("execute_at"):
                execute_at = datetime.fromisoformat(trigger.config["execute_at"])
                if execute_at > now:
                    return execute_at
            return None
            
        elif trigger.schedule_type == ScheduleType.INTERVAL:
            # Interval-based execution
            interval_seconds = trigger.config.get("interval_seconds", 3600)
            if trigger.last_triggered:
                return trigger.last_triggered + timedelta(seconds=interval_seconds)
            else:
                return now + timedelta(seconds=interval_seconds)
                
        elif trigger.schedule_type == ScheduleType.CRON:
            # Cron-based execution
            cron_expr = trigger.config.get("cron_expression", "0 * * * *")
            try:
                cron = croniter(cron_expr, now_tz)
                next_time = cron.get_next(datetime)
                return next_time.replace(tzinfo=None)
            except Exception as e:
                logger.error(f"Invalid cron expression: {cron_expr}, error: {e}")
                return None
                
        elif trigger.schedule_type == ScheduleType.EVENT_BASED:
            # Event-based execution (handled separately)
            return None
            
        elif trigger.schedule_type == ScheduleType.CONDITIONAL:
            # Conditional execution (handled separately)
            return None
            
        return None

    async def _cancel_execution(self, execution_id: str):
        """Cancel a running execution"""
        if execution_id in self.running_tasks:
            task = self.running_tasks[execution_id]
            if not task.done():
                task.cancel()
                
        if execution_id in self.executions:
            execution = self.executions[execution_id]
            execution.status = "cancelled"
            execution.completed_at = datetime.utcnow()

    async def trigger_event_based_execution(
        self,
        event_type: str,
        event_data: Dict[str, Any],
        workflow_id: Optional[str] = None
    ):
        """Trigger event-based workflow executions"""
        triggered_count = 0
        
        for schedule in self.schedules.values():
            if schedule.status != ScheduleStatus.ACTIVE:
                continue
                
            if schedule.trigger.schedule_type != ScheduleType.EVENT_BASED:
                continue
                
            if workflow_id and schedule.workflow_id != workflow_id:
                continue
                
            # Check if event matches trigger conditions
            trigger_events = schedule.trigger.config.get("events", [])
            if event_type in trigger_events:
                await self._trigger_execution(schedule)
                triggered_count += 1
                
        logger.info(f"Triggered {triggered_count} event-based executions for event: {event_type}")

    async def check_conditional_executions(self):
        """Check and trigger conditional executions"""
        triggered_count = 0
        
        for schedule in self.schedules.values():
            if schedule.status != ScheduleStatus.ACTIVE:
                continue
                
            if schedule.trigger.schedule_type != ScheduleType.CONDITIONAL:
                continue
                
            # Check condition
            condition = schedule.trigger.config.get("condition")
            if condition and await self._evaluate_condition(condition):
                await self._trigger_execution(schedule)
                triggered_count += 1
                
        if triggered_count > 0:
            logger.info(f"Triggered {triggered_count} conditional executions")

    async def _evaluate_condition(self, condition: Dict[str, Any]) -> bool:
        """Evaluate a conditional trigger condition"""
        # Simple condition evaluation - can be extended
        condition_type = condition.get("type")
        
        if condition_type == "time_based":
            # Check if current time matches condition
            current_hour = datetime.utcnow().hour
            target_hours = condition.get("hours", [])
            return current_hour in target_hours
            
        elif condition_type == "metric_based":
            # Check if metrics meet condition
            metric_name = condition.get("metric")
            threshold = condition.get("threshold")
            operator = condition.get("operator", ">")
            
            # This would integrate with actual metrics
            return True  # Placeholder
            
        elif condition_type == "external_api":
            # Check external API condition
            api_url = condition.get("api_url")
            expected_response = condition.get("expected_response")
            
            # This would make actual API call
            return True  # Placeholder
            
        return False

# Global scheduler instance
workflow_scheduler = WorkflowScheduler()

# Convenience functions
async def create_schedule(
    workflow_id: str,
    schedule_type: ScheduleType,
    config: Dict[str, Any],
    **kwargs
) -> str:
    """Create a new workflow schedule"""
    return await workflow_scheduler.create_schedule(
        workflow_id, schedule_type, config, **kwargs
    )

async def schedule_workflow_once(
    workflow_id: str,
    execute_at: datetime,
    **kwargs
) -> str:
    """Schedule a workflow to run once at a specific time"""
    config = {"execute_at": execute_at.isoformat()}
    return await create_schedule(
        workflow_id, ScheduleType.ONCE, config, **kwargs
    )

async def schedule_workflow_interval(
    workflow_id: str,
    interval_seconds: int,
    **kwargs
) -> str:
    """Schedule a workflow to run at regular intervals"""
    config = {"interval_seconds": interval_seconds}
    return await create_schedule(
        workflow_id, ScheduleType.INTERVAL, config, **kwargs
    )

async def schedule_workflow_cron(
    workflow_id: str,
    cron_expression: str,
    **kwargs
) -> str:
    """Schedule a workflow using cron expression"""
    config = {"cron_expression": cron_expression}
    return await create_schedule(
        workflow_id, ScheduleType.CRON, config, **kwargs
    )

async def schedule_workflow_event(
    workflow_id: str,
    events: List[str],
    **kwargs
) -> str:
    """Schedule a workflow to run on specific events"""
    config = {"events": events}
    return await create_schedule(
        workflow_id, ScheduleType.EVENT_BASED, config, **kwargs
    )

async def schedule_workflow_conditional(
    workflow_id: str,
    condition: Dict[str, Any],
    **kwargs
) -> str:
    """Schedule a workflow to run based on conditions"""
    config = {"condition": condition}
    return await create_schedule(
        workflow_id, ScheduleType.CONDITIONAL, config, **kwargs
    )
