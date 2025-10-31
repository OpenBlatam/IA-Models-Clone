"""
Automation Engine for OpusClip Improved
======================================

Advanced automation system for workflow orchestration and task scheduling.
"""

import asyncio
import logging
import json
from typing import Dict, Any, List, Optional, Callable, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
from uuid import UUID, uuid4
import croniter
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.date import DateTrigger

from .schemas import get_settings
from .exceptions import AutomationError, create_automation_error

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    """Task status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


class TaskPriority(str, Enum):
    """Task priority"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class TriggerType(str, Enum):
    """Trigger types"""
    CRON = "cron"
    INTERVAL = "interval"
    DATE = "date"
    EVENT = "event"
    MANUAL = "manual"


@dataclass
class TaskDefinition:
    """Task definition"""
    task_id: str
    name: str
    description: str
    function: str  # Function name to execute
    parameters: Dict[str, Any]
    trigger_type: TriggerType
    trigger_config: Dict[str, Any]
    priority: TaskPriority = TaskPriority.NORMAL
    max_retries: int = 3
    retry_delay: int = 60
    timeout: int = 300
    enabled: bool = True
    created_at: datetime = None
    updated_at: datetime = None


@dataclass
class TaskExecution:
    """Task execution record"""
    execution_id: str
    task_id: str
    status: TaskStatus
    started_at: datetime
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    retry_count: int = 0
    execution_time: Optional[float] = None


@dataclass
class WorkflowDefinition:
    """Workflow definition"""
    workflow_id: str
    name: str
    description: str
    steps: List[Dict[str, Any]]
    triggers: List[Dict[str, Any]]
    conditions: List[Dict[str, Any]]
    enabled: bool = True
    created_at: datetime = None
    updated_at: datetime = None


class TaskExecutor:
    """Task executor for running automation tasks"""
    
    def __init__(self):
        self.registered_functions: Dict[str, Callable] = {}
        self.running_tasks: Dict[str, asyncio.Task] = {}
        self.task_results: Dict[str, List[TaskExecution]] = {}
    
    def register_function(self, name: str, func: Callable):
        """Register a function for task execution"""
        self.registered_functions[name] = func
        logger.info(f"Registered function: {name}")
    
    async def execute_task(self, task_def: TaskDefinition) -> TaskExecution:
        """Execute a task"""
        execution_id = str(uuid4())
        execution = TaskExecution(
            execution_id=execution_id,
            task_id=task_def.task_id,
            status=TaskStatus.RUNNING,
            started_at=datetime.utcnow()
        )
        
        # Store execution record
        if task_def.task_id not in self.task_results:
            self.task_results[task_def.task_id] = []
        self.task_results[task_def.task_id].append(execution)
        
        try:
            # Get function
            if task_def.function not in self.registered_functions:
                raise ValueError(f"Function '{task_def.function}' not registered")
            
            func = self.registered_functions[task_def.function]
            
            # Execute function with timeout
            start_time = datetime.utcnow()
            result = await asyncio.wait_for(
                func(**task_def.parameters),
                timeout=task_def.timeout
            )
            
            # Update execution record
            execution.status = TaskStatus.COMPLETED
            execution.completed_at = datetime.utcnow()
            execution.result = {"data": result}
            execution.execution_time = (execution.completed_at - execution.started_at).total_seconds()
            
            logger.info(f"Task {task_def.task_id} completed successfully")
            
        except asyncio.TimeoutError:
            execution.status = TaskStatus.FAILED
            execution.completed_at = datetime.utcnow()
            execution.error = f"Task timed out after {task_def.timeout} seconds"
            execution.execution_time = (execution.completed_at - execution.started_at).total_seconds()
            
            logger.error(f"Task {task_def.task_id} timed out")
            
        except Exception as e:
            execution.status = TaskStatus.FAILED
            execution.completed_at = datetime.utcnow()
            execution.error = str(e)
            execution.execution_time = (execution.completed_at - execution.started_at).total_seconds()
            
            logger.error(f"Task {task_def.task_id} failed: {e}")
        
        return execution
    
    async def execute_task_with_retry(self, task_def: TaskDefinition) -> TaskExecution:
        """Execute task with retry logic"""
        last_execution = None
        
        for attempt in range(task_def.max_retries + 1):
            try:
                execution = await self.execute_task(task_def)
                
                if execution.status == TaskStatus.COMPLETED:
                    return execution
                
                # If failed and not last attempt, wait before retry
                if attempt < task_def.max_retries:
                    execution.status = TaskStatus.RETRYING
                    execution.retry_count = attempt + 1
                    
                    logger.info(f"Task {task_def.task_id} retrying in {task_def.retry_delay}s (attempt {attempt + 1})")
                    await asyncio.sleep(task_def.retry_delay)
                
                last_execution = execution
                
            except Exception as e:
                logger.error(f"Task {task_def.task_id} execution error: {e}")
                if attempt == task_def.max_retries:
                    raise
        
        return last_execution
    
    def get_task_history(self, task_id: str, limit: int = 100) -> List[TaskExecution]:
        """Get task execution history"""
        return self.task_results.get(task_id, [])[-limit:]
    
    def get_task_stats(self, task_id: str) -> Dict[str, Any]:
        """Get task statistics"""
        executions = self.task_results.get(task_id, [])
        
        if not executions:
            return {"total_executions": 0}
        
        total_executions = len(executions)
        successful_executions = len([e for e in executions if e.status == TaskStatus.COMPLETED])
        failed_executions = len([e for e in executions if e.status == TaskStatus.FAILED])
        
        success_rate = (successful_executions / total_executions * 100) if total_executions > 0 else 0
        
        # Calculate average execution time
        completed_executions = [e for e in executions if e.execution_time is not None]
        avg_execution_time = (
            sum(e.execution_time for e in completed_executions) / len(completed_executions)
            if completed_executions else 0
        )
        
        return {
            "total_executions": total_executions,
            "successful_executions": successful_executions,
            "failed_executions": failed_executions,
            "success_rate": round(success_rate, 2),
            "average_execution_time": round(avg_execution_time, 2),
            "last_execution": executions[-1].started_at if executions else None
        }


class WorkflowEngine:
    """Workflow engine for complex automation workflows"""
    
    def __init__(self, task_executor: TaskExecutor):
        self.task_executor = task_executor
        self.workflows: Dict[str, WorkflowDefinition] = {}
        self.workflow_executions: Dict[str, List[Dict[str, Any]]] = {}
    
    def register_workflow(self, workflow: WorkflowDefinition):
        """Register a workflow"""
        self.workflows[workflow.workflow_id] = workflow
        logger.info(f"Registered workflow: {workflow.workflow_id}")
    
    async def execute_workflow(self, workflow_id: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute a workflow"""
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow '{workflow_id}' not found")
        
        workflow = self.workflows[workflow_id]
        execution_id = str(uuid4())
        
        execution_context = {
            "execution_id": execution_id,
            "workflow_id": workflow_id,
            "context": context or {},
            "steps_completed": [],
            "steps_failed": [],
            "started_at": datetime.utcnow(),
            "status": "running"
        }
        
        # Store execution
        if workflow_id not in self.workflow_executions:
            self.workflow_executions[workflow_id] = []
        self.workflow_executions[workflow_id].append(execution_context)
        
        try:
            # Execute workflow steps
            for step in workflow.steps:
                step_result = await self._execute_workflow_step(step, execution_context)
                
                if step_result["status"] == "success":
                    execution_context["steps_completed"].append(step_result)
                else:
                    execution_context["steps_failed"].append(step_result)
                    
                    # Check if step is critical
                    if step.get("critical", False):
                        execution_context["status"] = "failed"
                        break
            
            # Set final status
            if execution_context["status"] == "running":
                execution_context["status"] = "completed"
            
            execution_context["completed_at"] = datetime.utcnow()
            
            logger.info(f"Workflow {workflow_id} execution {execution_id} completed with status: {execution_context['status']}")
            
        except Exception as e:
            execution_context["status"] = "failed"
            execution_context["error"] = str(e)
            execution_context["completed_at"] = datetime.utcnow()
            
            logger.error(f"Workflow {workflow_id} execution {execution_id} failed: {e}")
        
        return execution_context
    
    async def _execute_workflow_step(self, step: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a workflow step"""
        step_id = step.get("id", str(uuid4()))
        step_type = step.get("type", "task")
        
        try:
            if step_type == "task":
                # Execute task
                task_def = TaskDefinition(
                    task_id=step_id,
                    name=step.get("name", f"Step {step_id}"),
                    description=step.get("description", ""),
                    function=step["function"],
                    parameters=step.get("parameters", {}),
                    trigger_type=TriggerType.MANUAL,
                    trigger_config={}
                )
                
                execution = await self.task_executor.execute_task(task_def)
                
                return {
                    "step_id": step_id,
                    "status": "success" if execution.status == TaskStatus.COMPLETED else "failed",
                    "result": execution.result,
                    "error": execution.error,
                    "execution_time": execution.execution_time
                }
            
            elif step_type == "condition":
                # Evaluate condition
                condition = step.get("condition", "")
                result = self._evaluate_condition(condition, context)
                
                return {
                    "step_id": step_id,
                    "status": "success" if result else "skipped",
                    "result": {"condition_result": result}
                }
            
            elif step_type == "delay":
                # Wait for specified time
                delay = step.get("delay", 0)
                await asyncio.sleep(delay)
                
                return {
                    "step_id": step_id,
                    "status": "success",
                    "result": {"delay": delay}
                }
            
            else:
                raise ValueError(f"Unknown step type: {step_type}")
                
        except Exception as e:
            return {
                "step_id": step_id,
                "status": "failed",
                "error": str(e)
            }
    
    def _evaluate_condition(self, condition: str, context: Dict[str, Any]) -> bool:
        """Evaluate workflow condition"""
        try:
            # Simple condition evaluation - in production, use a proper expression evaluator
            # This is a placeholder implementation
            return True
        except Exception:
            return False
    
    def get_workflow_execution_history(self, workflow_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get workflow execution history"""
        return self.workflow_executions.get(workflow_id, [])[-limit:]


class AutomationScheduler:
    """Automation scheduler for managing scheduled tasks"""
    
    def __init__(self, task_executor: TaskExecutor):
        self.task_executor = task_executor
        self.scheduler = AsyncIOScheduler()
        self.scheduled_tasks: Dict[str, TaskDefinition] = {}
        self.task_jobs: Dict[str, str] = {}  # task_id -> job_id mapping
    
    async def start(self):
        """Start the scheduler"""
        self.scheduler.start()
        logger.info("Automation scheduler started")
    
    async def stop(self):
        """Stop the scheduler"""
        self.scheduler.shutdown()
        logger.info("Automation scheduler stopped")
    
    def schedule_task(self, task_def: TaskDefinition) -> str:
        """Schedule a task"""
        if task_def.trigger_type == TriggerType.CRON:
            trigger = CronTrigger.from_crontab(task_def.trigger_config["cron"])
        elif task_def.trigger_type == TriggerType.INTERVAL:
            trigger = IntervalTrigger(
                seconds=task_def.trigger_config.get("seconds", 0),
                minutes=task_def.trigger_config.get("minutes", 0),
                hours=task_def.trigger_config.get("hours", 0),
                days=task_def.trigger_config.get("days", 0)
            )
        elif task_def.trigger_type == TriggerType.DATE:
            trigger = DateTrigger(run_date=task_def.trigger_config["run_date"])
        else:
            raise ValueError(f"Unsupported trigger type: {task_def.trigger_type}")
        
        # Schedule job
        job = self.scheduler.add_job(
            func=self._execute_scheduled_task,
            trigger=trigger,
            args=[task_def],
            id=f"task_{task_def.task_id}",
            name=task_def.name,
            max_instances=1,
            replace_existing=True
        )
        
        # Store task and job mapping
        self.scheduled_tasks[task_def.task_id] = task_def
        self.task_jobs[task_def.task_id] = job.id
        
        logger.info(f"Scheduled task: {task_def.task_id}")
        return job.id
    
    def unschedule_task(self, task_id: str):
        """Unschedule a task"""
        if task_id in self.task_jobs:
            job_id = self.task_jobs[task_id]
            self.scheduler.remove_job(job_id)
            
            del self.scheduled_tasks[task_id]
            del self.task_jobs[task_id]
            
            logger.info(f"Unscheduled task: {task_id}")
    
    async def _execute_scheduled_task(self, task_def: TaskDefinition):
        """Execute a scheduled task"""
        try:
            await self.task_executor.execute_task_with_retry(task_def)
        except Exception as e:
            logger.error(f"Scheduled task {task_def.task_id} execution failed: {e}")
    
    def get_scheduled_tasks(self) -> List[TaskDefinition]:
        """Get all scheduled tasks"""
        return list(self.scheduled_tasks.values())
    
    def get_next_run_time(self, task_id: str) -> Optional[datetime]:
        """Get next run time for a scheduled task"""
        if task_id in self.task_jobs:
            job_id = self.task_jobs[task_id]
            job = self.scheduler.get_job(job_id)
            return job.next_run_time if job else None
        return None


class EventDrivenAutomation:
    """Event-driven automation system"""
    
    def __init__(self, task_executor: TaskExecutor):
        self.task_executor = task_executor
        self.event_handlers: Dict[str, List[TaskDefinition]] = {}
        self.event_queue = asyncio.Queue()
        self.event_worker_running = False
    
    async def start(self):
        """Start event processing"""
        if not self.event_worker_running:
            self.event_worker_running = True
            asyncio.create_task(self._event_worker())
            logger.info("Event-driven automation started")
    
    async def stop(self):
        """Stop event processing"""
        self.event_worker_running = False
        logger.info("Event-driven automation stopped")
    
    def register_event_handler(self, event_type: str, task_def: TaskDefinition):
        """Register an event handler"""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        
        self.event_handlers[event_type].append(task_def)
        logger.info(f"Registered event handler for {event_type}: {task_def.task_id}")
    
    async def trigger_event(self, event_type: str, event_data: Dict[str, Any]):
        """Trigger an event"""
        event = {
            "event_type": event_type,
            "event_data": event_data,
            "timestamp": datetime.utcnow()
        }
        
        await self.event_queue.put(event)
        logger.info(f"Triggered event: {event_type}")
    
    async def _event_worker(self):
        """Event processing worker"""
        while self.event_worker_running:
            try:
                # Get event from queue
                event = await asyncio.wait_for(self.event_queue.get(), timeout=1.0)
                
                # Process event
                await self._process_event(event)
                
                # Mark task as done
                self.event_queue.task_done()
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Event worker error: {e}")
    
    async def _process_event(self, event: Dict[str, Any]):
        """Process an event"""
        event_type = event["event_type"]
        event_data = event["event_data"]
        
        if event_type not in self.event_handlers:
            return
        
        # Execute all handlers for this event type
        for task_def in self.event_handlers[event_type]:
            try:
                # Update task parameters with event data
                updated_task = TaskDefinition(
                    task_id=task_def.task_id,
                    name=task_def.name,
                    description=task_def.description,
                    function=task_def.function,
                    parameters={**task_def.parameters, "event_data": event_data},
                    trigger_type=task_def.trigger_type,
                    trigger_config=task_def.trigger_config,
                    priority=task_def.priority,
                    max_retries=task_def.max_retries,
                    retry_delay=task_def.retry_delay,
                    timeout=task_def.timeout,
                    enabled=task_def.enabled
                )
                
                await self.task_executor.execute_task_with_retry(updated_task)
                
            except Exception as e:
                logger.error(f"Event handler {task_def.task_id} failed: {e}")


class AutomationManager:
    """Main automation manager"""
    
    def __init__(self):
        self.settings = get_settings()
        self.task_executor = TaskExecutor()
        self.workflow_engine = WorkflowEngine(self.task_executor)
        self.scheduler = AutomationScheduler(self.task_executor)
        self.event_automation = EventDrivenAutomation(self.task_executor)
        
        self._register_default_functions()
        self._register_default_workflows()
    
    def _register_default_functions(self):
        """Register default automation functions"""
        # Video processing functions
        self.task_executor.register_function("process_video", self._process_video)
        self.task_executor.register_function("generate_clips", self._generate_clips)
        self.task_executor.register_function("export_clips", self._export_clips)
        self.task_executor.register_function("cleanup_temp_files", self._cleanup_temp_files)
        
        # Analytics functions
        self.task_executor.register_function("generate_analytics", self._generate_analytics)
        self.task_executor.register_function("send_notifications", self._send_notifications)
        
        # System functions
        self.task_executor.register_function("backup_database", self._backup_database)
        self.task_executor.register_function("health_check", self._health_check)
    
    def _register_default_workflows(self):
        """Register default workflows"""
        # Daily cleanup workflow
        cleanup_workflow = WorkflowDefinition(
            workflow_id="daily_cleanup",
            name="Daily Cleanup",
            description="Daily cleanup of temporary files and old data",
            steps=[
                {
                    "id": "cleanup_temp_files",
                    "type": "task",
                    "name": "Cleanup Temporary Files",
                    "function": "cleanup_temp_files",
                    "parameters": {"max_age_hours": 24}
                },
                {
                    "id": "cleanup_old_analytics",
                    "type": "task",
                    "name": "Cleanup Old Analytics",
                    "function": "cleanup_temp_files",
                    "parameters": {"max_age_days": 30}
                }
            ],
            triggers=[],
            conditions=[]
        )
        
        self.workflow_engine.register_workflow(cleanup_workflow)
    
    async def start(self):
        """Start automation system"""
        await self.scheduler.start()
        await self.event_automation.start()
        logger.info("Automation system started")
    
    async def stop(self):
        """Stop automation system"""
        await self.scheduler.stop()
        await self.event_automation.stop()
        logger.info("Automation system stopped")
    
    # Default function implementations
    async def _process_video(self, video_url: str, **kwargs) -> Dict[str, Any]:
        """Process video function"""
        # Placeholder implementation
        return {"status": "processed", "video_url": video_url}
    
    async def _generate_clips(self, analysis_id: str, **kwargs) -> Dict[str, Any]:
        """Generate clips function"""
        # Placeholder implementation
        return {"status": "generated", "analysis_id": analysis_id}
    
    async def _export_clips(self, generation_id: str, **kwargs) -> Dict[str, Any]:
        """Export clips function"""
        # Placeholder implementation
        return {"status": "exported", "generation_id": generation_id}
    
    async def _cleanup_temp_files(self, max_age_hours: int = 24, **kwargs) -> Dict[str, Any]:
        """Cleanup temporary files function"""
        # Placeholder implementation
        return {"status": "cleaned", "max_age_hours": max_age_hours}
    
    async def _generate_analytics(self, **kwargs) -> Dict[str, Any]:
        """Generate analytics function"""
        # Placeholder implementation
        return {"status": "generated"}
    
    async def _send_notifications(self, message: str, **kwargs) -> Dict[str, Any]:
        """Send notifications function"""
        # Placeholder implementation
        return {"status": "sent", "message": message}
    
    async def _backup_database(self, **kwargs) -> Dict[str, Any]:
        """Backup database function"""
        # Placeholder implementation
        return {"status": "backed_up"}
    
    async def _health_check(self, **kwargs) -> Dict[str, Any]:
        """Health check function"""
        # Placeholder implementation
        return {"status": "healthy"}
    
    def get_automation_stats(self) -> Dict[str, Any]:
        """Get automation system statistics"""
        return {
            "registered_functions": len(self.task_executor.registered_functions),
            "scheduled_tasks": len(self.scheduler.scheduled_tasks),
            "registered_workflows": len(self.workflow_engine.workflows),
            "event_handlers": sum(len(handlers) for handlers in self.event_automation.event_handlers.values()),
            "queue_size": self.event_automation.event_queue.qsize()
        }


# Global automation manager
automation_manager = AutomationManager()





























