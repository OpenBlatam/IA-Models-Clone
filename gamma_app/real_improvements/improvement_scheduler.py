"""
Gamma App - Real Improvement Scheduler
Automated scheduling for real improvements that actually work
"""

import asyncio
import logging
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import uuid
from pathlib import Path
import schedule
import threading

logger = logging.getLogger(__name__)

class ScheduleType(Enum):
    """Schedule types"""
    ONCE = "once"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    CUSTOM = "custom"

class ScheduleStatus(Enum):
    """Schedule status"""
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class ScheduledTask:
    """Scheduled task"""
    task_id: str
    name: str
    description: str
    schedule_type: ScheduleType
    schedule_time: str  # cron expression or datetime
    improvement_id: str
    workflow_id: Optional[str] = None
    status: ScheduleStatus = ScheduleStatus.ACTIVE
    created_at: datetime = None
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    run_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    max_retries: int = 3
    retry_count: int = 0
    enabled: bool = True

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()

@dataclass
class ScheduleExecution:
    """Schedule execution record"""
    execution_id: str
    task_id: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    duration: float = 0.0
    success: bool = False
    error_message: str = ""
    output_data: Dict[str, Any] = None

    def __post_init__(self):
        if self.output_data is None:
            self.output_data = {}

class RealImprovementScheduler:
    """
    Automated scheduler for real improvements
    """
    
    def __init__(self, project_root: str = "."):
        """Initialize improvement scheduler"""
        self.project_root = Path(project_root)
        self.scheduled_tasks: Dict[str, ScheduledTask] = {}
        self.executions: Dict[str, ScheduleExecution] = {}
        self.execution_logs: Dict[str, List[Dict[str, Any]]] = {}
        self.scheduler_thread: Optional[threading.Thread] = None
        self.running: bool = False
        
        # Initialize with default schedules
        self._initialize_default_schedules()
        
        logger.info(f"Real Improvement Scheduler initialized for {self.project_root}")
    
    def _initialize_default_schedules(self):
        """Initialize default schedules"""
        # Daily code quality check
        daily_quality_task = ScheduledTask(
            task_id="daily_quality_check",
            name="Daily Code Quality Check",
            description="Daily automated code quality analysis and improvements",
            schedule_type=ScheduleType.DAILY,
            schedule_time="09:00",
            improvement_id="code_quality_analysis"
        )
        self.scheduled_tasks[daily_quality_task.task_id] = daily_quality_task
        
        # Weekly security scan
        weekly_security_task = ScheduledTask(
            task_id="weekly_security_scan",
            name="Weekly Security Scan",
            description="Weekly security vulnerability scan and fixes",
            schedule_type=ScheduleType.WEEKLY,
            schedule_time="monday 10:00",
            improvement_id="security_scan"
        )
        self.scheduled_tasks[weekly_security_task.task_id] = weekly_security_task
        
        # Monthly performance optimization
        monthly_performance_task = ScheduledTask(
            task_id="monthly_performance_optimization",
            name="Monthly Performance Optimization",
            description="Monthly performance analysis and optimization",
            schedule_type=ScheduleType.MONTHLY,
            schedule_time="1 14:00",
            improvement_id="performance_optimization"
        )
        self.scheduled_tasks[monthly_performance_task.task_id] = monthly_performance_task
    
    def create_scheduled_task(self, name: str, description: str, schedule_type: ScheduleType,
                            schedule_time: str, improvement_id: str, workflow_id: Optional[str] = None,
                            max_retries: int = 3) -> str:
        """Create scheduled task"""
        try:
            task_id = f"task_{int(time.time() * 1000)}"
            
            task = ScheduledTask(
                task_id=task_id,
                name=name,
                description=description,
                schedule_type=schedule_type,
                schedule_time=schedule_time,
                improvement_id=improvement_id,
                workflow_id=workflow_id,
                max_retries=max_retries
            )
            
            # Calculate next run time
            task.next_run = self._calculate_next_run(task)
            
            self.scheduled_tasks[task_id] = task
            
            logger.info(f"Scheduled task created: {name}")
            return task_id
            
        except Exception as e:
            logger.error(f"Failed to create scheduled task: {e}")
            raise
    
    def _calculate_next_run(self, task: ScheduledTask) -> datetime:
        """Calculate next run time for task"""
        try:
            now = datetime.utcnow()
            
            if task.schedule_type == ScheduleType.ONCE:
                # Parse datetime string
                return datetime.fromisoformat(task.schedule_time)
            
            elif task.schedule_type == ScheduleType.DAILY:
                # Parse time string (HH:MM)
                hour, minute = map(int, task.schedule_time.split(':'))
                next_run = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
                if next_run <= now:
                    next_run += timedelta(days=1)
                return next_run
            
            elif task.schedule_type == ScheduleType.WEEKLY:
                # Parse day and time (e.g., "monday 10:00")
                day_name, time_str = task.schedule_time.split()
                hour, minute = map(int, time_str.split(':'))
                
                # Map day names to numbers
                day_map = {
                    'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3,
                    'friday': 4, 'saturday': 5, 'sunday': 6
                }
                target_day = day_map[day_name.lower()]
                
                # Calculate next occurrence
                days_ahead = target_day - now.weekday()
                if days_ahead <= 0:  # Target day already happened this week
                    days_ahead += 7
                
                next_run = now + timedelta(days=days_ahead)
                next_run = next_run.replace(hour=hour, minute=minute, second=0, microsecond=0)
                return next_run
            
            elif task.schedule_type == ScheduleType.MONTHLY:
                # Parse day and time (e.g., "1 14:00")
                day, time_str = task.schedule_time.split()
                hour, minute = map(int, time_str.split(':'))
                day = int(day)
                
                # Calculate next month
                if now.month == 12:
                    next_month = now.replace(year=now.year + 1, month=1, day=day, hour=hour, minute=minute, second=0, microsecond=0)
                else:
                    next_month = now.replace(month=now.month + 1, day=day, hour=hour, minute=minute, second=0, microsecond=0)
                
                return next_month
            
            else:  # CUSTOM
                # For custom schedules, assume it's a cron-like expression
                # This is simplified - in production, use a proper cron parser
                return now + timedelta(hours=1)
            
        except Exception as e:
            logger.error(f"Failed to calculate next run time: {e}")
            return datetime.utcnow() + timedelta(hours=1)
    
    async def start_scheduler(self):
        """Start the scheduler"""
        try:
            if self.running:
                logger.warning("Scheduler is already running")
                return
            
            self.running = True
            self.scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
            self.scheduler_thread.start()
            
            logger.info("Improvement scheduler started")
            
        except Exception as e:
            logger.error(f"Failed to start scheduler: {e}")
            self.running = False
    
    def stop_scheduler(self):
        """Stop the scheduler"""
        try:
            self.running = False
            if self.scheduler_thread:
                self.scheduler_thread.join(timeout=5)
            
            logger.info("Improvement scheduler stopped")
            
        except Exception as e:
            logger.error(f"Failed to stop scheduler: {e}")
    
    def _scheduler_loop(self):
        """Main scheduler loop"""
        while self.running:
            try:
                now = datetime.utcnow()
                
                # Check for tasks that need to run
                for task in self.scheduled_tasks.values():
                    if (task.enabled and 
                        task.status == ScheduleStatus.ACTIVE and 
                        task.next_run and 
                        now >= task.next_run):
                        
                        # Run task in background
                        asyncio.create_task(self._execute_scheduled_task(task))
                
                # Sleep for 1 minute before next check
                time.sleep(60)
                
            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")
                time.sleep(60)
    
    async def _execute_scheduled_task(self, task: ScheduledTask):
        """Execute scheduled task"""
        try:
            execution_id = f"exec_{int(time.time() * 1000)}"
            
            execution = ScheduleExecution(
                execution_id=execution_id,
                task_id=task.task_id,
                started_at=datetime.utcnow()
            )
            
            self.executions[execution_id] = execution
            self.execution_logs[execution_id] = []
            
            self._log_execution(execution_id, "started", f"Task {task.name} started")
            
            # Execute the improvement
            if task.workflow_id:
                # Execute workflow
                from improvement_workflow import get_improvement_workflow
                workflow = get_improvement_workflow()
                result = await workflow.execute_workflow(task.workflow_id, dry_run=False)
            else:
                # Execute single improvement
                from improvement_executor import get_improvement_executor
                executor = get_improvement_executor()
                result = await executor.execute_improvement(
                    improvement_id=task.improvement_id,
                    execution_type="automated",
                    dry_run=False
                )
            
            # Update execution
            execution.completed_at = datetime.utcnow()
            execution.duration = (execution.completed_at - execution.started_at).total_seconds()
            execution.success = result.get("success", False)
            
            if execution.success:
                execution.output_data = result
                task.success_count += 1
                self._log_execution(execution_id, "completed", f"Task {task.name} completed successfully")
            else:
                execution.error_message = result.get("error", "Unknown error")
                task.failure_count += 1
                task.retry_count += 1
                self._log_execution(execution_id, "failed", f"Task {task.name} failed: {execution.error_message}")
                
                # Handle retries
                if task.retry_count < task.max_retries:
                    # Schedule retry
                    task.next_run = datetime.utcnow() + timedelta(minutes=5)
                    self._log_execution(execution_id, "retry_scheduled", f"Retry scheduled for task {task.name}")
                else:
                    task.status = ScheduleStatus.FAILED
                    self._log_execution(execution_id, "max_retries_reached", f"Max retries reached for task {task.name}")
            
            # Update task
            task.last_run = execution.started_at
            task.run_count += 1
            
            # Calculate next run time
            if task.status == ScheduleStatus.ACTIVE:
                task.next_run = self._calculate_next_run(task)
            
        except Exception as e:
            logger.error(f"Failed to execute scheduled task: {e}")
            if execution_id in self.executions:
                self.executions[execution_id].success = False
                self.executions[execution_id].error_message = str(e)
                self._log_execution(execution_id, "error", f"Task execution error: {e}")
    
    def _log_execution(self, execution_id: str, event: str, message: str):
        """Log execution event"""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "event": event,
            "message": message
        }
        
        if execution_id not in self.execution_logs:
            self.execution_logs[execution_id] = []
        
        self.execution_logs[execution_id].append(log_entry)
        
        logger.info(f"Execution {execution_id}: {event} - {message}")
    
    def pause_task(self, task_id: str) -> bool:
        """Pause scheduled task"""
        try:
            if task_id in self.scheduled_tasks:
                task = self.scheduled_tasks[task_id]
                task.status = ScheduleStatus.PAUSED
                task.enabled = False
                logger.info(f"Task paused: {task.name}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Failed to pause task: {e}")
            return False
    
    def resume_task(self, task_id: str) -> bool:
        """Resume scheduled task"""
        try:
            if task_id in self.scheduled_tasks:
                task = self.scheduled_tasks[task_id]
                task.status = ScheduleStatus.ACTIVE
                task.enabled = True
                task.next_run = self._calculate_next_run(task)
                logger.info(f"Task resumed: {task.name}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Failed to resume task: {e}")
            return False
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel scheduled task"""
        try:
            if task_id in self.scheduled_tasks:
                task = self.scheduled_tasks[task_id]
                task.status = ScheduleStatus.CANCELLED
                task.enabled = False
                logger.info(f"Task cancelled: {task.name}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Failed to cancel task: {e}")
            return False
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task status"""
        if task_id not in self.scheduled_tasks:
            return None
        
        task = self.scheduled_tasks[task_id]
        
        return {
            "task_id": task_id,
            "name": task.name,
            "status": task.status.value,
            "enabled": task.enabled,
            "schedule_type": task.schedule_type.value,
            "schedule_time": task.schedule_time,
            "last_run": task.last_run.isoformat() if task.last_run else None,
            "next_run": task.next_run.isoformat() if task.next_run else None,
            "run_count": task.run_count,
            "success_count": task.success_count,
            "failure_count": task.failure_count,
            "retry_count": task.retry_count,
            "max_retries": task.max_retries
        }
    
    def get_scheduler_summary(self) -> Dict[str, Any]:
        """Get scheduler summary"""
        total_tasks = len(self.scheduled_tasks)
        active_tasks = len([t for t in self.scheduled_tasks.values() if t.status == ScheduleStatus.ACTIVE])
        paused_tasks = len([t for t in self.scheduled_tasks.values() if t.status == ScheduleStatus.PAUSED])
        completed_tasks = len([t for t in self.scheduled_tasks.values() if t.status == ScheduleStatus.COMPLETED])
        failed_tasks = len([t for t in self.scheduled_tasks.values() if t.status == ScheduleStatus.FAILED])
        
        total_executions = len(self.executions)
        successful_executions = len([e for e in self.executions.values() if e.success])
        failed_executions = len([e for e in self.executions.values() if not e.success])
        
        return {
            "scheduler_running": self.running,
            "total_tasks": total_tasks,
            "active_tasks": active_tasks,
            "paused_tasks": paused_tasks,
            "completed_tasks": completed_tasks,
            "failed_tasks": failed_tasks,
            "total_executions": total_executions,
            "successful_executions": successful_executions,
            "failed_executions": failed_executions,
            "success_rate": (successful_executions / total_executions * 100) if total_executions > 0 else 0
        }
    
    def get_upcoming_tasks(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get upcoming tasks in the next N hours"""
        try:
            cutoff_time = datetime.utcnow() + timedelta(hours=hours)
            upcoming_tasks = []
            
            for task in self.scheduled_tasks.values():
                if (task.enabled and 
                    task.status == ScheduleStatus.ACTIVE and 
                    task.next_run and 
                    task.next_run <= cutoff_time):
                    
                    upcoming_tasks.append({
                        "task_id": task.task_id,
                        "name": task.name,
                        "next_run": task.next_run.isoformat(),
                        "schedule_type": task.schedule_type.value,
                        "improvement_id": task.improvement_id
                    })
            
            # Sort by next run time
            upcoming_tasks.sort(key=lambda x: x["next_run"])
            return upcoming_tasks
            
        except Exception as e:
            logger.error(f"Failed to get upcoming tasks: {e}")
            return []
    
    def get_execution_logs(self, execution_id: str) -> List[Dict[str, Any]]:
        """Get execution logs"""
        return self.execution_logs.get(execution_id, [])
    
    def get_task_executions(self, task_id: str) -> List[Dict[str, Any]]:
        """Get executions for a specific task"""
        try:
            task_executions = []
            for execution in self.executions.values():
                if execution.task_id == task_id:
                    task_executions.append({
                        "execution_id": execution.execution_id,
                        "started_at": execution.started_at.isoformat(),
                        "completed_at": execution.completed_at.isoformat() if execution.completed_at else None,
                        "duration": execution.duration,
                        "success": execution.success,
                        "error_message": execution.error_message
                    })
            
            # Sort by start time (newest first)
            task_executions.sort(key=lambda x: x["started_at"], reverse=True)
            return task_executions
            
        except Exception as e:
            logger.error(f"Failed to get task executions: {e}")
            return []

# Global scheduler instance
improvement_scheduler = None

def get_improvement_scheduler() -> RealImprovementScheduler:
    """Get improvement scheduler instance"""
    global improvement_scheduler
    if not improvement_scheduler:
        improvement_scheduler = RealImprovementScheduler()
    return improvement_scheduler













