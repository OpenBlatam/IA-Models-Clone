"""
Background Tasks Service
========================

Service for managing background tasks and scheduled jobs.
"""

from __future__ import annotations
import asyncio
import logging
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import uuid

from ..utils.helpers import DateTimeHelpers
from ..utils.decorators import retry, log_execution


logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    """Task status enumeration"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskPriority(int, Enum):
    """Task priority enumeration"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class Task:
    """Background task representation"""
    id: str
    name: str
    func: Callable
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)
    status: TaskStatus = TaskStatus.PENDING
    priority: TaskPriority = TaskPriority.NORMAL
    created_at: datetime = field(default_factory=DateTimeHelpers.now_utc)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Any = None
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    scheduled_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScheduledTask:
    """Scheduled task representation"""
    id: str
    name: str
    func: Callable
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)
    schedule: str  # Cron-like schedule or interval
    next_run: datetime = field(default_factory=DateTimeHelpers.now_utc)
    last_run: Optional[datetime] = None
    is_active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


class BackgroundTaskService:
    """Background task service"""
    
    def __init__(self, max_workers: int = 10):
        self.max_workers = max_workers
        self.tasks: Dict[str, Task] = {}
        self.scheduled_tasks: Dict[str, ScheduledTask] = {}
        self.running_tasks: Dict[str, asyncio.Task] = {}
        self.task_queue: asyncio.Queue = asyncio.Queue()
        self.scheduler_task: Optional[asyncio.Task] = None
        self.worker_tasks: List[asyncio.Task] = []
        self.is_running = False
    
    async def start(self):
        """Start the background task service"""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Start worker tasks
        for i in range(self.max_workers):
            worker_task = asyncio.create_task(self._worker(f"worker-{i}"))
            self.worker_tasks.append(worker_task)
        
        # Start scheduler
        self.scheduler_task = asyncio.create_task(self._scheduler())
        
        logger.info(f"Background task service started with {self.max_workers} workers")
    
    async def stop(self):
        """Stop the background task service"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Cancel scheduler
        if self.scheduler_task:
            self.scheduler_task.cancel()
            try:
                await self.scheduler_task
            except asyncio.CancelledError:
                pass
        
        # Cancel worker tasks
        for worker_task in self.worker_tasks:
            worker_task.cancel()
        
        # Wait for workers to finish
        if self.worker_tasks:
            await asyncio.gather(*self.worker_tasks, return_exceptions=True)
        
        # Cancel running tasks
        for task_id, asyncio_task in self.running_tasks.items():
            asyncio_task.cancel()
        
        logger.info("Background task service stopped")
    
    async def submit_task(
        self,
        name: str,
        func: Callable,
        *args,
        priority: TaskPriority = TaskPriority.NORMAL,
        max_retries: int = 3,
        scheduled_at: Optional[datetime] = None,
        **kwargs
    ) -> str:
        """Submit a background task"""
        task_id = str(uuid.uuid4())
        
        task = Task(
            id=task_id,
            name=name,
            func=func,
            args=args,
            kwargs=kwargs,
            priority=priority,
            max_retries=max_retries,
            scheduled_at=scheduled_at
        )
        
        self.tasks[task_id] = task
        
        if scheduled_at:
            # Task is scheduled for later
            logger.info(f"Task {task_id} scheduled for {scheduled_at}")
        else:
            # Task is ready to run
            await self.task_queue.put(task)
            logger.info(f"Task {task_id} submitted to queue")
        
        return task_id
    
    async def schedule_task(
        self,
        name: str,
        func: Callable,
        schedule: str,
        *args,
        **kwargs
    ) -> str:
        """Schedule a recurring task"""
        task_id = str(uuid.uuid4())
        
        scheduled_task = ScheduledTask(
            id=task_id,
            name=name,
            func=func,
            args=args,
            kwargs=kwargs,
            schedule=schedule,
            next_run=self._calculate_next_run(schedule)
        )
        
        self.scheduled_tasks[task_id] = scheduled_task
        
        logger.info(f"Scheduled task {task_id} created with schedule: {schedule}")
        return task_id
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a task"""
        if task_id in self.tasks:
            task = self.tasks[task_id]
            if task.status == TaskStatus.PENDING:
                task.status = TaskStatus.CANCELLED
                logger.info(f"Task {task_id} cancelled")
                return True
            elif task.status == TaskStatus.RUNNING:
                # Cancel the running asyncio task
                if task_id in self.running_tasks:
                    self.running_tasks[task_id].cancel()
                task.status = TaskStatus.CANCELLED
                logger.info(f"Running task {task_id} cancelled")
                return True
        
        if task_id in self.scheduled_tasks:
            scheduled_task = self.scheduled_tasks[task_id]
            scheduled_task.is_active = False
            logger.info(f"Scheduled task {task_id} deactivated")
            return True
        
        return False
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task status"""
        if task_id in self.tasks:
            task = self.tasks[task_id]
            return {
                "id": task.id,
                "name": task.name,
                "status": task.status.value,
                "priority": task.priority.value,
                "created_at": task.created_at.isoformat(),
                "started_at": task.started_at.isoformat() if task.started_at else None,
                "completed_at": task.completed_at.isoformat() if task.completed_at else None,
                "retry_count": task.retry_count,
                "max_retries": task.max_retries,
                "error": task.error,
                "metadata": task.metadata
            }
        
        if task_id in self.scheduled_tasks:
            scheduled_task = self.scheduled_tasks[task_id]
            return {
                "id": scheduled_task.id,
                "name": scheduled_task.name,
                "schedule": scheduled_task.schedule,
                "next_run": scheduled_task.next_run.isoformat(),
                "last_run": scheduled_task.last_run.isoformat() if scheduled_task.last_run else None,
                "is_active": scheduled_task.is_active,
                "metadata": scheduled_task.metadata
            }
        
        return None
    
    def get_all_tasks(self) -> List[Dict[str, Any]]:
        """Get all tasks"""
        tasks = []
        
        # Add regular tasks
        for task in self.tasks.values():
            tasks.append({
                "id": task.id,
                "name": task.name,
                "type": "task",
                "status": task.status.value,
                "priority": task.priority.value,
                "created_at": task.created_at.isoformat(),
                "started_at": task.started_at.isoformat() if task.started_at else None,
                "completed_at": task.completed_at.isoformat() if task.completed_at else None,
                "retry_count": task.retry_count,
                "error": task.error
            })
        
        # Add scheduled tasks
        for scheduled_task in self.scheduled_tasks.values():
            tasks.append({
                "id": scheduled_task.id,
                "name": scheduled_task.name,
                "type": "scheduled",
                "schedule": scheduled_task.schedule,
                "next_run": scheduled_task.next_run.isoformat(),
                "last_run": scheduled_task.last_run.isoformat() if scheduled_task.last_run else None,
                "is_active": scheduled_task.is_active
            })
        
        return tasks
    
    def get_service_stats(self) -> Dict[str, Any]:
        """Get service statistics"""
        total_tasks = len(self.tasks)
        pending_tasks = len([t for t in self.tasks.values() if t.status == TaskStatus.PENDING])
        running_tasks = len([t for t in self.tasks.values() if t.status == TaskStatus.RUNNING])
        completed_tasks = len([t for t in self.tasks.values() if t.status == TaskStatus.COMPLETED])
        failed_tasks = len([t for t in self.tasks.values() if t.status == TaskStatus.FAILED])
        cancelled_tasks = len([t for t in self.tasks.values() if t.status == TaskStatus.CANCELLED])
        
        return {
            "is_running": self.is_running,
            "max_workers": self.max_workers,
            "active_workers": len(self.worker_tasks),
            "queue_size": self.task_queue.qsize(),
            "total_tasks": total_tasks,
            "pending_tasks": pending_tasks,
            "running_tasks": running_tasks,
            "completed_tasks": completed_tasks,
            "failed_tasks": failed_tasks,
            "cancelled_tasks": cancelled_tasks,
            "scheduled_tasks": len(self.scheduled_tasks),
            "active_scheduled_tasks": len([t for t in self.scheduled_tasks.values() if t.is_active])
        }
    
    async def _worker(self, worker_name: str):
        """Worker coroutine"""
        logger.info(f"Worker {worker_name} started")
        
        while self.is_running:
            try:
                # Get task from queue
                task = await asyncio.wait_for(self.task_queue.get(), timeout=1.0)
                
                # Check if task is cancelled
                if task.status == TaskStatus.CANCELLED:
                    continue
                
                # Execute task
                await self._execute_task(task, worker_name)
                
            except asyncio.TimeoutError:
                # No tasks available, continue
                continue
            except Exception as e:
                logger.error(f"Worker {worker_name} error: {e}")
                await asyncio.sleep(1)
        
        logger.info(f"Worker {worker_name} stopped")
    
    async def _execute_task(self, task: Task, worker_name: str):
        """Execute a task"""
        task.status = TaskStatus.RUNNING
        task.started_at = DateTimeHelpers.now_utc()
        
        # Create asyncio task
        asyncio_task = asyncio.create_task(self._run_task_func(task))
        self.running_tasks[task.id] = asyncio_task
        
        try:
            # Wait for task completion
            result = await asyncio_task
            task.result = result
            task.status = TaskStatus.COMPLETED
            task.completed_at = DateTimeHelpers.now_utc()
            
            logger.info(f"Task {task.id} completed by {worker_name}")
        
        except asyncio.CancelledError:
            task.status = TaskStatus.CANCELLED
            task.completed_at = DateTimeHelpers.now_utc()
            logger.info(f"Task {task.id} cancelled")
        
        except Exception as e:
            task.error = str(e)
            task.retry_count += 1
            
            if task.retry_count < task.max_retries:
                # Retry task
                task.status = TaskStatus.PENDING
                task.started_at = None
                await self.task_queue.put(task)
                logger.warning(f"Task {task.id} failed, retrying ({task.retry_count}/{task.max_retries}): {e}")
            else:
                # Max retries reached
                task.status = TaskStatus.FAILED
                task.completed_at = DateTimeHelpers.now_utc()
                logger.error(f"Task {task.id} failed after {task.max_retries} retries: {e}")
        
        finally:
            # Clean up
            if task.id in self.running_tasks:
                del self.running_tasks[task.id]
    
    async def _run_task_func(self, task: Task):
        """Run the task function"""
        if asyncio.iscoroutinefunction(task.func):
            return await task.func(*task.args, **task.kwargs)
        else:
            # Run sync function in thread pool
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, task.func, *task.args, **task.kwargs)
    
    async def _scheduler(self):
        """Scheduler coroutine"""
        logger.info("Scheduler started")
        
        while self.is_running:
            try:
                now = DateTimeHelpers.now_utc()
                
                # Check scheduled tasks
                for scheduled_task in self.scheduled_tasks.values():
                    if not scheduled_task.is_active:
                        continue
                    
                    if now >= scheduled_task.next_run:
                        # Submit task for execution
                        await self.submit_task(
                            name=f"{scheduled_task.name}_scheduled",
                            func=scheduled_task.func,
                            *scheduled_task.args,
                            **scheduled_task.kwargs
                        )
                        
                        # Update next run time
                        scheduled_task.last_run = now
                        scheduled_task.next_run = self._calculate_next_run(scheduled_task.schedule)
                        
                        logger.info(f"Scheduled task {scheduled_task.id} triggered")
                
                # Check for scheduled regular tasks
                for task in self.tasks.values():
                    if (task.status == TaskStatus.PENDING and 
                        task.scheduled_at and 
                        now >= task.scheduled_at):
                        await self.task_queue.put(task)
                        task.scheduled_at = None
                        logger.info(f"Scheduled task {task.id} moved to queue")
                
                # Sleep for 1 second
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                await asyncio.sleep(1)
        
        logger.info("Scheduler stopped")
    
    def _calculate_next_run(self, schedule: str) -> datetime:
        """Calculate next run time for schedule"""
        # Simple interval-based scheduling (e.g., "5m", "1h", "1d")
        # In a real implementation, you would use a proper cron parser
        
        now = DateTimeHelpers.now_utc()
        
        if schedule.endswith('m'):
            minutes = int(schedule[:-1])
            return now + timedelta(minutes=minutes)
        elif schedule.endswith('h'):
            hours = int(schedule[:-1])
            return now + timedelta(hours=hours)
        elif schedule.endswith('d'):
            days = int(schedule[:-1])
            return now + timedelta(days=days)
        else:
            # Default to 1 hour
            return now + timedelta(hours=1)


# Global background task service
background_task_service = BackgroundTaskService()


# Utility functions
async def submit_background_task(
    name: str,
    func: Callable,
    *args,
    priority: TaskPriority = TaskPriority.NORMAL,
    max_retries: int = 3,
    scheduled_at: Optional[datetime] = None,
    **kwargs
) -> str:
    """Submit a background task"""
    return await background_task_service.submit_task(
        name, func, *args, priority=priority, max_retries=max_retries,
        scheduled_at=scheduled_at, **kwargs
    )


async def schedule_recurring_task(
    name: str,
    func: Callable,
    schedule: str,
    *args,
    **kwargs
) -> str:
    """Schedule a recurring task"""
    return await background_task_service.schedule_task(
        name, func, schedule, *args, **kwargs
    )


async def cancel_background_task(task_id: str) -> bool:
    """Cancel a background task"""
    return await background_task_service.cancel_task(task_id)


def get_task_status(task_id: str) -> Optional[Dict[str, Any]]:
    """Get task status"""
    return background_task_service.get_task_status(task_id)


def get_all_tasks() -> List[Dict[str, Any]]:
    """Get all tasks"""
    return background_task_service.get_all_tasks()


def get_service_stats() -> Dict[str, Any]:
    """Get service statistics"""
    return background_task_service.get_service_stats()


# Common background tasks
async def cleanup_old_tasks():
    """Cleanup old completed tasks"""
    now = DateTimeHelpers.now_utc()
    cutoff_time = now - timedelta(days=7)
    
    tasks_to_remove = []
    for task_id, task in background_task_service.tasks.items():
        if (task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED] and
            task.completed_at and task.completed_at < cutoff_time):
            tasks_to_remove.append(task_id)
    
    for task_id in tasks_to_remove:
        del background_task_service.tasks[task_id]
    
    logger.info(f"Cleaned up {len(tasks_to_remove)} old tasks")


async def health_check():
    """Health check task"""
    stats = get_service_stats()
    logger.info(f"Background task service health: {stats}")


# Initialize common scheduled tasks
async def initialize_background_tasks():
    """Initialize common background tasks"""
    # Schedule cleanup task to run daily
    await schedule_recurring_task(
        "cleanup_old_tasks",
        cleanup_old_tasks,
        "1d"
    )
    
    # Schedule health check to run every 5 minutes
    await schedule_recurring_task(
        "health_check",
        health_check,
        "5m"
    )
    
    logger.info("Background tasks initialized")




