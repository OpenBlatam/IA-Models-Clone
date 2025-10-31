"""
Task Manager
============

Central task management system for background processing.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from uuid import uuid4

from .types import Task, TaskResult, TaskStatus, TaskPriority, TaskType, TaskStatistics
from .queue import TaskQueue, InMemoryTaskQueue, RedisTaskQueue
from .executor import AsyncTaskExecutor, TaskRetryHandler
from ..config import config

logger = logging.getLogger(__name__)

class TaskManager:
    """Central task management system."""
    
    def __init__(self, queue_type: str = "memory", redis_url: str = None):
        self.queue_type = queue_type
        self.redis_url = redis_url or config.cache_url
        self.task_queue: Optional[TaskQueue] = None
        self.executor: Optional[AsyncTaskExecutor] = None
        self.task_history: Dict[str, Task] = {}
        self.scheduled_tasks: Dict[str, Task] = {}
        self._running = False
        self._cleanup_task: Optional[asyncio.Task] = None
    
    async def initialize(self):
        """Initialize the task manager."""
        try:
            # Initialize task queue
            if self.queue_type == "redis":
                self.task_queue = RedisTaskQueue(self.redis_url)
                await self.task_queue.initialize()
            else:
                self.task_queue = InMemoryTaskQueue()
            
            # Initialize executor
            self.executor = AsyncTaskExecutor(max_concurrent_tasks=config.max_concurrent_tasks)
            
            logger.info("Task manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize task manager: {str(e)}")
            raise
    
    async def start(self):
        """Start the task manager."""
        if self._running:
            return
        
        try:
            await self.initialize()
            
            # Start executor
            await self.executor.start(self.task_queue)
            
            # Start cleanup task
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            
            self._running = True
            logger.info("Task manager started")
            
        except Exception as e:
            logger.error(f"Failed to start task manager: {str(e)}")
            raise
    
    async def stop(self):
        """Stop the task manager."""
        if not self._running:
            return
        
        self._running = False
        
        # Stop cleanup task
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Stop executor
        if self.executor:
            await self.executor.stop()
        
        logger.info("Task manager stopped")
    
    async def submit_task(
        self,
        task_type: TaskType,
        name: str,
        description: str,
        function_name: str,
        args: tuple = (),
        kwargs: Dict[str, Any] = None,
        priority: TaskPriority = TaskPriority.NORMAL,
        max_retries: int = 3,
        retry_delay: int = 60,
        timeout: int = 300,
        scheduled_at: Optional[datetime] = None,
        tags: List[str] = None
    ) -> str:
        """Submit a task for execution."""
        try:
            task_id = str(uuid4())
            
            task = Task(
                task_id=task_id,
                task_type=task_type,
                name=name,
                description=description,
                function_name=function_name,
                args=args,
                kwargs=kwargs or {},
                priority=priority,
                max_retries=max_retries,
                retry_delay=retry_delay,
                timeout=timeout,
                scheduled_at=scheduled_at,
                tags=tags or []
            )
            
            # Store in history
            self.task_history[task_id] = task
            
            # Submit to executor
            if scheduled_at and scheduled_at > datetime.now():
                # Schedule for later
                self.scheduled_tasks[task_id] = task
                logger.info(f"Scheduled task: {task_id} for {scheduled_at}")
            else:
                # Execute immediately
                await self.executor.submit_task(task, self.task_queue)
                logger.info(f"Submitted task: {task_id}")
            
            return task_id
            
        except Exception as e:
            logger.error(f"Failed to submit task: {str(e)}")
            raise
    
    async def get_task(self, task_id: str) -> Optional[Task]:
        """Get a task by ID."""
        return self.task_history.get(task_id)
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a task."""
        try:
            task = self.task_history.get(task_id)
            if not task:
                return False
            
            if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                return False
            
            # Update task status
            task.status = TaskStatus.CANCELLED
            task.completed_at = datetime.now()
            
            # Remove from scheduled tasks if present
            self.scheduled_tasks.pop(task_id, None)
            
            logger.info(f"Cancelled task: {task_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cancel task {task_id}: {str(e)}")
            return False
    
    async def retry_task(self, task_id: str) -> bool:
        """Retry a failed task."""
        try:
            task = self.task_history.get(task_id)
            if not task:
                return False
            
            if task.status != TaskStatus.FAILED:
                return False
            
            # Prepare task for retry
            retry_task = await TaskRetryHandler.prepare_retry_task(task)
            
            # Submit for execution
            await self.executor.submit_task(retry_task, self.task_queue)
            
            logger.info(f"Retrying task: {task_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to retry task {task_id}: {str(e)}")
            return False
    
    async def list_tasks(
        self,
        status: Optional[TaskStatus] = None,
        task_type: Optional[TaskType] = None,
        priority: Optional[TaskPriority] = None,
        limit: int = 100
    ) -> List[Task]:
        """List tasks with optional filters."""
        tasks = list(self.task_history.values())
        
        if status:
            tasks = [task for task in tasks if task.status == status]
        
        if task_type:
            tasks = [task for task in tasks if task.task_type == task_type]
        
        if priority:
            tasks = [task for task in tasks if task.priority == priority]
        
        # Sort by creation time (newest first)
        tasks.sort(key=lambda x: x.created_at, reverse=True)
        
        return tasks[:limit]
    
    async def get_task_statistics(self) -> TaskStatistics:
        """Get task execution statistics."""
        total_tasks = len(self.task_history)
        
        if total_tasks == 0:
            return TaskStatistics()
        
        # Count by status
        status_counts = {}
        for task in self.task_history.values():
            status = task.status.value
            status_counts[status] = status_counts.get(status, 0) + 1
        
        # Count by type
        type_counts = {}
        for task in self.task_history.values():
            task_type = task.task_type.value
            type_counts[task_type] = type_counts.get(task_type, 0) + 1
        
        # Count by priority
        priority_counts = {}
        for task in self.task_history.values():
            priority = task.priority.value
            priority_counts[priority] = priority_counts.get(priority, 0) + 1
        
        # Calculate success rate
        completed_tasks = status_counts.get("completed", 0)
        failed_tasks = status_counts.get("failed", 0)
        success_rate = (completed_tasks / (completed_tasks + failed_tasks) * 100) if (completed_tasks + failed_tasks) > 0 else 0
        
        # Calculate average execution time
        execution_times = []
        for task in self.task_history.values():
            if task.started_at and task.completed_at:
                duration = (task.completed_at - task.started_at).total_seconds()
                execution_times.append(duration)
        
        average_execution_time = sum(execution_times) / len(execution_times) if execution_times else 0.0
        
        return TaskStatistics(
            total_tasks=total_tasks,
            pending_tasks=status_counts.get("pending", 0),
            running_tasks=status_counts.get("running", 0),
            completed_tasks=completed_tasks,
            failed_tasks=failed_tasks,
            cancelled_tasks=status_counts.get("cancelled", 0),
            average_execution_time=average_execution_time,
            success_rate=success_rate,
            tasks_by_type=type_counts,
            tasks_by_priority=priority_counts
        )
    
    async def _cleanup_loop(self):
        """Background cleanup loop."""
        while self._running:
            try:
                await self._cleanup_old_tasks()
                await self._process_scheduled_tasks()
                await asyncio.sleep(60)  # Run every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup loop error: {str(e)}")
                await asyncio.sleep(60)
    
    async def _cleanup_old_tasks(self):
        """Clean up old completed tasks."""
        try:
            cutoff_time = datetime.now() - timedelta(days=7)  # Keep tasks for 7 days
            
            old_tasks = [
                task_id for task_id, task in self.task_history.items()
                if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]
                and task.completed_at and task.completed_at < cutoff_time
            ]
            
            for task_id in old_tasks:
                del self.task_history[task_id]
            
            if old_tasks:
                logger.info(f"Cleaned up {len(old_tasks)} old tasks")
                
        except Exception as e:
            logger.error(f"Failed to cleanup old tasks: {str(e)}")
    
    async def _process_scheduled_tasks(self):
        """Process scheduled tasks that are ready to run."""
        try:
            current_time = datetime.now()
            ready_tasks = []
            
            for task_id, task in self.scheduled_tasks.items():
                if task.scheduled_at and task.scheduled_at <= current_time:
                    ready_tasks.append(task_id)
            
            for task_id in ready_tasks:
                task = self.scheduled_tasks.pop(task_id)
                await self.executor.submit_task(task, self.task_queue)
                logger.info(f"Executed scheduled task: {task_id}")
                
        except Exception as e:
            logger.error(f"Failed to process scheduled tasks: {str(e)}")
    
    def get_manager_status(self) -> Dict[str, Any]:
        """Get task manager status."""
        executor_status = self.executor.get_executor_status() if self.executor else {}
        
        return {
            "running": self._running,
            "queue_type": self.queue_type,
            "total_tasks": len(self.task_history),
            "scheduled_tasks": len(self.scheduled_tasks),
            "executor": executor_status
        }

# Global task manager instance
task_manager = TaskManager()
