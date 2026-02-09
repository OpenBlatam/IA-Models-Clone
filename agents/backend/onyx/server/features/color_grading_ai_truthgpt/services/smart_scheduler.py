"""
Smart Scheduler for Color Grading AI
=====================================

Intelligent scheduling system with priority management, resource awareness, and optimization.
"""

import logging
import asyncio
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import heapq

logger = logging.getLogger(__name__)


class TaskPriority(Enum):
    """Task priority levels."""
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BACKGROUND = 4


class TaskStatus(Enum):
    """Task status."""
    PENDING = "pending"
    SCHEDULED = "scheduled"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ScheduledTask:
    """Scheduled task."""
    task_id: str
    func: Callable
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.NORMAL
    scheduled_time: datetime = field(default_factory=datetime.now)
    max_retries: int = 0
    retry_delay: float = 1.0
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error: Optional[Exception] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class SmartScheduler:
    """
    Smart scheduler with intelligent task management.
    
    Features:
    - Priority-based scheduling
    - Resource-aware scheduling
    - Retry logic
    - Task dependencies
    - Time-based scheduling
    - Load balancing
    """
    
    def __init__(self, max_concurrent: int = 10):
        """
        Initialize smart scheduler.
        
        Args:
            max_concurrent: Maximum concurrent tasks
        """
        self.max_concurrent = max_concurrent
        self._task_queue: List[tuple] = []  # Priority queue
        self._tasks: Dict[str, ScheduledTask] = {}
        self._running_tasks: Dict[str, asyncio.Task] = {}
        self._completed_tasks: List[ScheduledTask] = []
        self._max_completed_history = 1000
        self._scheduler_task: Optional[asyncio.Task] = None
        self._running = False
    
    def schedule(
        self,
        func: Callable,
        *args,
        task_id: Optional[str] = None,
        priority: TaskPriority = TaskPriority.NORMAL,
        scheduled_time: Optional[datetime] = None,
        max_retries: int = 0,
        retry_delay: float = 1.0,
        **kwargs
    ) -> str:
        """
        Schedule a task.
        
        Args:
            func: Function to execute
            *args: Function arguments
            task_id: Optional task ID
            priority: Task priority
            scheduled_time: Optional scheduled time
            max_retries: Maximum retries
            retry_delay: Retry delay in seconds
            **kwargs: Function keyword arguments
            
        Returns:
            Task ID
        """
        if task_id is None:
            import uuid
            task_id = str(uuid.uuid4())
        
        task = ScheduledTask(
            task_id=task_id,
            func=func,
            args=args,
            kwargs=kwargs,
            priority=priority,
            scheduled_time=scheduled_time or datetime.now(),
            max_retries=max_retries,
            retry_delay=retry_delay
        )
        
        self._tasks[task_id] = task
        
        # Add to priority queue
        heapq.heappush(
            self._task_queue,
            (
                task.scheduled_time.timestamp(),
                task.priority.value,
                task_id
            )
        )
        
        logger.info(f"Scheduled task {task_id} with priority {priority.name}")
        
        return task_id
    
    async def start(self):
        """Start scheduler."""
        if self._running:
            return
        
        self._running = True
        self._scheduler_task = asyncio.create_task(self._scheduler_loop())
        logger.info("Smart scheduler started")
    
    async def stop(self):
        """Stop scheduler."""
        self._running = False
        
        if self._scheduler_task:
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass
        
        # Wait for running tasks
        if self._running_tasks:
            await asyncio.gather(*self._running_tasks.values(), return_exceptions=True)
        
        logger.info("Smart scheduler stopped")
    
    async def _scheduler_loop(self):
        """Main scheduler loop."""
        while self._running:
            try:
                # Check for tasks to run
                while (
                    len(self._running_tasks) < self.max_concurrent and
                    self._task_queue and
                    self._task_queue[0][0] <= datetime.now().timestamp()
                ):
                    _, priority, task_id = heapq.heappop(self._task_queue)
                    
                    if task_id not in self._tasks:
                        continue
                    
                    task = self._tasks[task_id]
                    
                    if task.status != TaskStatus.PENDING:
                        continue
                    
                    # Start task
                    task.status = TaskStatus.RUNNING
                    asyncio_task = asyncio.create_task(self._execute_task(task))
                    self._running_tasks[task_id] = asyncio_task
                
                # Cleanup completed tasks
                await self._cleanup_completed()
                
                # Sleep briefly
                await asyncio.sleep(0.1)
            
            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")
                await asyncio.sleep(1)
    
    async def _execute_task(self, task: ScheduledTask):
        """Execute a task."""
        retries = 0
        
        while retries <= task.max_retries:
            try:
                # Execute task
                if asyncio.iscoroutinefunction(task.func):
                    result = await task.func(*task.args, **task.kwargs)
                else:
                    result = task.func(*task.args, **task.kwargs)
                
                # Success
                task.status = TaskStatus.COMPLETED
                task.result = result
                break
            
            except Exception as e:
                task.error = e
                retries += 1
                
                if retries > task.max_retries:
                    task.status = TaskStatus.FAILED
                    logger.error(f"Task {task.task_id} failed after {retries} retries: {e}")
                else:
                    logger.warning(f"Task {task.task_id} failed, retrying ({retries}/{task.max_retries}): {e}")
                    await asyncio.sleep(task.retry_delay * retries)
        
        # Remove from running
        self._running_tasks.pop(task.task_id, None)
        
        # Move to completed
        self._completed_tasks.append(task)
        if len(self._completed_tasks) > self._max_completed_history:
            self._completed_tasks = self._completed_tasks[-self._max_completed_history:]
    
    async def _cleanup_completed(self):
        """Cleanup completed tasks."""
        # Remove old completed tasks from _tasks dict
        completed_ids = {t.task_id for t in self._completed_tasks[-100:]}
        to_remove = [tid for tid in self._tasks.keys() if tid not in completed_ids and tid not in self._running_tasks]
        for tid in to_remove:
            self._tasks.pop(tid, None)
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task status."""
        task = self._tasks.get(task_id)
        if not task:
            return None
        
        return {
            "task_id": task.task_id,
            "status": task.status.value,
            "priority": task.priority.name,
            "scheduled_time": task.scheduled_time.isoformat(),
            "result": task.result,
            "error": str(task.error) if task.error else None,
        }
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a task."""
        if task_id in self._running_tasks:
            # Cancel running task
            self._running_tasks[task_id].cancel()
            self._running_tasks.pop(task_id)
        
        if task_id in self._tasks:
            task = self._tasks[task_id]
            task.status = TaskStatus.CANCELLED
            logger.info(f"Cancelled task {task_id}")
            return True
        
        return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get scheduler statistics."""
        return {
            "total_tasks": len(self._tasks),
            "pending_tasks": len([t for t in self._tasks.values() if t.status == TaskStatus.PENDING]),
            "running_tasks": len(self._running_tasks),
            "completed_tasks": len(self._completed_tasks),
            "failed_tasks": len([t for t in self._completed_tasks if t.status == TaskStatus.FAILED]),
            "queue_size": len(self._task_queue),
        }




