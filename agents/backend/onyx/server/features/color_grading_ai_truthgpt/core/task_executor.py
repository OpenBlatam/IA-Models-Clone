"""
Task Executor for Color Grading AI
===================================

Unified task execution system consolidating queue, scheduler, and async operations.
"""

import logging
import asyncio
import uuid
from typing import Dict, Any, Optional, List, Callable, TypeVar, Generic
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from abc import ABC, abstractmethod
import heapq

logger = logging.getLogger(__name__)

T = TypeVar('T')


class UnifiedTaskPriority(Enum):
    """Unified task priority levels."""
    CRITICAL = 0
    URGENT = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4
    BACKGROUND = 5


class UnifiedTaskStatus(Enum):
    """Unified task status."""
    PENDING = "pending"
    SCHEDULED = "scheduled"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


class RetryStrategy(Enum):
    """Retry strategies."""
    IMMEDIATE = "immediate"
    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    FIXED = "fixed"
    NONE = "none"


@dataclass
class Task(Generic[T]):
    """Unified task definition."""
    task_id: str
    func: Callable
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)
    priority: UnifiedTaskPriority = UnifiedTaskPriority.NORMAL
    scheduled_time: Optional[datetime] = None
    max_retries: int = 0
    retry_strategy: RetryStrategy = RetryStrategy.EXPONENTIAL
    retry_delay: float = 1.0
    status: UnifiedTaskStatus = UnifiedTaskStatus.PENDING
    result: Optional[T] = None
    error: Optional[Exception] = None
    depends_on: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    
    def __lt__(self, other):
        """Compare for priority queue."""
        if self.priority.value != other.priority.value:
            return self.priority.value < other.priority.value
        scheduled_self = self.scheduled_time or datetime.min
        scheduled_other = other.scheduled_time or datetime.min
        return scheduled_self < scheduled_other


class TaskExecutor:
    """
    Unified task executor.
    
    Features:
    - Priority-based execution
    - Scheduled execution
    - Retry logic
    - Dependency management
    - Concurrent execution
    - Resource awareness
    - Progress tracking
    """
    
    def __init__(
        self,
        max_concurrent: int = 10,
        enable_resource_check: bool = False
    ):
        """
        Initialize task executor.
        
        Args:
            max_concurrent: Maximum concurrent tasks
            enable_resource_check: Enable resource checking
        """
        self.max_concurrent = max_concurrent
        self.enable_resource_check = enable_resource_check
        self._task_queue: List[tuple] = []  # Priority queue
        self._tasks: Dict[str, Task] = {}
        self._running_tasks: Dict[str, asyncio.Task] = {}
        self._completed_tasks: List[Task] = []
        self._max_completed_history = 1000
        self._executor_task: Optional[asyncio.Task] = None
        self._running = False
        self._resource_optimizer: Optional[Any] = None
    
    def set_resource_optimizer(self, resource_optimizer: Any):
        """Set resource optimizer."""
        self._resource_optimizer = resource_optimizer
        self.enable_resource_check = True
    
    def submit(
        self,
        func: Callable,
        *args,
        task_id: Optional[str] = None,
        priority: UnifiedTaskPriority = UnifiedTaskPriority.NORMAL,
        scheduled_time: Optional[datetime] = None,
        max_retries: int = 0,
        retry_strategy: RetryStrategy = RetryStrategy.EXPONENTIAL,
        retry_delay: float = 1.0,
        depends_on: Optional[List[str]] = None,
        **kwargs
    ) -> str:
        """
        Submit a task for execution.
        
        Args:
            func: Function to execute
            *args: Function arguments
            task_id: Optional task ID
            priority: Task priority
            scheduled_time: Optional scheduled time
            max_retries: Maximum retries
            retry_strategy: Retry strategy
            retry_delay: Retry delay in seconds
            depends_on: Optional task dependencies
            **kwargs: Function keyword arguments
            
        Returns:
            Task ID
        """
        if task_id is None:
            task_id = str(uuid.uuid4())
        
        task = Task(
            task_id=task_id,
            func=func,
            args=args,
            kwargs=kwargs,
            priority=priority,
            scheduled_time=scheduled_time,
            max_retries=max_retries,
            retry_strategy=retry_strategy,
            retry_delay=retry_delay,
            depends_on=depends_on or []
        )
        
        self._tasks[task_id] = task
        
        # Add to priority queue
        scheduled = scheduled_time or datetime.now()
        heapq.heappush(
            self._task_queue,
            (
                scheduled.timestamp(),
                task.priority.value,
                task_id
            )
        )
        
        task.status = UnifiedTaskStatus.QUEUED
        logger.info(f"Submitted task {task_id} with priority {priority.name}")
        
        return task_id
    
    async def start(self):
        """Start task executor."""
        if self._running:
            return
        
        self._running = True
        self._executor_task = asyncio.create_task(self._executor_loop())
        logger.info("Task executor started")
    
    async def stop(self, wait_for_tasks: bool = True):
        """
        Stop task executor.
        
        Args:
            wait_for_tasks: Whether to wait for running tasks
        """
        self._running = False
        
        if self._executor_task:
            self._executor_task.cancel()
            try:
                await self._executor_task
            except asyncio.CancelledError:
                pass
        
        # Wait for running tasks
        if wait_for_tasks and self._running_tasks:
            await asyncio.gather(*self._running_tasks.values(), return_exceptions=True)
        
        logger.info("Task executor stopped")
    
    async def _executor_loop(self):
        """Main executor loop."""
        while self._running:
            try:
                # Check for tasks to run
                while (
                    len(self._running_tasks) < self.max_concurrent and
                    self._task_queue
                ):
                    scheduled_time, priority, task_id = self._task_queue[0]
                    
                    # Check if scheduled time has arrived
                    if scheduled_time > datetime.now().timestamp():
                        break
                    
                    heapq.heappop(self._task_queue)
                    
                    if task_id not in self._tasks:
                        continue
                    
                    task = self._tasks[task_id]
                    
                    if task.status not in [UnifiedTaskStatus.QUEUED, UnifiedTaskStatus.SCHEDULED]:
                        continue
                    
                    # Check dependencies
                    if not self._check_dependencies(task):
                        # Reschedule
                        heapq.heappush(
                            self._task_queue,
                            (scheduled_time, priority, task_id)
                        )
                        await asyncio.sleep(0.1)
                        continue
                    
                    # Check resources if enabled
                    if self.enable_resource_check and self._resource_optimizer:
                        if not self._check_resources():
                            # Reschedule
                            heapq.heappush(
                                self._task_queue,
                                (datetime.now().timestamp() + 5, priority, task_id)
                            )
                            await asyncio.sleep(0.1)
                            continue
                    
                    # Start task
                    task.status = UnifiedTaskStatus.RUNNING
                    asyncio_task = asyncio.create_task(self._execute_task(task))
                    self._running_tasks[task_id] = asyncio_task
                
                # Cleanup completed tasks
                await self._cleanup_completed()
                
                # Sleep briefly
                await asyncio.sleep(0.1)
            
            except Exception as e:
                logger.error(f"Error in executor loop: {e}")
                await asyncio.sleep(1)
    
    def _check_dependencies(self, task: Task) -> bool:
        """Check if task dependencies are met."""
        if not task.depends_on:
            return True
        
        for dep_id in task.depends_on:
            dep_task = self._tasks.get(dep_id)
            if not dep_task:
                return False
            if dep_task.status != UnifiedTaskStatus.COMPLETED:
                return False
        
        return True
    
    def _check_resources(self) -> bool:
        """Check if resources are available."""
        if not self._resource_optimizer:
            return True
        
        try:
            from ..services.resource_optimizer import ResourceType
            cpu_usage = self._resource_optimizer.get_resource_usage(ResourceType.CPU)
            return cpu_usage.percentage < 80.0
        except Exception:
            return True
    
    async def _execute_task(self, task: Task):
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
                task.status = UnifiedTaskStatus.COMPLETED
                task.result = result
                break
            
            except Exception as e:
                task.error = e
                retries += 1
                
                if retries > task.max_retries:
                    task.status = UnifiedTaskStatus.FAILED
                    logger.error(f"Task {task.task_id} failed after {retries} retries: {e}")
                else:
                    task.status = UnifiedTaskStatus.RETRYING
                    delay = self._calculate_retry_delay(task, retries)
                    logger.warning(f"Task {task.task_id} failed, retrying ({retries}/{task.max_retries}) in {delay}s: {e}")
                    await asyncio.sleep(delay)
        
        # Remove from running
        self._running_tasks.pop(task.task_id, None)
        
        # Move to completed
        self._completed_tasks.append(task)
        if len(self._completed_tasks) > self._max_completed_history:
            self._completed_tasks = self._completed_tasks[-self._max_completed_history:]
    
    def _calculate_retry_delay(self, task: Task, retry_count: int) -> float:
        """Calculate retry delay based on strategy."""
        if task.retry_strategy == RetryStrategy.IMMEDIATE:
            return 0.0
        elif task.retry_strategy == RetryStrategy.EXPONENTIAL:
            return task.retry_delay * (2 ** (retry_count - 1))
        elif task.retry_strategy == RetryStrategy.LINEAR:
            return task.retry_delay * retry_count
        elif task.retry_strategy == RetryStrategy.FIXED:
            return task.retry_delay
        else:
            return 0.0
    
    async def _cleanup_completed(self):
        """Cleanup completed tasks."""
        # Remove old completed tasks from _tasks dict
        completed_ids = {t.task_id for t in self._completed_tasks[-100:]}
        to_remove = [
            tid for tid in self._tasks.keys()
            if tid not in completed_ids and tid not in self._running_tasks
        ]
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
            "scheduled_time": task.scheduled_time.isoformat() if task.scheduled_time else None,
            "result": task.result,
            "error": str(task.error) if task.error else None,
            "retry_count": task.max_retries,
        }
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a task."""
        if task_id in self._running_tasks:
            self._running_tasks[task_id].cancel()
            self._running_tasks.pop(task_id)
        
        if task_id in self._tasks:
            task = self._tasks[task_id]
            task.status = UnifiedTaskStatus.CANCELLED
            logger.info(f"Cancelled task {task_id}")
            return True
        
        return False
    
    async def wait_for_task(self, task_id: str, timeout: Optional[float] = None) -> Any:
        """
        Wait for task to complete.
        
        Args:
            task_id: Task ID
            timeout: Optional timeout in seconds
            
        Returns:
            Task result
        """
        task = self._tasks.get(task_id)
        if not task:
            raise ValueError(f"Task {task_id} not found")
        
        start_time = datetime.now()
        
        while task.status not in [UnifiedTaskStatus.COMPLETED, UnifiedTaskStatus.FAILED, UnifiedTaskStatus.CANCELLED]:
            if timeout and (datetime.now() - start_time).total_seconds() > timeout:
                raise TimeoutError(f"Task {task_id} timed out")
            
            await asyncio.sleep(0.1)
        
        if task.status == UnifiedTaskStatus.COMPLETED:
            return task.result
        elif task.status == UnifiedTaskStatus.FAILED:
            raise task.error or Exception(f"Task {task_id} failed")
        else:
            raise Exception(f"Task {task_id} was cancelled")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get executor statistics."""
        return {
            "total_tasks": len(self._tasks),
            "pending_tasks": len([t for t in self._tasks.values() if t.status == UnifiedTaskStatus.PENDING]),
            "queued_tasks": len([t for t in self._tasks.values() if t.status == UnifiedTaskStatus.QUEUED]),
            "running_tasks": len(self._running_tasks),
            "completed_tasks": len(self._completed_tasks),
            "failed_tasks": len([t for t in self._completed_tasks if t.status == UnifiedTaskStatus.FAILED]),
            "queue_size": len(self._task_queue),
        }




