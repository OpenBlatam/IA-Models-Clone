"""
High-performance task scheduler and job queue system for Blaze AI.

This module provides advanced task scheduling, job queuing, worker pool management,
and intelligent load balancing for optimal performance and resource utilization.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Callable, Awaitable, Coroutine
from collections import defaultdict, deque
import heapq
import threading
import weakref
import pickle

from ..core.interfaces import CoreConfig
from ..utils.logging import get_logger

# =============================================================================
# Core Types and Enums
# =============================================================================

class TaskPriority(Enum):
    """Task priority levels."""
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BACKGROUND = 4

class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"

class WorkerStatus(Enum):
    """Worker status."""
    IDLE = "idle"
    BUSY = "busy"
    OFFLINE = "offline"
    ERROR = "error"

@dataclass
class Task:
    """Task definition with metadata."""
    id: str
    name: str
    func: Callable[..., Awaitable[Any]]
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.NORMAL
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    timeout: Optional[float] = None
    retries: int = 0
    max_retries: int = 3
    result: Optional[Any] = None
    error: Optional[str] = None
    worker_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-initialization setup."""
        if not self.id:
            self.id = str(uuid.uuid4())
    
    @property
    def duration(self) -> Optional[float]:
        """Get task duration in seconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        elif self.started_at:
            return (datetime.utcnow() - self.started_at).total_seconds()
        return None
    
    @property
    def age(self) -> float:
        """Get task age in seconds."""
        return (datetime.utcnow() - self.created_at).total_seconds()
    
    def is_expired(self) -> bool:
        """Check if task has expired."""
        if self.timeout is None:
            return False
        return self.age > self.timeout

@dataclass
class Worker:
    """Worker instance for task execution."""
    id: str
    name: str
    status: WorkerStatus = WorkerStatus.IDLE
    current_task: Optional[str] = None
    tasks_completed: int = 0
    tasks_failed: int = 0
    total_work_time: float = 0.0
    last_heartbeat: datetime = field(default_factory=datetime.utcnow)
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-initialization setup."""
        if not self.id:
            self.id = str(uuid.uuid4())
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        total = self.tasks_completed + self.tasks_failed
        return self.tasks_completed / total if total > 0 else 0.0
    
    @property
    def average_task_time(self) -> float:
        """Calculate average task execution time."""
        return self.total_work_time / self.tasks_completed if self.tasks_completed > 0 else 0.0

@dataclass
class SchedulerConfig:
    """Task scheduler configuration."""
    max_workers: int = 10
    min_workers: int = 2
    worker_timeout: float = 300.0  # 5 minutes
    task_timeout: float = 600.0  # 10 minutes
    max_queue_size: int = 10000
    enable_priority_queue: bool = True
    enable_worker_autoscaling: bool = True
    autoscaling_threshold: float = 0.8  # 80% worker utilization
    autoscaling_cooldown: float = 60.0  # 1 minute
    enable_task_persistence: bool = False
    persistence_file: str = "tasks_backup.pkl"
    enable_metrics: bool = True
    cleanup_interval: float = 300.0  # 5 minutes

# =============================================================================
# Priority Queue Implementation
# =============================================================================

class PriorityTaskQueue:
    """Priority queue for tasks with efficient ordering."""
    
    def __init__(self):
        self._queue: List[tuple] = []
        self._entry_count = 0
        self._lock = asyncio.Lock()
    
    async def put(self, task: Task):
        """Add task to priority queue."""
        async with self._lock:
            # Use negative priority for min-heap (lower numbers = higher priority)
            priority = -task.priority.value
            entry = (priority, self._entry_count, task)
            heapq.heappush(self._queue, entry)
            self._entry_count += 1
    
    async def get(self) -> Optional[Task]:
        """Get highest priority task from queue."""
        async with self._lock:
            if not self._queue:
                return None
            
            _, _, task = heapq.heappop(self._queue)
            return task
    
    async def peek(self) -> Optional[Task]:
        """Peek at highest priority task without removing."""
        async with self._lock:
            if not self._queue:
                return None
            
            return self._queue[0][2]
    
    async def size(self) -> int:
        """Get queue size."""
        async with self._lock:
            return len(self._queue)
    
    async def clear(self):
        """Clear all tasks from queue."""
        async with self._lock:
            self._queue.clear()
            self._entry_count = 0
    
    async def remove_task(self, task_id: str) -> bool:
        """Remove specific task from queue."""
        async with self._lock:
            for i, (_, _, task) in enumerate(self._queue):
                if task.id == task_id:
                    self._queue.pop(i)
                    heapq.heapify(self._queue)  # Rebuild heap
                    return True
            return False

# =============================================================================
# Worker Pool Management
# =============================================================================

class WorkerPool:
    """Manages worker instances and load balancing."""
    
    def __init__(self, config: SchedulerConfig):
        self.config = config
        self.logger = get_logger("worker_pool")
        self.workers: Dict[str, Worker] = {}
        self.available_workers: deque = deque()
        self.busy_workers: Dict[str, Worker] = {}
        self._lock = asyncio.Lock()
        self._worker_counter = 0
        
        self._initialize_workers()
    
    def _initialize_workers(self):
        """Initialize initial worker pool."""
        for i in range(self.config.min_workers):
            worker = Worker(
                id=f"worker_{i}",
                name=f"Worker-{i}",
                status=WorkerStatus.IDLE
            )
            self.workers[worker.id] = worker
            self.available_workers.append(worker.id)
    
    async def get_available_worker(self) -> Optional[Worker]:
        """Get next available worker."""
        async with self._lock:
            if not self.available_workers:
                return None
            
            worker_id = self.available_workers.popleft()
            worker = self.workers[worker_id]
            worker.status = WorkerStatus.BUSY
            self.busy_workers[worker_id] = worker
            
            return worker
    
    async def release_worker(self, worker_id: str):
        """Release worker back to available pool."""
        async with self._lock:
            if worker_id in self.busy_workers:
                worker = self.busy_workers.pop(worker_id)
                worker.status = WorkerStatus.IDLE
                worker.current_task = None
                self.available_workers.append(worker_id)
    
    async def add_worker(self) -> Worker:
        """Add new worker to pool."""
        async with self._lock:
            worker_id = f"worker_{self._worker_counter}"
            self._worker_counter += 1
            
            worker = Worker(
                id=worker_id,
                name=f"Worker-{self._worker_counter}",
                status=WorkerStatus.IDLE
            )
            
            self.workers[worker_id] = worker
            self.available_workers.append(worker_id)
            
            self.logger.info(f"Added new worker: {worker_id}")
            return worker
    
    async def remove_worker(self, worker_id: str) -> bool:
        """Remove worker from pool."""
        async with self._lock:
            if worker_id in self.workers:
                worker = self.workers[worker_id]
                
                # Remove from appropriate collections
                if worker_id in self.available_workers:
                    self.available_workers.remove(worker_id)
                if worker_id in self.busy_workers:
                    del self.busy_workers[worker_id]
                
                del self.workers[worker_id]
                
                self.logger.info(f"Removed worker: {worker_id}")
                return True
            
            return False
    
    async def get_worker_stats(self) -> Dict[str, Any]:
        """Get worker pool statistics."""
        async with self._lock:
            total_workers = len(self.workers)
            available_workers = len(self.available_workers)
            busy_workers = len(self.busy_workers)
            
            if total_workers > 0:
                utilization = busy_workers / total_workers
            else:
                utilization = 0.0
            
            return {
                "total_workers": total_workers,
                "available_workers": available_workers,
                "busy_workers": busy_workers,
                "utilization": utilization,
                "workers": {
                    wid: {
                        "status": worker.status.value,
                        "current_task": worker.current_task,
                        "tasks_completed": worker.tasks_completed,
                        "tasks_failed": worker.tasks_failed,
                        "success_rate": worker.success_rate,
                        "average_task_time": worker.average_task_time
                    }
                    for wid, worker in self.workers.items()
                }
            }
    
    async def cleanup_dead_workers(self):
        """Remove workers that haven't reported in recently."""
        async with self._lock:
            current_time = datetime.utcnow()
            timeout = timedelta(seconds=self.config.worker_timeout)
            
            dead_workers = []
            for worker_id, worker in self.workers.items():
                if current_time - worker.last_heartbeat > timeout:
                    dead_workers.append(worker_id)
            
            for worker_id in dead_workers:
                await self.remove_worker(worker_id)
                self.logger.warning(f"Removed dead worker: {worker_id}")
    
    async def should_autoscale(self) -> bool:
        """Check if worker pool should be autoscaled."""
        if not self.config.enable_worker_autoscaling:
            return False
        
        stats = await self.get_worker_stats()
        return stats["utilization"] > self.config.autoscaling_threshold

# =============================================================================
# Task Execution Engine
# =============================================================================

class TaskExecutor:
    """Executes tasks using worker pool."""
    
    def __init__(self, worker_pool: WorkerPool):
        self.worker_pool = worker_pool
        self.logger = get_logger("task_executor")
        self.running_tasks: Dict[str, asyncio.Task] = {}
        self._lock = asyncio.Lock()
    
    async def execute_task(self, task: Task) -> Any:
        """Execute task using available worker."""
        worker = await self.worker_pool.get_available_worker()
        if not worker:
            raise RuntimeError("No available workers")
        
        try:
            # Update task status
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.utcnow()
            task.worker_id = worker.id
            worker.current_task = task.id
            
            # Execute task
            if asyncio.iscoroutinefunction(task.func):
                result = await task.func(*task.args, **task.kwargs)
            else:
                # Handle sync functions
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, task.func, *task.args, **task.kwargs)
            
            # Update task status
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.utcnow()
            task.result = result
            
            # Update worker stats
            worker.tasks_completed += 1
            if task.duration:
                worker.total_work_time += task.duration
            
            self.logger.info(f"Task {task.id} completed successfully in {task.duration:.2f}s")
            return result
            
        except asyncio.CancelledError:
            task.status = TaskStatus.CANCELLED
            worker.tasks_failed += 1
            raise
            
        except Exception as e:
            # Handle task failure
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.completed_at = datetime.utcnow()
            worker.tasks_failed += 1
            
            self.logger.error(f"Task {task.id} failed: {e}")
            
            # Retry logic
            if task.retries < task.max_retries:
                task.retries += 1
                task.status = TaskStatus.PENDING
                task.started_at = None
                task.completed_at = None
                task.error = None
                task.worker_id = None
                
                self.logger.info(f"Retrying task {task.id} (attempt {task.retries}/{task.max_retries})")
                return await self.execute_task(task)
            
            raise
            
        finally:
            # Release worker
            await self.worker_pool.release_worker(worker.id)
    
    async def execute_task_async(self, task: Task) -> asyncio.Task:
        """Execute task asynchronously and return task handle."""
        async with self._lock:
            if task.id in self.running_tasks:
                raise ValueError(f"Task {task.id} is already running")
            
            # Create execution task
            execution_task = asyncio.create_task(self.execute_task(task))
            self.running_tasks[task.id] = execution_task
            
            # Add cleanup callback
            execution_task.add_done_callback(
                lambda t: asyncio.create_task(self._cleanup_task(task.id))
            )
            
            return execution_task
    
    async def _cleanup_task(self, task_id: str):
        """Clean up completed task."""
        async with self._lock:
            if task_id in self.running_tasks:
                del self.running_tasks[task_id]
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel running task."""
        async with self._lock:
            if task_id in self.running_tasks:
                task = self.running_tasks[task_id]
                task.cancel()
                return True
            return False
    
    async def get_running_tasks(self) -> List[str]:
        """Get list of running task IDs."""
        async with self._lock:
            return list(self.running_tasks.keys())

# =============================================================================
# Main Task Scheduler
# =============================================================================

class TaskScheduler:
    """Main task scheduler coordinating all components."""
    
    def __init__(self, config: Optional[SchedulerConfig] = None):
        self.config = config or SchedulerConfig()
        self.logger = get_logger("task_scheduler")
        
        # Initialize components
        self.task_queue = PriorityTaskQueue()
        self.worker_pool = WorkerPool(self.config)
        self.task_executor = TaskExecutor(self.worker_pool)
        
        # Task tracking
        self.tasks: Dict[str, Task] = {}
        self.completed_tasks: List[Task] = []
        self.failed_tasks: List[Task] = []
        
        # Background tasks
        self._scheduler_task: Optional[asyncio.Task] = None
        self._autoscaling_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
        
        self._start_background_tasks()
    
    def _start_background_tasks(self):
        """Start background scheduler and maintenance tasks."""
        if self._scheduler_task is None or self._scheduler_task.done():
            self._scheduler_task = asyncio.create_task(self._scheduler_loop())
        
        if self.config.enable_worker_autoscaling and (self._autoscaling_task is None or self._autoscaling_task.done()):
            self._autoscaling_task = asyncio.create_task(self._autoscaling_loop())
        
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
    
    async def submit_task(self, name: str, func: Callable[..., Awaitable[Any]], 
                         *args, priority: TaskPriority = TaskPriority.NORMAL,
                         timeout: Optional[float] = None, max_retries: int = 3,
                         **kwargs) -> str:
        """Submit task for execution."""
        # Check queue size limit
        if await self.task_queue.size() >= self.config.max_queue_size:
            raise RuntimeError("Task queue is full")
        
        # Create task
        task = Task(
            name=name,
            func=func,
            args=args,
            kwargs=kwargs,
            priority=priority,
            timeout=timeout or self.config.task_timeout,
            max_retries=max_retries
        )
        
        # Add to queue and tracking
        await self.task_queue.put(task)
        self.tasks[task.id] = task
        
        self.logger.info(f"Submitted task {task.id} ({name}) with priority {priority.value}")
        return task.id
    
    async def submit_task_sync(self, name: str, func: Callable[..., Any],
                              *args, priority: TaskPriority = TaskPriority.NORMAL,
                              timeout: Optional[float] = None, max_retries: int = 3,
                              **kwargs) -> str:
        """Submit synchronous function as task."""
        # Wrap sync function in async wrapper
        async def async_wrapper():
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, func, *args, **kwargs)
        
        return await self.submit_task(name, async_wrapper, priority=priority, 
                                    timeout=timeout, max_retries=max_retries)
    
    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task status and details."""
        task = self.tasks.get(task_id)
        if not task:
            return None
        
        return {
            "id": task.id,
            "name": task.name,
            "status": task.status.value,
            "priority": task.priority.value,
            "created_at": task.created_at.isoformat(),
            "started_at": task.started_at.isoformat() if task.started_at else None,
            "completed_at": task.completed_at.isoformat() if task.completed_at else None,
            "duration": task.duration,
            "age": task.age,
            "retries": task.retries,
            "max_retries": task.max_retries,
            "worker_id": task.worker_id,
            "error": task.error,
            "metadata": task.metadata
        }
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel pending or running task."""
        task = self.tasks.get(task_id)
        if not task:
            return False
        
        if task.status == TaskStatus.PENDING:
            # Remove from queue
            await self.task_queue.remove_task(task_id)
            task.status = TaskStatus.CANCELLED
            return True
        
        elif task.status == TaskStatus.RUNNING:
            # Cancel execution
            cancelled = await self.task_executor.cancel_task(task_id)
            if cancelled:
                task.status = TaskStatus.CANCELLED
            return cancelled
        
        return False
    
    async def wait_for_task(self, task_id: str, timeout: Optional[float] = None) -> Any:
        """Wait for task completion and return result."""
        task = self.tasks.get(task_id)
        if not task:
            raise ValueError(f"Task {task_id} not found")
        
        if task.status == TaskStatus.COMPLETED:
            return task.result
        
        if task.status == TaskStatus.FAILED:
            raise RuntimeError(f"Task {task_id} failed: {task.error}")
        
        if task.status == TaskStatus.CANCELLED:
            raise RuntimeError(f"Task {task_id} was cancelled")
        
        # Wait for completion
        start_time = time.time()
        while task.status not in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
            if timeout and (time.time() - start_time) > timeout:
                raise asyncio.TimeoutError(f"Task {task_id} timed out")
            
            await asyncio.sleep(0.1)
        
        if task.status == TaskStatus.COMPLETED:
            return task.result
        elif task.status == TaskStatus.FAILED:
            raise RuntimeError(f"Task {task_id} failed: {task.error}")
        else:
            raise RuntimeError(f"Task {task_id} was cancelled")
    
    async def get_scheduler_stats(self) -> Dict[str, Any]:
        """Get comprehensive scheduler statistics."""
        queue_size = await self.task_queue.size()
        worker_stats = await self.worker_pool.get_worker_stats()
        running_tasks = await self.task_executor.get_running_tasks()
        
        # Count tasks by status
        status_counts = defaultdict(int)
        for task in self.tasks.values():
            status_counts[task.status.value] += 1
        
        return {
            "queue": {
                "size": queue_size,
                "max_size": self.config.max_queue_size
            },
            "workers": worker_stats,
            "tasks": {
                "total": len(self.tasks),
                "running": len(running_tasks),
                "by_status": dict(status_counts),
                "completed": len(self.completed_tasks),
                "failed": len(self.failed_tasks)
            },
            "performance": {
                "enable_priority_queue": self.config.enable_priority_queue,
                "enable_worker_autoscaling": self.config.enable_worker_autoscaling,
                "enable_metrics": self.config.enable_metrics
            }
        }
    
    async def _scheduler_loop(self):
        """Main scheduler loop for processing tasks."""
        while not self._shutdown_event.is_set():
            try:
                # Get next task from queue
                task = await self.task_queue.get()
                if not task:
                    await asyncio.sleep(0.1)
                    continue
                
                # Check if task has expired
                if task.is_expired():
                    task.status = TaskStatus.TIMEOUT
                    task.error = "Task expired"
                    self.failed_tasks.append(task)
                    self.logger.warning(f"Task {task.id} expired")
                    continue
                
                # Execute task
                try:
                    await self.task_executor.execute_task_async(task)
                except Exception as e:
                    self.logger.error(f"Failed to execute task {task.id}: {e}")
                    task.status = TaskStatus.FAILED
                    task.error = str(e)
                    self.failed_tasks.append(task)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Scheduler loop error: {e}")
                await asyncio.sleep(1)
    
    async def _autoscaling_loop(self):
        """Worker pool autoscaling loop."""
        while not self._shutdown_event.is_set():
            try:
                if await self.worker_pool.should_autoscale():
                    # Add new worker
                    await self.worker_pool.add_worker()
                
                await asyncio.sleep(self.config.autoscaling_cooldown)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Autoscaling error: {e}")
                await asyncio.sleep(60)
    
    async def _cleanup_loop(self):
        """Cleanup and maintenance loop."""
        while not self._shutdown_event.is_set():
            try:
                # Cleanup dead workers
                await self.worker_pool.cleanup_dead_workers()
                
                # Move completed/failed tasks to appropriate collections
                completed = []
                failed = []
                
                for task in self.tasks.values():
                    if task.status == TaskStatus.COMPLETED:
                        completed.append(task)
                    elif task.status in [TaskStatus.FAILED, TaskStatus.CANCELLED, TaskStatus.TIMEOUT]:
                        failed.append(task)
                
                for task in completed:
                    if task not in self.completed_tasks:
                        self.completed_tasks.append(task)
                
                for task in failed:
                    if task not in self.failed_tasks:
                        self.failed_tasks.append(task)
                
                # Keep only recent completed/failed tasks
                max_history = 1000
                if len(self.completed_tasks) > max_history:
                    self.completed_tasks = self.completed_tasks[-max_history:]
                
                if len(self.failed_tasks) > max_history:
                    self.failed_tasks = self.failed_tasks[-max_history:]
                
                await asyncio.sleep(self.config.cleanup_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Cleanup error: {e}")
                await asyncio.sleep(600)
    
    async def shutdown(self):
        """Shutdown task scheduler."""
        self.logger.info("Shutting down task scheduler...")
        self._shutdown_event.set()
        
        # Cancel background tasks
        for task in [self._scheduler_task, self._autoscaling_task, self._cleanup_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Cancel running tasks
        running_tasks = await self.task_executor.get_running_tasks()
        for task_id in running_tasks:
            await self.task_executor.cancel_task(task_id)
        
        self.logger.info("Task scheduler shutdown complete")

# =============================================================================
# Global Scheduler Instance
# =============================================================================

_default_scheduler: Optional[TaskScheduler] = None

def get_task_scheduler(config: Optional[SchedulerConfig] = None) -> TaskScheduler:
    """Get the global task scheduler instance."""
    global _default_scheduler
    if _default_scheduler is None:
        _default_scheduler = TaskScheduler(config)
    return _default_scheduler

async def shutdown_task_scheduler():
    """Shutdown the global task scheduler."""
    global _default_scheduler
    if _default_scheduler:
        await _default_scheduler.shutdown()
        _default_scheduler = None

# Export main classes
__all__ = [
    "TaskScheduler",
    "SchedulerConfig",
    "TaskPriority",
    "TaskStatus",
    "WorkerStatus",
    "Task",
    "Worker",
    "get_task_scheduler",
    "shutdown_task_scheduler"
]


