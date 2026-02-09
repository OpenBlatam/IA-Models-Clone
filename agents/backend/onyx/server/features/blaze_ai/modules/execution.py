"""
Blaze AI Execution Module v7.2.0

This module provides intelligent task execution capabilities with load balancing,
priority management, and performance optimization through the modular system.
"""

import asyncio
import logging
import time
import threading
from typing import Any, Dict, List, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import uuid
import weakref

from .base import BaseModule, ModuleConfig, ModuleStatus

logger = logging.getLogger(__name__)

# ============================================================================
# EXECUTION MODULE CONFIGURATION
# ============================================================================

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

class ExecutionStrategy(Enum):
    """Task execution strategies."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    PRIORITY_QUEUE = "priority_queue"
    LOAD_BALANCED = "load_balanced"
    ADAPTIVE = "adaptive"

class ExecutionModuleConfig(ModuleConfig):
    """Configuration for the Execution Module."""
    
    def __init__(self, **kwargs):
        super().__init__(
            name="execution",
            module_type="EXECUTION",
            priority=1,  # High priority for task execution
            **kwargs
        )
        
        # Execution-specific configurations
        self.max_workers: int = kwargs.get("max_workers", 8)
        self.max_process_workers: int = kwargs.get("max_process_workers", 4)
        self.default_timeout: float = kwargs.get("default_timeout", 60.0)
        self.execution_strategy: ExecutionStrategy = kwargs.get("execution_strategy", ExecutionStrategy.ADAPTIVE)
        self.enable_priority_queue: bool = kwargs.get("enable_priority_queue", True)
        self.enable_load_balancing: bool = kwargs.get("enable_load_balancing", True)
        self.enable_adaptive_scaling: bool = kwargs.get("enable_adaptive_scaling", True)
        self.worker_idle_timeout: float = kwargs.get("worker_idle_timeout", 300.0)  # 5 minutes
        self.max_queue_size: int = kwargs.get("max_queue_size", 1000)
        self.retry_attempts: int = kwargs.get("retry_attempts", 3)
        self.retry_delay: float = kwargs.get("retry_delay", 1.0)

class ExecutionMetrics:
    """Metrics specific to execution operations."""
    
    def __init__(self):
        self.total_tasks: int = 0
        self.completed_tasks: int = 0
        self.failed_tasks: int = 0
        self.cancelled_tasks: int = 0
        self.timed_out_tasks: int = 0
        self.active_workers: int = 0
        self.queue_size: int = 0
        self.average_execution_time: float = 0.0
        self.average_queue_wait_time: float = 0.0
        self.throughput_tasks_per_second: float = 0.0
        self.worker_utilization: float = 0.0

# ============================================================================
# TASK AND EXECUTION IMPLEMENTATIONS
# ============================================================================

@dataclass
class Task:
    """Represents a task to be executed."""
    
    id: str
    function: Callable
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.NORMAL
    timeout: Optional[float] = None
    retry_attempts: int = 0
    max_retries: int = 3
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Execution state
    status: TaskStatus = TaskStatus.PENDING
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Optional[Any] = None
    error: Optional[Exception] = None
    attempts: int = 0
    
    def __post_init__(self):
        if self.timeout is None:
            self.timeout = 60.0  # Default timeout
    
    def start_execution(self):
        """Mark task as started."""
        self.status = TaskStatus.RUNNING
        self.started_at = time.time()
        self.attempts += 1
    
    def complete_execution(self, result: Any):
        """Mark task as completed."""
        self.status = TaskStatus.COMPLETED
        self.completed_at = time.time()
        self.result = result
    
    def fail_execution(self, error: Exception):
        """Mark task as failed."""
        self.status = TaskStatus.FAILED
        self.completed_at = time.time()
        self.error = error
    
    def can_retry(self) -> bool:
        """Check if task can be retried."""
        return self.attempts < self.max_retries and self.status == TaskStatus.FAILED
    
    def get_execution_time(self) -> float:
        """Get total execution time."""
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        elif self.started_at:
            return time.time() - self.started_at
        return 0.0
    
    def get_queue_wait_time(self) -> float:
        """Get time spent in queue."""
        if self.started_at:
            return self.started_at - self.created_at
        return time.time() - self.created_at

class PriorityQueue:
    """Priority queue for task execution."""
    
    def __init__(self):
        self._queues: Dict[TaskPriority, List[Task]] = {
            priority: [] for priority in TaskPriority
        }
        self._lock = threading.Lock()
    
    def put(self, task: Task):
        """Add task to priority queue."""
        with self._lock:
            self._queues[task.priority].append(task)
    
    def get(self) -> Optional[Task]:
        """Get highest priority task."""
        with self._lock:
            for priority in TaskPriority:
                if self._queues[priority]:
                    return self._queues[priority].pop(0)
        return None
    
    def size(self) -> int:
        """Get total queue size."""
        with self._lock:
            return sum(len(queue) for queue in self._queues.values())
    
    def empty(self) -> bool:
        """Check if queue is empty."""
        return self.size() == 0
    
    def clear(self):
        """Clear all queues."""
        with self._lock:
            for queue in self._queues.values():
                queue.clear()

class Worker:
    """Worker for executing tasks."""
    
    def __init__(self, worker_id: str, execution_module: 'ExecutionModule'):
        self.worker_id = worker_id
        self.execution_module = execution_module
        self.current_task: Optional[Task] = None
        self.is_active = False
        self.last_activity = time.time()
        self.total_tasks_completed = 0
        self.total_execution_time = 0.0
    
    async def execute_task(self, task: Task):
        """Execute a single task."""
        try:
            self.is_active = True
            self.current_task = task
            self.last_activity = time.time()
            
            task.start_execution()
            
            # Execute the task
            if asyncio.iscoroutinefunction(task.function):
                result = await asyncio.wait_for(
                    task.function(*task.args, **task.kwargs),
                    timeout=task.timeout
                )
            else:
                # Run in thread pool for synchronous functions
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    self.execution_module.thread_pool,
                    task.function,
                    *task.args,
                    **task.kwargs
                )
            
            task.complete_execution(result)
            self.total_tasks_completed += 1
            self.total_execution_time += task.get_execution_time()
            
            logger.info(f"Task {task.id} completed successfully by worker {self.worker_id}")
            
        except asyncio.TimeoutError:
            task.status = TaskStatus.TIMEOUT
            logger.warning(f"Task {task.id} timed out in worker {self.worker_id}")
            
        except Exception as e:
            task.fail_execution(e)
            logger.error(f"Task {task.id} failed in worker {self.worker_id}: {e}")
            
        finally:
            self.is_active = False
            self.current_task = None
    
    def is_idle(self, idle_timeout: float) -> bool:
        """Check if worker is idle."""
        return not self.is_active and (time.time() - self.last_activity) > idle_timeout
    
    def get_stats(self) -> Dict[str, Any]:
        """Get worker statistics."""
        return {
            "worker_id": self.worker_id,
            "is_active": self.is_active,
            "current_task": self.current_task.id if self.current_task else None,
            "last_activity": self.last_activity,
            "total_tasks_completed": self.total_tasks_completed,
            "total_execution_time": self.total_execution_time,
            "average_execution_time": (
                self.total_execution_time / self.total_tasks_completed 
                if self.total_tasks_completed > 0 else 0.0
            )
        }

# ============================================================================
# EXECUTION MODULE IMPLEMENTATION
# ============================================================================

class ExecutionModule(BaseModule):
    """
    Execution Module - Provides intelligent task execution capabilities.
    
    This module provides:
    - Priority-based task scheduling
    - Load balancing and worker management
    - Adaptive scaling
    - Task monitoring and metrics
    - Retry mechanisms
    """
    
    def __init__(self, config: ExecutionModuleConfig):
        super().__init__(config)
        self.task_queue = PriorityQueue()
        self.workers: List[Worker] = []
        self.execution_metrics = ExecutionMetrics()
        self.thread_pool: Optional[ThreadPoolExecutor] = None
        self.process_pool: Optional[ProcessPoolExecutor] = None
        self.worker_management_task: Optional[asyncio.Task] = None
        self.metrics_update_task: Optional[asyncio.Task] = None
        self._lock = threading.RLock()
        self._task_registry: Dict[str, Task] = {}
        
    async def initialize(self) -> bool:
        """Initialize the Execution Module."""
        try:
            logger.info("Initializing Execution Module...")
            
            # Create thread and process pools
            self.thread_pool = ThreadPoolExecutor(max_workers=self.config.max_workers)
            if self.config.max_process_workers > 0:
                self.process_pool = ProcessPoolExecutor(max_workers=self.config.max_process_workers)
            
            # Create initial workers
            await self._create_workers(self.config.max_workers)
            
            # Start background tasks
            self.worker_management_task = asyncio.create_task(self._worker_management_loop())
            self.metrics_update_task = asyncio.create_task(self._metrics_update_loop())
            
            self.status = ModuleStatus.ACTIVE
            logger.info("Execution Module initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Execution Module: {e}")
            self.status = ModuleStatus.ERROR
            return False
    
    async def shutdown(self) -> bool:
        """Shutdown the Execution Module."""
        try:
            logger.info("Shutting down Execution Module...")
            
            # Stop background tasks
            if self.worker_management_task:
                self.worker_management_task.cancel()
                try:
                    await self.worker_management_task
                except asyncio.CancelledError:
                    pass
            
            if self.metrics_update_task:
                self.metrics_update_task.cancel()
                try:
                    await self.metrics_update_task
                except asyncio.CancelledError:
                    pass
            
            # Shutdown pools
            if self.thread_pool:
                self.thread_pool.shutdown(wait=True)
                self.thread_pool = None
            
            if self.process_pool:
                self.process_pool.shutdown(wait=True)
                self.process_pool = None
            
            # Clear workers and tasks
            self.workers.clear()
            self.task_queue.clear()
            self._task_registry.clear()
            
            self.status = ModuleStatus.SHUTDOWN
            logger.info("Execution Module shutdown successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error during Execution Module shutdown: {e}")
            return False
    
    async def submit_task(
        self,
        function: Callable,
        *args,
        priority: TaskPriority = TaskPriority.NORMAL,
        timeout: Optional[float] = None,
        retry_attempts: int = 0,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> str:
        """Submit a task for execution."""
        try:
            # Check queue size limit
            if self.task_queue.size() >= self.config.max_queue_size:
                raise RuntimeError(f"Task queue is full ({self.config.max_queue_size} tasks)")
            
            # Create task
            task = Task(
                id=str(uuid.uuid4()),
                function=function,
                args=args,
                kwargs=kwargs,
                priority=priority,
                timeout=timeout or self.config.default_timeout,
                retry_attempts=retry_attempts or self.config.retry_attempts,
                max_retries=self.config.retry_attempts,
                tags=tags or [],
                metadata=metadata or {}
            )
            
            # Add to queue and registry
            self.task_queue.put(task)
            self._task_registry[task.id] = task
            
            # Update metrics
            self.execution_metrics.total_tasks += 1
            self.execution_metrics.queue_size = self.task_queue.size()
            
            logger.info(f"Task {task.id} submitted with priority {priority.value}")
            return task.id
            
        except Exception as e:
            logger.error(f"Failed to submit task: {e}")
            raise
    
    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific task."""
        task = self._task_registry.get(task_id)
        if not task:
            return None
        
        return {
            "id": task.id,
            "status": task.status.value,
            "priority": task.priority.value,
            "created_at": task.created_at,
            "started_at": task.started_at,
            "completed_at": task.completed_at,
            "execution_time": task.get_execution_time(),
            "queue_wait_time": task.get_queue_wait_time(),
            "attempts": task.attempts,
            "result": task.result,
            "error": str(task.error) if task.error else None,
            "tags": task.tags,
            "metadata": task.metadata
        }
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending task."""
        task = self._task_registry.get(task_id)
        if not task:
            return False
        
        if task.status == TaskStatus.PENDING:
            task.status = TaskStatus.CANCELLED
            self.execution_metrics.cancelled_tasks += 1
            logger.info(f"Task {task_id} cancelled")
            return True
        
        return False
    
    async def wait_for_task(self, task_id: str, timeout: Optional[float] = None) -> Any:
        """Wait for a task to complete and return its result."""
        task = self._task_registry.get(task_id)
        if not task:
            raise ValueError(f"Task {task_id} not found")
        
        start_time = time.time()
        while task.status not in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED, TaskStatus.TIMEOUT]:
            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError(f"Timeout waiting for task {task_id}")
            await asyncio.sleep(0.1)
        
        if task.status == TaskStatus.COMPLETED:
            return task.result
        elif task.status == TaskStatus.FAILED:
            raise task.error
        elif task.status == TaskStatus.CANCELLED:
            raise RuntimeError(f"Task {task_id} was cancelled")
        elif task.status == TaskStatus.TIMEOUT:
            raise TimeoutError(f"Task {task_id} timed out")
        
        raise RuntimeError(f"Unexpected task status: {task.status}")
    
    async def _create_workers(self, count: int):
        """Create the specified number of workers."""
        for i in range(count):
            worker = Worker(f"worker_{i}", self)
            self.workers.append(worker)
            self.execution_metrics.active_workers = len(self.workers)
    
    async def _worker_management_loop(self):
        """Background loop for managing workers and task execution."""
        while self.status == ModuleStatus.ACTIVE:
            try:
                await asyncio.sleep(0.1)  # Check every 100ms
                
                # Process tasks with available workers
                await self._process_tasks()
                
                # Adaptive scaling
                if self.config.enable_adaptive_scaling:
                    await self._adaptive_scaling()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in worker management loop: {e}")
    
    async def _process_tasks(self):
        """Process tasks with available workers."""
        while not self.task_queue.empty():
            # Find available worker
            available_worker = None
            for worker in self.workers:
                if not worker.is_active:
                    available_worker = worker
                    break
            
            if not available_worker:
                break  # No available workers
            
            # Get next task
            task = self.task_queue.get()
            if not task:
                break
            
            # Update metrics
            self.execution_metrics.queue_size = self.task_queue.size()
            
            # Execute task
            asyncio.create_task(available_worker.execute_task(task))
    
    async def _adaptive_scaling(self):
        """Adaptively scale workers based on load."""
        try:
            queue_size = self.task_queue.size()
            active_workers = sum(1 for w in self.workers if w.is_active)
            total_workers = len(self.workers)
            
            # Scale up if queue is growing
            if queue_size > total_workers * 2 and total_workers < self.config.max_workers * 2:
                await self._create_workers(1)
                logger.info("Scaled up: added 1 worker")
            
            # Scale down if workers are idle
            elif queue_size < total_workers // 2 and total_workers > 1:
                # Remove idle workers
                idle_workers = [w for w in self.workers if w.is_idle(self.config.worker_idle_timeout)]
                if idle_workers:
                    worker_to_remove = idle_workers[0]
                    self.workers.remove(worker_to_remove)
                    self.execution_metrics.active_workers = len(self.workers)
                    logger.info("Scaled down: removed 1 idle worker")
                    
        except Exception as e:
            logger.error(f"Error in adaptive scaling: {e}")
    
    async def _metrics_update_loop(self):
        """Background loop for updating metrics."""
        while self.status == ModuleStatus.ACTIVE:
            try:
                await asyncio.sleep(5.0)  # Update every 5 seconds
                await self._update_metrics()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in metrics update loop: {e}")
    
    async def _update_metrics(self):
        """Update execution metrics."""
        try:
            # Calculate throughput
            if self.execution_metrics.total_tasks > 0:
                uptime = self.get_uptime()
                if uptime > 0:
                    self.execution_metrics.throughput_tasks_per_second = (
                        self.execution_metrics.completed_tasks / uptime
                    )
            
            # Calculate worker utilization
            active_workers = sum(1 for w in self.workers if w.is_active)
            total_workers = len(self.workers)
            if total_workers > 0:
                self.execution_metrics.worker_utilization = active_workers / total_workers
            
            # Update average execution time
            completed_tasks = [t for t in self._task_registry.values() if t.status == TaskStatus.COMPLETED]
            if completed_tasks:
                total_time = sum(t.get_execution_time() for t in completed_tasks)
                self.execution_metrics.average_execution_time = total_time / len(completed_tasks)
            
            # Update average queue wait time
            started_tasks = [t for t in self._task_registry.values() if t.started_at]
            if started_tasks:
                total_wait_time = sum(t.get_queue_wait_time() for t in started_tasks)
                self.execution_metrics.average_queue_wait_time = total_wait_time / len(started_tasks)
                
        except Exception as e:
            logger.error(f"Error updating metrics: {e}")
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get module metrics."""
        return {
            "module": "execution",
            "status": self.status.value,
            "execution_metrics": {
                "total_tasks": self.execution_metrics.total_tasks,
                "completed_tasks": self.execution_metrics.completed_tasks,
                "failed_tasks": self.execution_metrics.failed_tasks,
                "cancelled_tasks": self.execution_metrics.cancelled_tasks,
                "timed_out_tasks": self.execution_metrics.timed_out_tasks,
                "active_workers": self.execution_metrics.active_workers,
                "queue_size": self.execution_metrics.queue_size,
                "average_execution_time": self.execution_metrics.average_execution_time,
                "average_queue_wait_time": self.execution_metrics.average_queue_wait_time,
                "throughput_tasks_per_second": self.execution_metrics.throughput_tasks_per_second,
                "worker_utilization": self.execution_metrics.worker_utilization
            }
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Check module health."""
        try:
            health_status = "healthy"
            issues = []
            
            # Check worker status
            if len(self.workers) == 0:
                health_status = "unhealthy"
                issues.append("No workers available")
            
            # Check queue size
            if self.task_queue.size() > self.config.max_queue_size * 0.8:
                health_status = "warning"
                issues.append(f"Queue size approaching limit: {self.task_queue.size()}")
            
            # Check worker utilization
            if self.execution_metrics.worker_utilization < 0.1:
                health_status = "warning"
                issues.append("Low worker utilization")
            
            # Check error rate
            if self.execution_metrics.total_tasks > 0:
                error_rate = self.execution_metrics.failed_tasks / self.execution_metrics.total_tasks
                if error_rate > 0.1:  # More than 10% error rate
                    health_status = "warning"
                    issues.append(f"High error rate: {error_rate:.2%}")
            
            return {
                "status": health_status,
                "issues": issues,
                "active_workers": len([w for w in self.workers if w.is_active]),
                "total_workers": len(self.workers),
                "queue_size": self.task_queue.size(),
                "uptime": self.get_uptime()
            }
            
        except Exception as e:
            return {
                "status": "error",
                "issues": [f"Health check failed: {e}"],
                "error": str(e)
            }

# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

def create_execution_module(**kwargs) -> ExecutionModule:
    """Create an Execution Module instance."""
    config = ExecutionModuleConfig(**kwargs)
    return ExecutionModule(config)

def create_execution_module_with_defaults() -> ExecutionModule:
    """Create an Execution Module with default configurations."""
    return create_execution_module(
        max_workers=8,
        max_process_workers=4,
        default_timeout=60.0,
        execution_strategy=ExecutionStrategy.ADAPTIVE,
        enable_priority_queue=True,
        enable_load_balancing=True,
        enable_adaptive_scaling=True,
        worker_idle_timeout=300.0,
        max_queue_size=1000,
        retry_attempts=3,
        retry_delay=1.0
    )

# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "ExecutionModule",
    "ExecutionModuleConfig",
    "ExecutionMetrics",
    "TaskPriority",
    "TaskStatus",
    "ExecutionStrategy",
    "Task",
    "PriorityQueue",
    "Worker",
    "create_execution_module",
    "create_execution_module_with_defaults"
]
