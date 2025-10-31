from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import asyncio
import logging
import time
import threading
from typing import Any, Dict, List, Optional, Callable, Union, Coroutine
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import heapq
import uuid
import psutil
import gc
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from functools import partial, wraps
import signal
import weakref
from contextlib import asynccontextmanager

        import signal
from typing import Any, List, Dict, Optional
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TaskPriority(Enum):
    LOW = 3
    NORMAL = 2
    HIGH = 1
    CRITICAL = 0

class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"

class TaskType(Enum):
    CPU_INTENSIVE = "cpu_intensive"
    IO_INTENSIVE = "io_intensive"
    MEMORY_INTENSIVE = "memory_intensive"
    MIXED = "mixed"

@dataclass
class Task:
    id: str
    func: Callable
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.NORMAL
    task_type: TaskType = TaskType.MIXED
    timeout: Optional[float] = None
    retries: int = 0
    max_retries: int = 3
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error: Optional[Exception] = None
    progress: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ProcessorConfig:
    max_workers: int = min(mp.cpu_count(), 8)
    max_thread_workers: int = 20
    max_process_workers: int = 4
    queue_size: int = 1000
    enable_priority_queue: bool = True
    enable_task_pooling: bool = True
    enable_auto_scaling: bool = True
    enable_monitoring: bool = True
    task_timeout: float = 300.0  # 5 minutes
    cleanup_interval: float = 60.0  # 1 minute
    memory_threshold: float = 0.8  # 80% memory usage
    cpu_threshold: float = 0.9  # 90% CPU usage

class PriorityQueue:
    """Thread-safe priority queue implementation"""
    
    def __init__(self) -> Any:
        self._queue = []
        self._lock = threading.RLock()
        self._not_empty = threading.Condition(self._lock)
    
    def put(self, task: Task) -> None:
        with self._lock:
            # Priority is negated because heapq is a min-heap
            heapq.heappush(self._queue, (-task.priority.value, task.created_at, task))
            self._not_empty.notify()
    
    def get(self) -> Optional[Task]:
        with self._not_empty:
            while not self._queue:
                self._not_empty.wait()
            
            with self._lock:
                if self._queue:
                    return heapq.heappop(self._queue)[2]
                return None
    
    def empty(self) -> bool:
        with self._lock:
            return len(self._queue) == 0
    
    def size(self) -> int:
        with self._lock:
            return len(self._queue)

class TaskPool:
    """Task pool for managing different types of tasks"""
    
    def __init__(self, config: ProcessorConfig):
        
    """__init__ function."""
self.config = config
        self.tasks: Dict[str, Task] = {}
        self.task_lock = threading.RLock()
        
        # Priority queues for different task types
        self.cpu_queue = PriorityQueue()
        self.io_queue = PriorityQueue()
        self.memory_queue = PriorityQueue()
        self.mixed_queue = PriorityQueue()
        
        # Statistics
        self.stats = {
            'total_tasks': 0,
            'completed_tasks': 0,
            'failed_tasks': 0,
            'cancelled_tasks': 0,
            'avg_execution_time': 0.0,
            'queue_sizes': defaultdict(int)
        }
    
    def add_task(self, task: Task) -> None:
        with self.task_lock:
            self.tasks[task.id] = task
            
            # Add to appropriate queue
            if self.config.enable_priority_queue:
                if task.task_type == TaskType.CPU_INTENSIVE:
                    self.cpu_queue.put(task)
                elif task.task_type == TaskType.IO_INTENSIVE:
                    self.io_queue.put(task)
                elif task.task_type == TaskType.MEMORY_INTENSIVE:
                    self.memory_queue.put(task)
                else:
                    self.mixed_queue.put(task)
            
            self.stats['total_tasks'] += 1
            self._update_queue_stats()
    
    def get_next_task(self, task_type: TaskType = None) -> Optional[Task]:
        if not self.config.enable_priority_queue:
            return None
        
        if task_type == TaskType.CPU_INTENSIVE:
            return self.cpu_queue.get()
        elif task_type == TaskType.IO_INTENSIVE:
            return self.io_queue.get()
        elif task_type == TaskType.MEMORY_INTENSIVE:
            return self.memory_queue.get()
        else:
            # Try mixed queue first, then others
            task = self.mixed_queue.get()
            if task is None:
                task = self.io_queue.get()
            if task is None:
                task = self.cpu_queue.get()
            if task is None:
                task = self.memory_queue.get()
            return task
    
    def update_task(self, task_id: str, **kwargs) -> bool:
        with self.task_lock:
            if task_id in self.tasks:
                task = self.tasks[task_id]
                for key, value in kwargs.items():
                    if hasattr(task, key):
                        setattr(task, key, value)
                return True
            return False
    
    def get_task(self, task_id: str) -> Optional[Task]:
        with self.task_lock:
            return self.tasks.get(task_id)
    
    def remove_task(self, task_id: str) -> bool:
        with self.task_lock:
            if task_id in self.tasks:
                del self.tasks[task_id]
                return True
            return False
    
    def _update_queue_stats(self) -> None:
        self.stats['queue_sizes'] = {
            'cpu': self.cpu_queue.size(),
            'io': self.io_queue.size(),
            'memory': self.memory_queue.size(),
            'mixed': self.mixed_queue.size()
        }
    
    def get_stats(self) -> Dict[str, Any]:
        with self.task_lock:
            self._update_queue_stats()
            return self.stats.copy()

class OptimizedAsyncProcessor:
    def __init__(self, config: ProcessorConfig = None):
        
    """__init__ function."""
self.config = config or ProcessorConfig()
        
        # Task management
        self.task_pool = TaskPool(self.config)
        
        # Thread and process pools
        self.thread_pool = ThreadPoolExecutor(
            max_workers=self.config.max_thread_workers,
            thread_name_prefix="AsyncProcessor"
        )
        self.process_pool = ProcessPoolExecutor(
            max_workers=self.config.max_process_workers
        )
        
        # Worker management
        self.workers: Dict[str, asyncio.Task] = {}
        self.worker_lock = asyncio.Lock()
        
        # Monitoring
        self.monitoring_task: Optional[asyncio.Task] = None
        self.cleanup_task: Optional[asyncio.Task] = None
        
        # Performance tracking
        self.performance_stats = {
            'active_workers': 0,
            'queue_size': 0,
            'memory_usage': 0,
            'cpu_usage': 0,
            'throughput': 0
        }
        
        # Shutdown flag
        self._shutdown = False
        
        logger.info(f"OptimizedAsyncProcessor initialized with {self.config.max_workers} workers")

    async def start(self) -> None:
        """Start the processor"""
        try:
            # Start monitoring
            if self.config.enable_monitoring:
                self.monitoring_task = asyncio.create_task(self._monitor_performance())
            
            # Start cleanup task
            self.cleanup_task = asyncio.create_task(self._cleanup_old_tasks())
            
            # Start workers
            await self._start_workers()
            
            logger.info("OptimizedAsyncProcessor started successfully")
            
        except Exception as e:
            logger.error(f"Error starting processor: {e}")
            raise

    async def stop(self) -> None:
        """Stop the processor"""
        try:
            self._shutdown = True
            
            # Cancel monitoring and cleanup tasks
            if self.monitoring_task:
                self.monitoring_task.cancel()
            if self.cleanup_task:
                self.cleanup_task.cancel()
            
            # Cancel all workers
            async with self.worker_lock:
                for worker in self.workers.values():
                    worker.cancel()
                
                # Wait for workers to finish
                if self.workers:
                    await asyncio.gather(*self.workers.values(), return_exceptions=True)
            
            # Shutdown thread and process pools
            self.thread_pool.shutdown(wait=True)
            self.process_pool.shutdown(wait=True)
            
            logger.info("OptimizedAsyncProcessor stopped")
            
        except Exception as e:
            logger.error(f"Error stopping processor: {e}")

    async def submit_task(self, 
                         func: Callable,
                         *args,
                         priority: TaskPriority = TaskPriority.NORMAL,
                         task_type: TaskType = TaskType.MIXED,
                         timeout: Optional[float] = None,
                         retries: int = 0,
                         **kwargs) -> str:
        """Submit a task for execution"""
        task_id = str(uuid.uuid4())
        
        task = Task(
            id=task_id,
            func=func,
            args=args,
            kwargs=kwargs,
            priority=priority,
            task_type=task_type,
            timeout=timeout or self.config.task_timeout,
            retries=retries,
            max_retries=self.config.max_retries
        )
        
        self.task_pool.add_task(task)
        
        logger.debug(f"Task {task_id} submitted with priority {priority.value}")
        return task_id

    async def get_task_result(self, task_id: str, timeout: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """Get the result of a completed task"""
        start_time = time.time()
        
        while True:
            task = self.task_pool.get_task(task_id)
            
            if task is None:
                raise ValueError(f"Task {task_id} not found")
            
            if task.status == TaskStatus.COMPLETED:
                return task.result
            
            elif task.status == TaskStatus.FAILED:
                raise task.error or Exception(f"Task {task_id} failed")
            
            elif task.status == TaskStatus.CANCELLED:
                raise asyncio.CancelledError(f"Task {task_id} was cancelled")
            
            elif task.status == TaskStatus.TIMEOUT:
                raise asyncio.TimeoutError(f"Task {task_id} timed out")
            
            # Check timeout
            if timeout and (time.time() - start_time) > timeout:
                raise asyncio.TimeoutError(f"Timeout waiting for task {task_id}")
            
            # Wait before checking again
            await asyncio.sleep(0.1)

    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending task"""
        task = self.task_pool.get_task(task_id)
        
        if task and task.status == TaskStatus.PENDING:
            task.status = TaskStatus.CANCELLED
            self.task_pool.stats['cancelled_tasks'] += 1
            logger.info(f"Task {task_id} cancelled")
            return True
        
        return False

    async def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """Get the status of a task"""
        task = self.task_pool.get_task(task_id)
        return task.status if task else None

    async def get_all_tasks(self, status: Optional[TaskStatus] = None) -> List[Task]:
        """Get all tasks, optionally filtered by status"""
        with self.task_pool.task_lock:
            tasks = list(self.task_pool.tasks.values())
            
            if status:
                tasks = [task for task in tasks if task.status == status]
            
            return tasks

    async def _start_workers(self) -> None:
        """Start worker tasks"""
        async with self.worker_lock:
            for i in range(self.config.max_workers):
                worker_id = f"worker_{i}"
                worker_task = asyncio.create_task(
                    self._worker_loop(worker_id),
                    name=worker_id
                )
                self.workers[worker_id] = worker_task

    async def _worker_loop(self, worker_id: str) -> None:
        """Main worker loop"""
        logger.debug(f"Worker {worker_id} started")
        
        while not self._shutdown:
            try:
                # Get next task
                task = self.task_pool.get_next_task()
                
                if task is None:
                    # No tasks available, wait a bit
                    await asyncio.sleep(0.1)
                    continue
                
                # Execute task
                await self._execute_task(task)
                
            except asyncio.CancelledError:
                logger.debug(f"Worker {worker_id} cancelled")
                break
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
                await asyncio.sleep(1)  # Wait before retrying
        
        logger.debug(f"Worker {worker_id} stopped")

    async def _execute_task(self, task: Task) -> None:
        """Execute a single task"""
        try:
            # Update task status
            task.status = TaskStatus.RUNNING
            task.started_at = time.time()
            
            logger.debug(f"Executing task {task.id}")
            
            # Execute based on task type
            if task.task_type == TaskType.CPU_INTENSIVE:
                result = await self._execute_cpu_task(task)
            elif task.task_type == TaskType.IO_INTENSIVE:
                result = await self._execute_io_task(task)
            elif task.task_type == TaskType.MEMORY_INTENSIVE:
                result = await self._execute_memory_task(task)
            else:
                result = await self._execute_mixed_task(task)
            
            # Update task with result
            task.status = TaskStatus.COMPLETED
            task.completed_at = time.time()
            task.result = result
            task.progress = 1.0
            
            self.task_pool.stats['completed_tasks'] += 1
            
            logger.debug(f"Task {task.id} completed successfully")
            
        except asyncio.TimeoutError:
            task.status = TaskStatus.TIMEOUT
            task.error = asyncio.TimeoutError(f"Task {task.id} timed out")
            logger.warning(f"Task {task.id} timed out")
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = e
            
            # Retry logic
            if task.retries < task.max_retries:
                task.retries += 1
                task.status = TaskStatus.PENDING
                task.started_at = None
                task.completed_at = None
                task.error = None
                
                # Re-add to queue
                self.task_pool.add_task(task)
                logger.info(f"Retrying task {task.id} (attempt {task.retries})")
            else:
                self.task_pool.stats['failed_tasks'] += 1
                logger.error(f"Task {task.id} failed after {task.max_retries} retries: {e}")

    async def _execute_cpu_task(self, task: Task) -> Any:
        """Execute CPU-intensive task in process pool"""
        loop = asyncio.get_event_loop()
        
        # Run in process pool for CPU-intensive tasks
        result = await loop.run_in_executor(
            self.process_pool,
            self._run_with_timeout,
            task.func,
            task.timeout,
            *task.args,
            **task.kwargs
        )
        
        return result

    async def _execute_io_task(self, task: Task) -> Any:
        """Execute I/O-intensive task in thread pool"""
        loop = asyncio.get_event_loop()
        
        # Run in thread pool for I/O-intensive tasks
        result = await loop.run_in_executor(
            self.thread_pool,
            self._run_with_timeout,
            task.func,
            task.timeout,
            *task.args,
            **task.kwargs
        )
        
        return result

    async def _execute_memory_task(self, task: Task) -> Any:
        """Execute memory-intensive task with garbage collection"""
        # Force garbage collection before memory-intensive task
        gc.collect()
        
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self.thread_pool,
            self._run_with_timeout,
            task.func,
            task.timeout,
            *task.args,
            **task.kwargs
        )
        
        # Force garbage collection after memory-intensive task
        gc.collect()
        
        return result

    async def _execute_mixed_task(self, task: Task) -> Any:
        """Execute mixed task (default to thread pool)"""
        return await self._execute_io_task(task)

    def _run_with_timeout(self, func: Callable, timeout: float, *args, **kwargs) -> Any:
        """Run function with timeout"""
        
        def timeout_handler(signum, frame) -> Any:
            raise TimeoutError("Function execution timed out")
        
        # Set up timeout handler
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(int(timeout))
        
        try:
            result = func(*args, **kwargs)
            signal.alarm(0)  # Cancel alarm
            return result
        finally:
            signal.signal(signal.SIGALRM, old_handler)

    async def _monitor_performance(self) -> None:
        """Monitor system performance and adjust scaling"""
        while not self._shutdown:
            try:
                # Get system metrics
                memory_usage = psutil.virtual_memory().percent / 100
                cpu_usage = psutil.cpu_percent(interval=1) / 100
                
                # Update performance stats
                self.performance_stats.update({
                    'active_workers': len(self.workers),
                    'queue_size': sum(self.task_pool.stats['queue_sizes'].values()),
                    'memory_usage': memory_usage,
                    'cpu_usage': cpu_usage
                })
                
                # Auto-scaling logic
                if self.config.enable_auto_scaling:
                    await self._adjust_scaling(memory_usage, cpu_usage)
                
                await asyncio.sleep(10)  # Monitor every 10 seconds
                
            except Exception as e:
                logger.error(f"Performance monitoring error: {e}")
                await asyncio.sleep(10)

    async def _adjust_scaling(self, memory_usage: float, cpu_usage: float) -> None:
        """Adjust worker scaling based on system metrics"""
        current_workers = len(self.workers)
        
        # Scale up if system is underutilized and there are queued tasks
        if (memory_usage < self.config.memory_threshold and 
            cpu_usage < self.config.cpu_threshold and
            self.performance_stats['queue_size'] > current_workers):
            
            # Add worker
            worker_id = f"worker_{current_workers}"
            worker_task = asyncio.create_task(
                self._worker_loop(worker_id),
                name=worker_id
            )
            
            async with self.worker_lock:
                self.workers[worker_id] = worker_task
            
            logger.info(f"Scaled up: added worker {worker_id}")
        
        # Scale down if system is overutilized
        elif (memory_usage > self.config.memory_threshold or 
              cpu_usage > self.config.cpu_threshold) and current_workers > 1:
            
            # Remove worker
            async with self.worker_lock:
                if self.workers:
                    worker_id, worker_task = self.workers.popitem()
                    worker_task.cancel()
            
            logger.info(f"Scaled down: removed worker {worker_id}")

    async def _cleanup_old_tasks(self) -> None:
        """Clean up old completed/failed tasks"""
        while not self._shutdown:
            try:
                current_time = time.time()
                tasks_to_remove = []
                
                # Find old tasks to remove
                for task in self.task_pool.tasks.values():
                    if (task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED] and
                        current_time - task.completed_at > self.config.cleanup_interval):
                        tasks_to_remove.append(task.id)
                
                # Remove old tasks
                for task_id in tasks_to_remove:
                    self.task_pool.remove_task(task_id)
                
                if tasks_to_remove:
                    logger.debug(f"Cleaned up {len(tasks_to_remove)} old tasks")
                
                await asyncio.sleep(self.config.cleanup_interval)
                
            except Exception as e:
                logger.error(f"Cleanup error: {e}")
                await asyncio.sleep(self.config.cleanup_interval)

    def get_stats(self) -> Dict[str, Any]:
        """Get processor statistics"""
        stats = {
            'processor': self.performance_stats.copy(),
            'tasks': self.task_pool.get_stats(),
            'workers': {
                'active': len(self.workers),
                'max_workers': self.config.max_workers
            }
        }
        return stats

# Decorator for easy task submission
def async_task(priority: TaskPriority = TaskPriority.NORMAL,
               task_type: TaskType = TaskType.MIXED,
               timeout: Optional[float] = None):
    """Decorator to make a function an async task"""
    def decorator(func) -> Any:
        @wraps(func)
        async def wrapper(processor: OptimizedAsyncProcessor, *args, **kwargs):
            
    """wrapper function."""
task_id = await processor.submit_task(
                func, *args,
                priority=priority,
                task_type=task_type,
                timeout=timeout,
                **kwargs
            )
            return await processor.get_task_result(task_id)
        return wrapper
    return decorator

# Usage example
async def main():
    
    """main function."""
# Initialize processor
    config = ProcessorConfig(
        max_workers=4,
        max_thread_workers=10,
        max_process_workers=2,
        enable_priority_queue=True,
        enable_auto_scaling=True,
        enable_monitoring=True
    )
    
    processor = OptimizedAsyncProcessor(config)
    
    try:
        # Start processor
        await processor.start()
        
        # Example CPU-intensive task
        def cpu_intensive_task(n: int) -> int:
            result = 0
            for i in range(n):
                result += i ** 2
            return result
        
        # Example I/O-intensive task
        async def io_intensive_task(delay: float) -> str:
            await asyncio.sleep(delay)
            return f"IO task completed after {delay}s"
        
        # Submit tasks
        task1_id = await processor.submit_task(
            cpu_intensive_task, 1000000,
            priority=TaskPriority.HIGH,
            task_type=TaskType.CPU_INTENSIVE
        )
        
        task2_id = await processor.submit_task(
            lambda: io_intensive_task(2),
            priority=TaskPriority.NORMAL,
            task_type=TaskType.IO_INTENSIVE
        )
        
        # Get results
        result1 = await processor.get_task_result(task1_id)
        result2 = await processor.get_task_result(task2_id)
        
        print(f"CPU task result: {result1}")
        print(f"IO task result: {result2}")
        
        # Get statistics
        stats = processor.get_stats()
        print(f"Processor stats: {stats}")
        
    finally:
        await processor.stop()

match __name__:
    case "__main__":
    asyncio.run(main()) 