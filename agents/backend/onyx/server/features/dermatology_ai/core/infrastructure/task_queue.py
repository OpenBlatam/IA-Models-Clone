"""
Async Task Queue
Background task processing with priority and retry support
"""

import asyncio
from enum import Enum
from dataclasses import dataclass, field
from typing import Callable, Any, Optional, Dict
from datetime import datetime
import uuid
import logging

logger = logging.getLogger(__name__)


class TaskPriority(Enum):
    """Task priority levels"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class TaskStatus(Enum):
    """Task status"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"


@dataclass
class Task:
    """Task definition"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    func: Callable = None
    args: tuple = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.MEDIUM
    max_retries: int = 3
    retry_delay: float = 1.0
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    retry_count: int = 0


class TaskQueue:
    """Async task queue with priority support"""
    
    def __init__(self, max_workers: int = 5):
        """
        Initialize task queue
        
        Args:
            max_workers: Maximum concurrent workers
        """
        self.max_workers = max_workers
        self.queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self.workers: list[asyncio.Task] = []
        self.tasks: Dict[str, Task] = {}
        self.running = False
        self._lock = asyncio.Lock()
    
    async def start(self):
        """Start task queue workers"""
        if self.running:
            logger.warning("Task queue already running")
            return
        
        self.running = True
        self.workers = [
            asyncio.create_task(self._worker(f"worker-{i}"))
            for i in range(self.max_workers)
        ]
        logger.info(f"Task queue started with {self.max_workers} workers")
    
    async def stop(self):
        """Stop task queue workers"""
        self.running = False
        
        # Wait for queue to empty
        await self.queue.join()
        
        # Cancel workers
        for worker in self.workers:
            worker.cancel()
        
        await asyncio.gather(*self.workers, return_exceptions=True)
        logger.info("Task queue stopped")
    
    async def enqueue(
        self,
        func: Callable,
        *args,
        priority: TaskPriority = TaskPriority.MEDIUM,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        **kwargs
    ) -> str:
        """
        Enqueue a task
        
        Args:
            func: Function to execute
            *args: Function arguments
            priority: Task priority
            max_retries: Maximum retry attempts
            retry_delay: Delay between retries (seconds)
            **kwargs: Function keyword arguments
            
        Returns:
            Task ID
        """
        task = Task(
            func=func,
            args=args,
            kwargs=kwargs,
            priority=priority,
            max_retries=max_retries,
            retry_delay=retry_delay
        )
        
        async with self._lock:
            self.tasks[task.id] = task
        
        # Priority queue uses negative priority for higher priority first
        await self.queue.put((-priority.value, task.id, task))
        logger.debug(f"Task {task.id} enqueued with priority {priority.value}")
        
        return task.id
    
    async def get_task_status(self, task_id: str) -> Optional[Task]:
        """Get task status"""
        return self.tasks.get(task_id)
    
    async def _worker(self, worker_name: str):
        """Worker coroutine"""
        logger.debug(f"Worker {worker_name} started")
        
        while self.running:
            try:
                # Get task from queue (with timeout to check running status)
                try:
                    _, task_id, task = await asyncio.wait_for(
                        self.queue.get(),
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                await self._process_task(task)
                self.queue.task_done()
                
            except asyncio.CancelledError:
                logger.debug(f"Worker {worker_name} cancelled")
                break
            except Exception as e:
                logger.error(f"Worker {worker_name} error: {e}", exc_info=True)
    
    async def _process_task(self, task: Task):
        """Process a single task"""
        task.status = TaskStatus.PROCESSING
        task.started_at = datetime.utcnow()
        
        try:
            if asyncio.iscoroutinefunction(task.func):
                result = await task.func(*task.args, **task.kwargs)
            else:
                result = task.func(*task.args, **task.kwargs)
            
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.utcnow()
            logger.debug(f"Task {task.id} completed")
            
        except Exception as e:
            task.error = str(e)
            task.retry_count += 1
            
            if task.retry_count < task.max_retries:
                task.status = TaskStatus.RETRYING
                logger.warning(
                    f"Task {task.id} failed, retrying ({task.retry_count}/{task.max_retries}): {e}"
                )
                
                # Re-enqueue with delay
                await asyncio.sleep(task.retry_delay * task.retry_count)
                await self.queue.put((-task.priority.value, task.id, task))
            else:
                task.status = TaskStatus.FAILED
                logger.error(f"Task {task.id} failed after {task.max_retries} retries: {e}")


# Global task queue instance
_task_queue: Optional[TaskQueue] = None


def get_task_queue() -> TaskQueue:
    """Get global task queue instance"""
    global _task_queue
    if _task_queue is None:
        _task_queue = TaskQueue()
    return _task_queue















