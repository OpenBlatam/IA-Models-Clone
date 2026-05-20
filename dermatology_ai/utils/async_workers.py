"""
Async Workers for background task processing
Lightweight alternative to Celery for serverless environments
"""

import asyncio
import uuid
from typing import Callable, Any, Optional, Dict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Task status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Task:
    """Background task"""
    id: str
    func: Callable
    args: tuple
    kwargs: dict
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    priority: int = 0  # Higher priority = executed first


class AsyncWorkerPool:
    """
    Lightweight async worker pool for background tasks.
    Optimized for serverless environments where Celery might be overkill.
    """
    
    def __init__(self, max_workers: int = 4, queue_size: int = 1000):
        self.max_workers = max_workers
        self.queue_size = queue_size
        self.tasks: Dict[str, Task] = {}
        self.task_queue: asyncio.PriorityQueue = asyncio.PriorityQueue(maxsize=queue_size)
        self.workers: list = []
        self.running = False
        self._lock = asyncio.Lock()
    
    async def start(self):
        """Start worker pool"""
        if self.running:
            return
        
        self.running = True
        self.workers = [
            asyncio.create_task(self._worker(f"worker-{i}"))
            for i in range(self.max_workers)
        ]
        logger.info(f"Started {self.max_workers} async workers")
    
    async def stop(self, timeout: float = 30.0):
        """Stop worker pool gracefully"""
        if not self.running:
            return
        
        self.running = False
        
        # Wait for queue to empty (with timeout)
        try:
            await asyncio.wait_for(self._drain_queue(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning("Timeout waiting for queue to drain")
        
        # Cancel workers
        for worker in self.workers:
            worker.cancel()
        
        # Wait for workers to finish
        await asyncio.gather(*self.workers, return_exceptions=True)
        logger.info("Worker pool stopped")
    
    async def _drain_queue(self):
        """Drain task queue"""
        while not self.task_queue.empty():
            await asyncio.sleep(0.1)
    
    async def _worker(self, name: str):
        """Worker coroutine"""
        logger.debug(f"Worker {name} started")
        
        while self.running:
            try:
                # Get task from queue (with timeout to allow checking running flag)
                try:
                    priority, task_id = await asyncio.wait_for(
                        self.task_queue.get(),
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                task = self.tasks.get(task_id)
                if not task:
                    continue
                
                # Execute task
                task.status = TaskStatus.RUNNING
                task.started_at = datetime.utcnow()
                
                try:
                    if asyncio.iscoroutinefunction(task.func):
                        result = await task.func(*task.args, **task.kwargs)
                    else:
                        # Run sync function in executor
                        loop = asyncio.get_event_loop()
                        result = await loop.run_in_executor(
                            None, task.func, *task.args, **task.kwargs
                        )
                    
                    task.result = result
                    task.status = TaskStatus.COMPLETED
                    task.completed_at = datetime.utcnow()
                    
                except Exception as e:
                    task.status = TaskStatus.FAILED
                    task.error = str(e)
                    task.completed_at = datetime.utcnow()
                    logger.error(f"Task {task_id} failed: {e}", exc_info=True)
                
                finally:
                    self.task_queue.task_done()
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker {name} error: {e}", exc_info=True)
                await asyncio.sleep(1)  # Brief pause before retrying
        
        logger.debug(f"Worker {name} stopped")
    
    async def submit(
        self,
        func: Callable,
        *args,
        priority: int = 0,
        **kwargs
    ) -> str:
        """
        Submit task to queue
        
        Args:
            func: Function to execute
            *args: Function arguments
            priority: Task priority (higher = executed first)
            **kwargs: Function keyword arguments
            
        Returns:
            Task ID
        """
        if not self.running:
            await self.start()
        
        task_id = str(uuid.uuid4())
        task = Task(
            id=task_id,
            func=func,
            args=args,
            kwargs=kwargs,
            priority=priority
        )
        
        async with self._lock:
            self.tasks[task_id] = task
        
        # Add to queue (negative priority for max-heap behavior)
        await self.task_queue.put((-priority, task_id))
        
        logger.debug(f"Task {task_id} submitted")
        return task_id
    
    async def get_task(self, task_id: str) -> Optional[Task]:
        """Get task by ID"""
        return self.tasks.get(task_id)
    
    async def get_task_result(self, task_id: str, timeout: Optional[float] = None) -> Any:
        """
        Wait for task to complete and get result
        
        Args:
            task_id: Task ID
            timeout: Optional timeout in seconds
            
        Returns:
            Task result
            
        Raises:
            TimeoutError: If timeout exceeded
            Exception: If task failed
        """
        task = self.tasks.get(task_id)
        if not task:
            raise ValueError(f"Task {task_id} not found")
        
        start_time = asyncio.get_event_loop().time()
        
        while task.status in (TaskStatus.PENDING, TaskStatus.RUNNING):
            if timeout:
                elapsed = asyncio.get_event_loop().time() - start_time
                if elapsed >= timeout:
                    raise asyncio.TimeoutError(f"Task {task_id} timeout after {timeout}s")
            
            await asyncio.sleep(0.1)
        
        if task.status == TaskStatus.COMPLETED:
            return task.result
        elif task.status == TaskStatus.FAILED:
            raise Exception(f"Task failed: {task.error}")
        else:
            raise Exception(f"Task {task_id} in unexpected state: {task.status}")
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel pending task"""
        task = self.tasks.get(task_id)
        if not task:
            return False
        
        if task.status == TaskStatus.PENDING:
            task.status = TaskStatus.CANCELLED
            return True
        
        return False
    
    def get_stats(self) -> dict:
        """Get worker pool statistics"""
        status_counts = {}
        for task in self.tasks.values():
            status_counts[task.status.value] = status_counts.get(task.status.value, 0) + 1
        
        return {
            "workers": len(self.workers),
            "queue_size": self.task_queue.qsize(),
            "total_tasks": len(self.tasks),
            "status_counts": status_counts,
            "running": self.running,
        }


# Global worker pool instance
_worker_pool: Optional[AsyncWorkerPool] = None


def get_worker_pool(max_workers: int = 4) -> AsyncWorkerPool:
    """Get or create global worker pool"""
    global _worker_pool
    if _worker_pool is None:
        _worker_pool = AsyncWorkerPool(max_workers=max_workers)
    return _worker_pool

