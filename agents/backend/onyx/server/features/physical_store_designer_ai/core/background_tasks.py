"""
Background task processing utilities
"""

import asyncio
import threading
from typing import Callable, Any, Dict, Optional, List
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field
from queue import Queue, Empty
import time

from .logging_config import get_logger

logger = get_logger(__name__)


class TaskStatus(Enum):
    """Task status enumeration"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class BackgroundTask:
    """Represents a background task"""
    task_id: str
    func: Callable
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    retries: int = 0
    max_retries: int = 0


class TaskQueue:
    """Simple task queue for background processing"""
    
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.queue: Queue = Queue()
        self.tasks: Dict[str, BackgroundTask] = {}
        self.workers: List[threading.Thread] = []
        self.running = False
        self._lock = threading.Lock()
    
    def start(self):
        """Start worker threads"""
        if self.running:
            return
        
        self.running = True
        for i in range(self.max_workers):
            worker = threading.Thread(target=self._worker, daemon=True, name=f"TaskWorker-{i}")
            worker.start()
            self.workers.append(worker)
        logger.info(f"Started {self.max_workers} task workers")
    
    def stop(self):
        """Stop worker threads"""
        self.running = False
        # Add None to queue to signal workers to stop
        for _ in range(self.max_workers):
            self.queue.put(None)
        
        for worker in self.workers:
            worker.join(timeout=5)
        self.workers.clear()
        logger.info("Stopped task workers")
    
    def add_task(
        self,
        task_id: str,
        func: Callable,
        *args,
        max_retries: int = 0,
        **kwargs
    ) -> BackgroundTask:
        """Add task to queue"""
        task = BackgroundTask(
            task_id=task_id,
            func=func,
            args=args,
            kwargs=kwargs,
            max_retries=max_retries
        )
        
        with self._lock:
            self.tasks[task_id] = task
        
        self.queue.put(task)
        logger.debug(f"Added task {task_id} to queue")
        return task
    
    def get_task(self, task_id: str) -> Optional[BackgroundTask]:
        """Get task by ID"""
        with self._lock:
            return self.tasks.get(task_id)
    
    def _worker(self):
        """Worker thread function"""
        while self.running:
            try:
                task = self.queue.get(timeout=1)
                if task is None:
                    break
                
                self._execute_task(task)
                self.queue.task_done()
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Worker error: {e}", exc_info=True)
    
    def _execute_task(self, task: BackgroundTask):
        """Execute a task"""
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.now()
        
        try:
            # Execute function
            if asyncio.iscoroutinefunction(task.func):
                # For async functions, run in event loop
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(task.func(*task.args, **task.kwargs))
                loop.close()
            else:
                result = task.func(*task.args, **task.kwargs)
            
            task.result = result
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()
            logger.info(f"Task {task.task_id} completed successfully")
        
        except Exception as e:
            error_msg = str(e)
            task.error = error_msg
            
            # Retry logic
            if task.retries < task.max_retries:
                task.retries += 1
                task.status = TaskStatus.PENDING
                logger.warning(f"Task {task.task_id} failed, retrying ({task.retries}/{task.max_retries}): {e}")
                time.sleep(2 ** task.retries)  # Exponential backoff
                self.queue.put(task)
            else:
                task.status = TaskStatus.FAILED
                task.completed_at = datetime.now()
                logger.error(f"Task {task.task_id} failed after {task.retries} retries: {e}")


class AsyncTaskExecutor:
    """Async task executor for background tasks"""
    
    def __init__(self):
        self.tasks: Dict[str, asyncio.Task] = {}
        self.results: Dict[str, Any] = {}
        self.errors: Dict[str, Exception] = {}
    
    async def execute(
        self,
        task_id: str,
        coro: Callable,
        *args,
        **kwargs
    ) -> str:
        """Execute async task"""
        async def _wrapper():
            try:
                result = await coro(*args, **kwargs)
                self.results[task_id] = result
                return result
            except Exception as e:
                self.errors[task_id] = e
                logger.error(f"Async task {task_id} failed: {e}", exc_info=True)
                raise
        
        task = asyncio.create_task(_wrapper())
        self.tasks[task_id] = task
        return task_id
    
    def get_result(self, task_id: str) -> Optional[Any]:
        """Get task result"""
        return self.results.get(task_id)
    
    def get_error(self, task_id: str) -> Optional[Exception]:
        """Get task error"""
        return self.errors.get(task_id)
    
    def is_complete(self, task_id: str) -> bool:
        """Check if task is complete"""
        task = self.tasks.get(task_id)
        if task is None:
            return False
        return task.done()
    
    def cancel(self, task_id: str) -> bool:
        """Cancel task"""
        task = self.tasks.get(task_id)
        if task and not task.done():
            task.cancel()
            return True
        return False


# Global task queue instance
_task_queue: Optional[TaskQueue] = None


def get_task_queue() -> TaskQueue:
    """Get global task queue instance"""
    global _task_queue
    if _task_queue is None:
        _task_queue = TaskQueue()
        _task_queue.start()
    return _task_queue


def submit_background_task(
    task_id: str,
    func: Callable,
    *args,
    max_retries: int = 0,
    **kwargs
) -> BackgroundTask:
    """Submit a background task"""
    queue = get_task_queue()
    return queue.add_task(task_id, func, *args, max_retries=max_retries, **kwargs)


def get_task_status(task_id: str) -> Optional[Dict[str, Any]]:
    """Get task status"""
    queue = get_task_queue()
    task = queue.get_task(task_id)
    if task is None:
        return None
    
    return {
        "task_id": task.task_id,
        "status": task.status.value,
        "created_at": task.created_at.isoformat(),
        "started_at": task.started_at.isoformat() if task.started_at else None,
        "completed_at": task.completed_at.isoformat() if task.completed_at else None,
        "retries": task.retries,
        "error": task.error,
        "has_result": task.result is not None
    }

