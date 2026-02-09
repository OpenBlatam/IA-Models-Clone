"""
Queue Manager
=============

Queue management utilities for task queuing.
"""

import asyncio
import logging
from typing import Callable, Any, Optional, Dict
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from collections import deque

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    """Task status"""
    PENDING = "pending"
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Task:
    """Task in queue"""
    task_id: str
    func: Callable
    args: tuple
    kwargs: dict
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error: Optional[str] = None
    created_at: datetime = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


class TaskQueue:
    """
    Task queue for managing async tasks.
    
    Features:
    - Task queuing
    - Priority support
    - Concurrency control
    - Task tracking
    """
    
    def __init__(self, max_concurrent: int = 5):
        """
        Initialize task queue.
        
        Args:
            max_concurrent: Maximum concurrent tasks
        """
        self.max_concurrent = max_concurrent
        self.queue: asyncio.Queue = asyncio.Queue()
        self.tasks: Dict[str, Task] = {}
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.worker_task: Optional[asyncio.Task] = None
        self.running = False
    
    async def start(self):
        """Start queue worker"""
        if not self.running:
            self.running = True
            self.worker_task = asyncio.create_task(self._worker())
            logger.info("Task queue started")
    
    async def stop(self):
        """Stop queue worker"""
        self.running = False
        if self.worker_task:
            await self.worker_task
        logger.info("Task queue stopped")
    
    async def add_task(
        self,
        task_id: str,
        func: Callable,
        *args,
        **kwargs
    ) -> str:
        """
        Add task to queue.
        
        Args:
            task_id: Unique task ID
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Task ID
        """
        task = Task(
            task_id=task_id,
            func=func,
            args=args,
            kwargs=kwargs
        )
        
        self.tasks[task_id] = task
        await self.queue.put(task)
        logger.debug(f"Task {task_id} added to queue")
        
        return task_id
    
    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Get task status.
        
        Args:
            task_id: Task ID
            
        Returns:
            Dictionary with task status or None if not found
        """
        if task_id not in self.tasks:
            return None
        
        task = self.tasks[task_id]
        
        return {
            "task_id": task_id,
            "status": task.status.value,
            "created_at": task.created_at.isoformat(),
            "started_at": task.started_at.isoformat() if task.started_at else None,
            "completed_at": task.completed_at.isoformat() if task.completed_at else None,
            "error": task.error,
            "has_result": task.result is not None
        }
    
    async def _worker(self):
        """Queue worker loop"""
        while self.running:
            try:
                task = await asyncio.wait_for(self.queue.get(), timeout=1.0)
                
                # Process task with semaphore
                asyncio.create_task(self._process_task(task))
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Queue worker error: {e}", exc_info=True)
    
    async def _process_task(self, task: Task):
        """Process a single task"""
        async with self.semaphore:
            task.status = TaskStatus.PROCESSING
            task.started_at = datetime.now()
            
            try:
                # Execute function
                if asyncio.iscoroutinefunction(task.func):
                    result = await task.func(*task.args, **task.kwargs)
                else:
                    result = task.func(*task.args, **task.kwargs)
                
                task.result = result
                task.status = TaskStatus.COMPLETED
                task.completed_at = datetime.now()
                
                logger.debug(f"Task {task.task_id} completed")
                
            except Exception as e:
                task.status = TaskStatus.FAILED
                task.error = str(e)
                task.completed_at = datetime.now()
                
                logger.error(f"Task {task.task_id} failed: {e}", exc_info=True)
            
            self.queue.task_done()

