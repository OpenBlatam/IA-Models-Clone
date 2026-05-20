"""
Background task queue

This module provides a scalable background task queue for async operations.
"""

import asyncio
import heapq
from typing import Callable, Awaitable, Any, Optional, Dict
from dataclasses import dataclass, field
from datetime import datetime
import uuid

from utils.logger import logger


@dataclass
class Task:
    """Background task representation"""
    id: str
    func: Callable[[], Awaitable[Any]]
    priority: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)
    retries: int = 0
    max_retries: int = 3
    
    def __lt__(self, other):
        """Compare tasks by priority (higher priority first)"""
        if self.priority != other.priority:
            return self.priority > other.priority
        return self.created_at < other.created_at


class BackgroundTaskQueue:
    """
    Scalable background task queue
    
    Processes tasks asynchronously with configurable concurrency.
    """
    
    def __init__(self, max_workers: int = 5, max_queue_size: int = 1000):
        """
        Initialize background task queue
        
        Args:
            max_workers: Maximum concurrent workers
            max_queue_size: Maximum queue size
        """
        self.max_workers = max_workers
        self.max_queue_size = max_queue_size
        self.queue: list = []
        self.workers: list = []
        self.running = False
        self._lock = asyncio.Lock()
        self._stats = {
            "processed": 0,
            "failed": 0,
            "queued": 0
        }
    
    async def start(self) -> None:
        """Start background workers"""
        if self.running:
            return
        
        self.running = True
        for i in range(self.max_workers):
            worker = asyncio.create_task(self._worker(f"worker-{i}"))
            self.workers.append(worker)
        logger.info(f"Started {self.max_workers} background task workers")
    
    async def stop(self) -> None:
        """Stop background workers"""
        self.running = False
        await asyncio.gather(*self.workers, return_exceptions=True)
        self.workers.clear()
        logger.info("Stopped background task workers")
    
    async def enqueue(
        self,
        func: Callable[[], Awaitable[Any]],
        priority: int = 0,
        max_retries: int = 3
    ) -> str:
        """
        Enqueue a background task
        
        Args:
            func: Async function to execute
            priority: Task priority (higher = more important)
            max_retries: Maximum retry attempts
            
        Returns:
            Task ID
            
        Raises:
            RuntimeError: If queue is full
        """
        if len(self.queue) >= self.max_queue_size:
            raise RuntimeError(f"Task queue is full (max: {self.max_queue_size})")
        
        task_id = str(uuid.uuid4())
        task = Task(
            id=task_id,
            func=func,
            priority=priority,
            max_retries=max_retries
        )
        
        async with self._lock:
            heapq.heappush(self.queue, task)
            self._stats["queued"] += 1
        
        if not self.running:
            await self.start()
        
        logger.debug(f"Enqueued task {task_id} with priority {priority}")
        return task_id
    
    async def _worker(self, name: str) -> None:
        """Background worker that processes tasks"""
        logger.debug(f"Worker {name} started")
        
        while self.running:
            try:
                task = await self._get_next_task()
                if task:
                    await self._process_task(task, name)
                else:
                    await asyncio.sleep(0.1)
            except Exception as e:
                logger.error(f"Worker {name} error: {e}")
                await asyncio.sleep(1)
        
        logger.debug(f"Worker {name} stopped")
    
    async def _get_next_task(self) -> Optional[Task]:
        """Get next task from queue (priority-based using heap)"""
        async with self._lock:
            if not self.queue:
                return None
            return heapq.heappop(self.queue)
    
    async def _process_task(self, task: Task, worker_name: str) -> None:
        """Process a single task"""
        try:
            logger.debug(f"Worker {worker_name} processing task {task.id}")
            await task.func()
            async with self._lock:
                self._stats["processed"] += 1
            logger.debug(f"Task {task.id} completed successfully")
        except Exception as e:
            logger.error(f"Task {task.id} failed: {e}")
            async with self._lock:
                self._stats["failed"] += 1
            
            if task.retries < task.max_retries:
                task.retries += 1
                async with self._lock:
                    heapq.heappush(self.queue, task)
                logger.debug(f"Task {task.id} will retry (attempt {task.retries}/{task.max_retries})")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics"""
        async with self._lock:
            return {
                "queued": len(self.queue),
                "workers": len(self.workers),
                "running": self.running,
                **self._stats
            }


background_task_queue = BackgroundTaskQueue()

