"""
Unified Queue for Color Grading AI
===================================

Unified queue implementation with priority, scheduling, and retry.
"""

import logging
import asyncio
import json
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
import heapq
import uuid

logger = logging.getLogger(__name__)


class TaskPriority(Enum):
    """Task priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4


class TaskStatus(Enum):
    """Task status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class RetryStrategy(Enum):
    """Retry strategies."""
    IMMEDIATE = "immediate"
    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    FIXED = "fixed"


@dataclass
class UnifiedQueueTask:
    """Unified queue task."""
    task_id: str
    task_type: str
    parameters: Dict[str, Any]
    priority: TaskPriority
    scheduled_at: datetime
    created_at: datetime
    retry_count: int = 0
    max_retries: int = 3
    retry_strategy: RetryStrategy = RetryStrategy.EXPONENTIAL
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    depends_on: List[str] = field(default_factory=list)
    
    def __lt__(self, other):
        """Compare for priority queue."""
        if self.priority.value != other.priority.value:
            return self.priority.value > other.priority.value
        return self.scheduled_at < other.scheduled_at
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "parameters": self.parameters,
            "priority": self.priority.value,
            "scheduled_at": self.scheduled_at.isoformat(),
            "created_at": self.created_at.isoformat(),
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "retry_strategy": self.retry_strategy.value,
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
            "depends_on": self.depends_on,
        }


class UnifiedQueue:
    """
    Unified queue with priority, scheduling, and retry.
    
    Features:
    - Priority queue
    - Scheduled execution
    - Retry strategies
    - Task dependencies
    - Rate limiting
    - Persistence (optional)
    """
    
    def __init__(
        self,
        max_workers: int = 5,
        storage_dir: Optional[str] = None
    ):
        """
        Initialize unified queue.
        
        Args:
            max_workers: Maximum concurrent workers
            storage_dir: Optional directory for persistence
        """
        self.max_workers = max_workers
        self.storage_dir = Path(storage_dir) if storage_dir else None
        if self.storage_dir:
            self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        self._queue: List[UnifiedQueueTask] = []
        self._processing: Dict[str, UnifiedQueueTask] = {}
        self._completed: Dict[str, UnifiedQueueTask] = {}
        self._workers: List[asyncio.Task] = []
        self._running = False
        self._rate_limiter: Dict[str, datetime] = {}
        self._task_handlers: Dict[str, Callable] = {}
    
    def register_handler(self, task_type: str, handler: Callable):
        """
        Register task handler.
        
        Args:
            task_type: Task type
            handler: Handler function (async)
        """
        self._task_handlers[task_type] = handler
        logger.info(f"Registered handler for task type: {task_type}")
    
    async def enqueue(
        self,
        task_type: str,
        parameters: Dict[str, Any],
        priority: TaskPriority = TaskPriority.NORMAL,
        scheduled_at: Optional[datetime] = None,
        max_retries: int = 3,
        retry_strategy: RetryStrategy = RetryStrategy.EXPONENTIAL,
        depends_on: Optional[List[str]] = None
    ) -> str:
        """
        Enqueue task.
        
        Args:
            task_type: Task type
            parameters: Task parameters
            priority: Task priority
            scheduled_at: Optional scheduled execution time
            max_retries: Maximum retry attempts
            retry_strategy: Retry strategy
            depends_on: Optional list of task IDs this depends on
            
        Returns:
            Task ID
        """
        task_id = str(uuid.uuid4())
        
        if not scheduled_at:
            scheduled_at = datetime.now()
        
        task = UnifiedQueueTask(
            task_id=task_id,
            task_type=task_type,
            parameters=parameters,
            priority=priority,
            scheduled_at=scheduled_at,
            created_at=datetime.now(),
            max_retries=max_retries,
            retry_strategy=retry_strategy,
            depends_on=depends_on or []
        )
        
        heapq.heappush(self._queue, task)
        self._save_task(task)
        logger.info(f"Enqueued task {task_id} with priority {priority.value}")
        
        return task_id
    
    async def start(self):
        """Start queue workers."""
        if self._running:
            return
        
        self._running = True
        for i in range(self.max_workers):
            worker = asyncio.create_task(self._worker(f"worker-{i}"))
            self._workers.append(worker)
        
        logger.info(f"Started {self.max_workers} queue workers")
    
    async def stop(self):
        """Stop queue workers."""
        self._running = False
        for worker in self._workers:
            worker.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()
        logger.info("Stopped queue workers")
    
    async def _worker(self, worker_id: str):
        """Worker coroutine."""
        while self._running:
            try:
                if not self._queue:
                    await asyncio.sleep(0.1)
                    continue
                
                task = heapq.heappop(self._queue)
                
                # Check if scheduled
                if task.scheduled_at > datetime.now():
                    heapq.heappush(self._queue, task)
                    await asyncio.sleep(0.1)
                    continue
                
                # Check dependencies
                if task.depends_on:
                    if not all(dep_id in self._completed for dep_id in task.depends_on):
                        heapq.heappush(self._queue, task)
                        await asyncio.sleep(0.1)
                        continue
                
                # Check rate limit
                if not self._check_rate_limit(task.task_type):
                    heapq.heappush(self._queue, task)
                    await asyncio.sleep(1)
                    continue
                
                # Process task
                task.status = TaskStatus.PROCESSING
                self._processing[task.task_id] = task
                self._save_task(task)
                
                try:
                    handler = self._task_handlers.get(task.task_type)
                    if handler:
                        result = await handler(task.parameters)
                    else:
                        result = {"status": "no_handler", "task_id": task.task_id}
                    
                    task.status = TaskStatus.COMPLETED
                    task.result = result
                    self._completed[task.task_id] = task
                    del self._processing[task.task_id]
                    self._save_task(task)
                except Exception as e:
                    task.status = TaskStatus.FAILED
                    task.error = str(e)
                    
                    # Retry logic
                    if task.retry_count < task.max_retries:
                        task.retry_count += 1
                        task.scheduled_at = self._calculate_retry_time(task)
                        task.status = TaskStatus.PENDING
                        heapq.heappush(self._queue, task)
                        self._save_task(task)
                        logger.warning(f"Task {task.task_id} failed, retrying ({task.retry_count}/{task.max_retries})")
                    else:
                        self._completed[task.task_id] = task
                        del self._processing[task.task_id]
                        self._save_task(task)
                        logger.error(f"Task {task.task_id} failed after {task.max_retries} retries")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}", exc_info=True)
                await asyncio.sleep(1)
    
    def _calculate_retry_time(self, task: UnifiedQueueTask) -> datetime:
        """Calculate retry time based on strategy."""
        base_delay = 5  # seconds
        
        if task.retry_strategy == RetryStrategy.IMMEDIATE:
            delay = 0
        elif task.retry_strategy == RetryStrategy.EXPONENTIAL:
            delay = base_delay * (2 ** task.retry_count)
        elif task.retry_strategy == RetryStrategy.LINEAR:
            delay = base_delay * task.retry_count
        else:  # FIXED
            delay = base_delay
        
        return datetime.now() + timedelta(seconds=delay)
    
    def _check_rate_limit(self, task_type: str, min_interval: float = 1.0) -> bool:
        """Check rate limit for task type."""
        last_execution = self._rate_limiter.get(task_type)
        if last_execution:
            elapsed = (datetime.now() - last_execution).total_seconds()
            if elapsed < min_interval:
                return False
        
        self._rate_limiter[task_type] = datetime.now()
        return True
    
    def _save_task(self, task: UnifiedQueueTask):
        """Save task to disk (if storage enabled)."""
        if not self.storage_dir:
            return
        
        try:
            task_file = self.storage_dir / f"{task.task_id}.json"
            with open(task_file, "w") as f:
                json.dump(task.to_dict(), f, default=str)
        except Exception as e:
            logger.debug(f"Error saving task: {e}")
    
    async def get_task(self, task_id: str) -> Optional[UnifiedQueueTask]:
        """Get task by ID."""
        # Check all collections
        for collection in [self._processing, self._completed]:
            if task_id in collection:
                return collection[task_id]
        
        # Check queue
        for task in self._queue:
            if task.task_id == task_id:
                return task
        
        # Check storage
        if self.storage_dir:
            task_file = self.storage_dir / f"{task_id}.json"
            if task_file.exists():
                try:
                    with open(task_file, "r") as f:
                        data = json.load(f)
                    return UnifiedQueueTask(**data)
                except:
                    pass
        
        return None
    
    async def get_queue_status(self) -> Dict[str, Any]:
        """Get queue status."""
        return {
            "queue_size": len(self._queue),
            "processing": len(self._processing),
            "completed": len(self._completed),
            "workers": len(self._workers),
            "running": self._running,
        }
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending task."""
        # Remove from queue
        new_queue = [t for t in self._queue if t.task_id != task_id]
        if len(new_queue) != len(self._queue):
            self._queue = new_queue
            heapq.heapify(self._queue)
            return True
        return False

