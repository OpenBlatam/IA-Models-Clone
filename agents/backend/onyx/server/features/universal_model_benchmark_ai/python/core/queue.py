"""
Queue Module - Task queue system.

Provides:
- Priority queue
- Task scheduling
- Retry logic
- Dead letter queue
"""

import logging
import time
import threading
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from queue import PriorityQueue, Queue
import json

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    """Task status."""
    PENDING = "pending"
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    DEAD_LETTER = "dead_letter"


@dataclass
class Task:
    """Task definition."""
    id: str
    task_type: str
    payload: Dict[str, Any]
    priority: int = 5  # 1 (highest) to 10 (lowest)
    max_retries: int = 3
    retry_count: int = 0
    status: TaskStatus = TaskStatus.PENDING
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error: Optional[str] = None
    result: Optional[Any] = None
    
    def __lt__(self, other):
        """Compare tasks by priority (lower number = higher priority)."""
        return self.priority < other.priority
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "task_type": self.task_type,
            "payload": self.payload,
            "priority": self.priority,
            "max_retries": self.max_retries,
            "retry_count": self.retry_count,
            "status": self.status.value,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "error": self.error,
        }


class TaskQueue:
    """Priority task queue."""
    
    def __init__(self, max_size: int = 1000):
        """
        Initialize task queue.
        
        Args:
            max_size: Maximum queue size
        """
        self.queue = PriorityQueue(maxsize=max_size)
        self.tasks: Dict[str, Task] = {}
        self.handlers: Dict[str, Callable] = {}
        self.lock = threading.Lock()
        self.workers: List[threading.Thread] = []
        self.running = False
    
    def register_handler(self, task_type: str, handler: Callable) -> None:
        """
        Register task handler.
        
        Args:
            task_type: Task type
            handler: Handler function
        """
        self.handlers[task_type] = handler
        logger.info(f"Registered handler for task type: {task_type}")
    
    def enqueue(
        self,
        task_type: str,
        payload: Dict[str, Any],
        priority: int = 5,
        max_retries: int = 3,
    ) -> Task:
        """
        Enqueue a task.
        
        Args:
            task_type: Task type
            payload: Task payload
            priority: Task priority (1-10)
            max_retries: Maximum retry attempts
            
        Returns:
            Created task
        """
        task_id = f"{task_type}_{int(time.time() * 1000)}"
        task = Task(
            id=task_id,
            task_type=task_type,
            payload=payload,
            priority=priority,
            max_retries=max_retries,
        )
        
        with self.lock:
            self.tasks[task_id] = task
            self.queue.put((priority, time.time(), task))
            task.status = TaskStatus.QUEUED
        
        logger.info(f"Enqueued task: {task_id} (priority: {priority})")
        return task
    
    def dequeue(self) -> Optional[Task]:
        """Dequeue a task."""
        try:
            _, _, task = self.queue.get_nowait()
            return task
        except:
            return None
    
    def process_task(self, task: Task) -> None:
        """Process a task."""
        if task.task_type not in self.handlers:
            task.status = TaskStatus.FAILED
            task.error = f"No handler for task type: {task.task_type}"
            logger.error(task.error)
            return
        
        handler = self.handlers[task.task_type]
        task.status = TaskStatus.PROCESSING
        task.started_at = datetime.now().isoformat()
        
        try:
            result = handler(task.payload)
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now().isoformat()
            task.result = result
            logger.info(f"Completed task: {task.id}")
        except Exception as e:
            task.error = str(e)
            logger.error(f"Task {task.id} failed: {e}")
            
            # Retry logic
            if task.retry_count < task.max_retries:
                task.retry_count += 1
                task.status = TaskStatus.RETRYING
                logger.info(f"Retrying task {task.id} (attempt {task.retry_count}/{task.max_retries})")
                
                # Re-enqueue with higher priority
                self.queue.put((task.priority - 1, time.time(), task))
            else:
                task.status = TaskStatus.DEAD_LETTER
                logger.error(f"Task {task.id} moved to dead letter queue")
    
    def worker_loop(self) -> None:
        """Worker loop."""
        while self.running:
            task = self.dequeue()
            if task:
                self.process_task(task)
            else:
                time.sleep(0.1)  # Small delay when queue is empty
    
    def start_workers(self, num_workers: int = 3) -> None:
        """
        Start worker threads.
        
        Args:
            num_workers: Number of worker threads
        """
        self.running = True
        
        for i in range(num_workers):
            worker = threading.Thread(target=self.worker_loop, daemon=True)
            worker.start()
            self.workers.append(worker)
            logger.info(f"Started worker {i+1}/{num_workers}")
    
    def stop_workers(self) -> None:
        """Stop worker threads."""
        self.running = False
        for worker in self.workers:
            worker.join(timeout=5)
        self.workers.clear()
        logger.info("Stopped all workers")
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """Get task by ID."""
        return self.tasks.get(task_id)
    
    def get_queue_size(self) -> int:
        """Get current queue size."""
        return self.queue.qsize()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        with self.lock:
            status_counts = {}
            for task in self.tasks.values():
                status = task.status.value
                status_counts[status] = status_counts.get(status, 0) + 1
            
            return {
                "queue_size": self.get_queue_size(),
                "total_tasks": len(self.tasks),
                "status_counts": status_counts,
                "workers": len(self.workers),
            }












