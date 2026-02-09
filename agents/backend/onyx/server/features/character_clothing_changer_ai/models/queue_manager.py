"""
Queue Manager for Flux2 Clothing Changer
=========================================

Asynchronous task queue management for clothing change operations.
"""

import queue
import threading
import time
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import uuid
import logging

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Task status enumeration."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Task:
    """Task in the queue."""
    task_id: str
    image: Any
    clothing_description: str
    mask: Optional[Any] = None
    prompt: Optional[str] = None
    negative_prompt: Optional[str] = None
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None
    created_at: float = 0.0
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.created_at == 0.0:
            self.created_at = time.time()
        if self.metadata is None:
            self.metadata = {}


class QueueManager:
    """Manages asynchronous task queue."""
    
    def __init__(
        self,
        model,
        max_queue_size: int = 100,
        max_workers: int = 2,
    ):
        """
        Initialize queue manager.
        
        Args:
            model: Flux2ClothingChangerModelV2 instance
            max_queue_size: Maximum queue size
            max_workers: Maximum worker threads
        """
        self.model = model
        self.max_queue_size = max_queue_size
        self.max_workers = max_workers
        
        self.task_queue = queue.Queue(maxsize=max_queue_size)
        self.tasks: Dict[str, Task] = {}
        self.tasks_lock = threading.Lock()
        
        self.workers: List[threading.Thread] = []
        self.stop_event = threading.Event()
        self.is_running = False
    
    def start(self) -> None:
        """Start queue manager and workers."""
        if self.is_running:
            return
        
        self.is_running = True
        self.stop_event.clear()
        
        # Start worker threads
        for i in range(self.max_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                name=f"QueueWorker-{i}",
                daemon=True,
            )
            worker.start()
            self.workers.append(worker)
        
        logger.info(f"Queue manager started with {self.max_workers} workers")
    
    def stop(self, wait: bool = True) -> None:
        """Stop queue manager and workers."""
        if not self.is_running:
            return
        
        self.is_running = False
        self.stop_event.set()
        
        if wait:
            for worker in self.workers:
                worker.join(timeout=5.0)
        
        self.workers.clear()
        logger.info("Queue manager stopped")
    
    def submit_task(
        self,
        image: Any,
        clothing_description: str,
        mask: Optional[Any] = None,
        prompt: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Submit a task to the queue.
        
        Args:
            image: Input image
            clothing_description: Clothing description
            mask: Optional mask
            prompt: Optional prompt
            negative_prompt: Optional negative prompt
            metadata: Optional metadata
            
        Returns:
            Task ID
        """
        task_id = str(uuid.uuid4())
        
        task = Task(
            task_id=task_id,
            image=image,
            clothing_description=clothing_description,
            mask=mask,
            prompt=prompt,
            negative_prompt=negative_prompt,
            metadata=metadata or {},
        )
        
        try:
            self.task_queue.put(task, timeout=5.0)
            
            with self.tasks_lock:
                self.tasks[task_id] = task
            
            logger.info(f"Task {task_id} submitted to queue")
            return task_id
            
        except queue.Full:
            raise RuntimeError("Task queue is full")
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Get task status.
        
        Args:
            task_id: Task ID
            
        Returns:
            Task status dictionary or None if not found
        """
        with self.tasks_lock:
            task = self.tasks.get(task_id)
        
        if not task:
            return None
        
        return {
            "task_id": task.task_id,
            "status": task.status.value,
            "created_at": task.created_at,
            "started_at": task.started_at,
            "completed_at": task.completed_at,
            "error": task.error,
            "has_result": task.result is not None,
            "metadata": task.metadata,
        }
    
    def get_task_result(self, task_id: str) -> Optional[Any]:
        """
        Get task result.
        
        Args:
            task_id: Task ID
            
        Returns:
            Task result or None if not found/not completed
        """
        with self.tasks_lock:
            task = self.tasks.get(task_id)
        
        if not task or task.status != TaskStatus.COMPLETED:
            return None
        
        return task.result
    
    def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a pending task.
        
        Args:
            task_id: Task ID
            
        Returns:
            True if cancelled, False if not found or already processing
        """
        with self.tasks_lock:
            task = self.tasks.get(task_id)
        
        if not task:
            return False
        
        if task.status == TaskStatus.PENDING:
            task.status = TaskStatus.CANCELLED
            logger.info(f"Task {task_id} cancelled")
            return True
        
        return False
    
    def _worker_loop(self) -> None:
        """Worker thread loop."""
        while not self.stop_event.is_set():
            try:
                task = self.task_queue.get(timeout=1.0)
                
                # Check if cancelled
                if task.status == TaskStatus.CANCELLED:
                    self.task_queue.task_done()
                    continue
                
                # Process task
                task.status = TaskStatus.PROCESSING
                task.started_at = time.time()
                
                try:
                    result = self.model.change_clothing(
                        image=task.image,
                        clothing_description=task.clothing_description,
                        mask=task.mask,
                        prompt=task.prompt,
                        negative_prompt=task.negative_prompt,
                    )
                    
                    task.result = result
                    task.status = TaskStatus.COMPLETED
                    task.completed_at = time.time()
                    logger.info(f"Task {task.task_id} completed")
                    
                except Exception as e:
                    task.status = TaskStatus.FAILED
                    task.error = str(e)
                    task.completed_at = time.time()
                    logger.error(f"Task {task.task_id} failed: {e}")
                
                self.task_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in worker loop: {e}")
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        with self.tasks_lock:
            total_tasks = len(self.tasks)
            pending = sum(1 for t in self.tasks.values() if t.status == TaskStatus.PENDING)
            processing = sum(1 for t in self.tasks.values() if t.status == TaskStatus.PROCESSING)
            completed = sum(1 for t in self.tasks.values() if t.status == TaskStatus.COMPLETED)
            failed = sum(1 for t in self.tasks.values() if t.status == TaskStatus.FAILED)
        
        return {
            "queue_size": self.task_queue.qsize(),
            "max_queue_size": self.max_queue_size,
            "workers": self.max_workers,
            "is_running": self.is_running,
            "tasks": {
                "total": total_tasks,
                "pending": pending,
                "processing": processing,
                "completed": completed,
                "failed": failed,
            },
        }


