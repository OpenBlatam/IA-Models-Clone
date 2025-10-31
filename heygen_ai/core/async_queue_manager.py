#!/usr/bin/env python3
"""
Async Queue Manager for Enhanced HeyGen AI
Handles background processing of video generation tasks with priorities and retries.
"""

import asyncio
import json
import time
import uuid
from typing import Any, Dict, Optional, Callable, List, Union
from dataclasses import dataclass, asdict
from enum import Enum
import structlog
from datetime import datetime, timedelta
import traceback

logger = structlog.get_logger()

class TaskPriority(Enum):
    """Task priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4
    CRITICAL = 5

class TaskStatus(Enum):
    """Task status values."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"

class TaskType(Enum):
    """Types of tasks."""
    VIDEO_GENERATION = "video_generation"
    VOICE_SYNTHESIS = "voice_synthesis"
    AVATAR_GENERATION = "avatar_generation"
    BATCH_PROCESSING = "batch_processing"
    MODEL_OPTIMIZATION = "model_optimization"

@dataclass
class QueueTask:
    """Represents a task in the queue."""
    id: str
    type: TaskType
    priority: TaskPriority
    status: TaskStatus
    payload: Dict[str, Any]
    created_at: float
    started_at: Optional[float]
    completed_at: Optional[float]
    retry_count: int
    max_retries: int
    error_message: Optional[str]
    result: Optional[Dict[str, Any]]
    tags: List[str]
    user_id: Optional[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def is_retryable(self) -> bool:
        """Check if task can be retried."""
        return self.retry_count < self.max_retries and self.status != TaskStatus.CANCELLED
    
    def should_retry(self) -> bool:
        """Check if task should be retried."""
        return (self.status == TaskStatus.FAILED and 
                self.is_retryable() and 
                self.retry_count < self.max_retries)

class AsyncQueueManager:
    """Manages asynchronous task processing with priorities and retries."""
    
    def __init__(
        self,
        max_workers: int = 4,
        max_queue_size: int = 1000,
        retry_delay_base: float = 1.0,
        max_retry_delay: float = 300.0
    ):
        self.max_workers = max_workers
        self.max_queue_size = max_queue_size
        self.retry_delay_base = retry_delay_base
        self.max_retry_delay = max_retry_delay
        
        # Task storage
        self.pending_queue: asyncio.PriorityQueue = asyncio.PriorityQueue(maxsize=max_queue_size)
        self.active_tasks: Dict[str, QueueTask] = {}
        self.completed_tasks: Dict[str, QueueTask] = {}
        self.failed_tasks: Dict[str, QueueTask] = {}
        
        # Worker management
        self.workers: List[asyncio.Task] = []
        self.is_running = False
        
        # Statistics
        self.stats = {
            "total_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "retried_tasks": 0,
            "average_processing_time": 0.0,
            "queue_size": 0
        }
        
        # Task handlers
        self.task_handlers: Dict[TaskType, Callable] = {}
        
        # Event callbacks
        self.on_task_complete: Optional[Callable] = None
        self.on_task_failed: Optional[Callable] = None
        self.on_task_retry: Optional[Callable] = None
    
    async def start(self):
        """Start the queue manager and workers."""
        if self.is_running:
            logger.warning("Queue manager is already running")
            return
        
        self.is_running = True
        
        # Start worker tasks
        for i in range(self.max_workers):
            worker = asyncio.create_task(self._worker(f"worker-{i}"))
            self.workers.append(worker)
        
        logger.info(f"Queue manager started with {self.max_workers} workers")
    
    async def stop(self):
        """Stop the queue manager and workers."""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Cancel all workers
        for worker in self.workers:
            worker.cancel()
        
        # Wait for workers to finish
        await asyncio.gather(*self.workers, return_exceptions=True)
        self.workers.clear()
        
        # Process remaining tasks
        await self._process_remaining_tasks()
        
        logger.info("Queue manager stopped")
    
    def register_handler(self, task_type: TaskType, handler: Callable):
        """Register a handler for a specific task type."""
        self.task_handlers[task_type] = handler
        logger.info(f"Registered handler for task type: {task_type.value}")
    
    async def submit_task(
        self,
        task_type: TaskType,
        payload: Dict[str, Any],
        priority: TaskPriority = TaskPriority.NORMAL,
        max_retries: int = 3,
        tags: Optional[List[str]] = None,
        user_id: Optional[str] = None
    ) -> str:
        """Submit a new task to the queue."""
        if self.pending_queue.qsize() >= self.max_queue_size:
            raise ValueError("Queue is full")
        
        task = QueueTask(
            id=str(uuid.uuid4()),
            type=task_type,
            priority=priority,
            status=TaskStatus.PENDING,
            payload=payload,
            created_at=time.time(),
            started_at=None,
            completed_at=None,
            retry_count=0,
            max_retries=max_retries,
            error_message=None,
            result=None,
            tags=tags or [],
            user_id=user_id
        )
        
        # Add to queue with priority (higher priority = lower number for heapq)
        priority_value = 10 - priority.value  # Invert priority for heapq
        await self.pending_queue.put((priority_value, task.created_at, task))
        
        self.stats["total_tasks"] += 1
        self.stats["queue_size"] = self.pending_queue.qsize()
        
        logger.info(f"Task submitted", 
                   task_id=task.id,
                   type=task_type.value,
                   priority=priority.value)
        
        return task.id
    
    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of a specific task."""
        # Check active tasks
        if task_id in self.active_tasks:
            return self.active_tasks[task_id].to_dict()
        
        # Check completed tasks
        if task_id in self.completed_tasks:
            return self.completed_tasks[task_id].to_dict()
        
        # Check failed tasks
        if task_id in self.failed_tasks:
            return self.failed_tasks[task_id].to_dict()
        
        return None
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending or processing task."""
        # Check if task is in queue
        queue_items = []
        while not self.pending_queue.empty():
            try:
                item = self.pending_queue.get_nowait()
                if item[2].id != task_id:
                    queue_items.append(item)
                else:
                    # Found the task, mark as cancelled
                    task = item[2]
                    task.status = TaskStatus.CANCELLED
                    self.completed_tasks[task_id] = task
                    logger.info(f"Task cancelled", task_id=task_id)
                    return True
            except asyncio.QueueEmpty:
                break
        
        # Restore queue
        for item in queue_items:
            await self.pending_queue.put(item)
        
        # Check active tasks
        if task_id in self.active_tasks:
            task = self.active_tasks[task_id]
            task.status = TaskStatus.CANCELLED
            del self.active_tasks[task_id]
            self.completed_tasks[task_id] = task
            logger.info(f"Active task cancelled", task_id=task_id)
            return True
        
        return False
    
    async def get_queue_stats(self) -> Dict[str, Any]:
        """Get current queue statistics."""
        return {
            **self.stats,
            "queue_size": self.pending_queue.qsize(),
            "active_tasks": len(self.active_tasks),
            "completed_tasks": len(self.completed_tasks),
            "failed_tasks": len(self.failed_tasks),
            "workers": len(self.workers),
            "is_running": self.is_running
        }
    
    async def _worker(self, worker_name: str):
        """Worker task that processes queued tasks."""
        logger.info(f"Worker {worker_name} started")
        
        while self.is_running:
            try:
                # Get task from queue
                try:
                    priority, created_at, task = await asyncio.wait_for(
                        self.pending_queue.get(), 
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                # Process task
                await self._process_task(task, worker_name)
                
                # Mark task as done
                self.pending_queue.task_done()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker {worker_name} error: {e}")
                await asyncio.sleep(1)
        
        logger.info(f"Worker {worker_name} stopped")
    
    async def _process_task(self, task: QueueTask, worker_name: str):
        """Process a single task."""
        try:
            # Update task status
            task.status = TaskStatus.PROCESSING
            task.started_at = time.time()
            self.active_tasks[task.id] = task
            
            logger.info(f"Processing task", 
                       task_id=task.id,
                       type=task.type.value,
                       worker=worker_name)
            
            # Get handler for task type
            handler = self.task_handlers.get(task.type)
            if not handler:
                raise ValueError(f"No handler registered for task type: {task.type.value}")
            
            # Execute task
            result = await handler(task.payload)
            
            # Mark as completed
            task.status = TaskStatus.COMPLETED
            task.completed_at = time.time()
            task.result = result
            
            # Move to completed tasks
            del self.active_tasks[task.id]
            self.completed_tasks[task.id] = task
            
            # Update statistics
            self.stats["completed_tasks"] += 1
            processing_time = task.completed_at - task.started_at
            self._update_average_processing_time(processing_time)
            
            logger.info(f"Task completed", 
                       task_id=task.id,
                       processing_time=processing_time)
            
            # Call completion callback
            if self.on_task_complete:
                try:
                    await self.on_task_complete(task)
                except Exception as e:
                    logger.error(f"Task completion callback failed: {e}")
            
        except Exception as e:
            await self._handle_task_failure(task, e, worker_name)
    
    async def _handle_task_failure(self, task: QueueTask, error: Exception, worker_name: str):
        """Handle task failure and retry logic."""
        error_message = str(error)
        task.error_message = error_message
        
        logger.error(f"Task failed", 
                    task_id=task.id,
                    error=error_message,
                    worker=worker_name)
        
        if task.should_retry():
            # Retry task
            task.retry_count += 1
            task.status = TaskStatus.RETRYING
            
            # Calculate retry delay with exponential backoff
            delay = min(
                self.retry_delay_base * (2 ** (task.retry_count - 1)),
                self.max_retry_delay
            )
            
            logger.info(f"Retrying task", 
                       task_id=task.id,
                       retry_count=task.retry_count,
                       delay=delay)
            
            # Call retry callback
            if self.on_task_retry:
                try:
                    await self.on_task_retry(task, delay)
                except Exception as e:
                    logger.error(f"Task retry callback failed: {e}")
            
            # Schedule retry
            asyncio.create_task(self._schedule_retry(task, delay))
            
        else:
            # Mark as failed
            task.status = TaskStatus.FAILED
            task.completed_at = time.time()
            
            # Move to failed tasks
            del self.active_tasks[task.id]
            self.failed_tasks[task.id] = task
            
            # Update statistics
            self.stats["failed_tasks"] += 1
            
            # Call failure callback
            if self.on_task_failed:
                try:
                    await self.on_task_failed(task)
                except Exception as e:
                    logger.error(f"Task failure callback failed: {e}")
    
    async def _schedule_retry(self, task: QueueTask, delay: float):
        """Schedule a task retry after delay."""
        await asyncio.sleep(delay)
        
        if not self.is_running:
            return
        
        # Reset task for retry
        task.status = TaskStatus.PENDING
        task.started_at = None
        task.completed_at = None
        task.error_message = None
        task.result = None
        
        # Add back to queue
        priority_value = 10 - task.priority.value
        await self.pending_queue.put((priority_value, task.created_at, task))
        
        self.stats["retried_tasks"] += 1
        logger.info(f"Task scheduled for retry", 
                   task_id=task.id,
                   retry_count=task.retry_count)
    
    def _update_average_processing_time(self, new_time: float):
        """Update average processing time statistics."""
        current_avg = self.stats["average_processing_time"]
        completed_count = self.stats["completed_tasks"]
        
        if completed_count == 1:
            self.stats["average_processing_time"] = new_time
        else:
            self.stats["average_processing_time"] = (
                (current_avg * (completed_count - 1) + new_time) / completed_count
            )
    
    async def _process_remaining_tasks(self):
        """Process any remaining tasks when stopping."""
        remaining_tasks = []
        
        # Collect remaining tasks from queue
        while not self.pending_queue.empty():
            try:
                priority, created_at, task = self.pending_queue.get_nowait()
                remaining_tasks.append(task)
            except asyncio.QueueEmpty:
                break
        
        # Mark remaining tasks as cancelled
        for task in remaining_tasks:
            task.status = TaskStatus.CANCELLED
            self.completed_tasks[task.id] = task
        
        logger.info(f"Marked {len(remaining_tasks)} remaining tasks as cancelled")
    
    async def cleanup_old_tasks(self, max_age_hours: int = 24):
        """Clean up old completed and failed tasks."""
        cutoff_time = time.time() - (max_age_hours * 3600)
        
        # Clean completed tasks
        old_completed = [
            task_id for task_id, task in self.completed_tasks.items()
            if task.completed_at and task.completed_at < cutoff_time
        ]
        for task_id in old_completed:
            del self.completed_tasks[task_id]
        
        # Clean failed tasks
        old_failed = [
            task_id for task_id, task in self.failed_tasks.items()
            if task.completed_at and task.completed_at < cutoff_time
        ]
        for task_id in old_failed:
            del self.failed_tasks[task_id]
        
        if old_completed or old_failed:
            logger.info(f"Cleaned up {len(old_completed)} old completed tasks and {len(old_failed)} old failed tasks")

# Global queue manager instance
queue_manager: Optional[AsyncQueueManager] = None

def get_queue_manager() -> AsyncQueueManager:
    """Get global queue manager instance."""
    global queue_manager
    if queue_manager is None:
        queue_manager = AsyncQueueManager()
    return queue_manager

async def shutdown_queue_manager():
    """Shutdown global queue manager."""
    global queue_manager
    if queue_manager:
        await queue_manager.stop()
        queue_manager = None

