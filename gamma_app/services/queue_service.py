"""
Gamma App - Queue Service
Advanced task queue and job processing service
"""

import asyncio
import json
import time
import uuid
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, asdict
from enum import Enum
import redis
from datetime import datetime, timedelta
import logging
import pickle
import hashlib

logger = logging.getLogger(__name__)

class TaskStatus(Enum):
    """Task status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    CANCELLED = "cancelled"

class TaskPriority(Enum):
    """Task priority"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4

@dataclass
class Task:
    """Task definition"""
    id: str
    name: str
    args: List[Any]
    kwargs: Dict[str, Any]
    priority: TaskPriority
    status: TaskStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    retry_count: int = 0
    max_retries: int = 3
    timeout: int = 300
    result: Any = None
    error: Optional[str] = None
    worker_id: Optional[str] = None
    scheduled_at: Optional[datetime] = None

@dataclass
class Worker:
    """Worker definition"""
    id: str
    name: str
    status: str
    current_task: Optional[str] = None
    tasks_completed: int = 0
    tasks_failed: int = 0
    last_heartbeat: datetime = None
    created_at: datetime = None

class QueueService:
    """Advanced queue service"""
    
    def __init__(self, redis_client: redis.Redis, config: Dict[str, Any]):
        self.redis = redis_client
        self.config = config
        self.queues: Dict[str, asyncio.Queue] = {}
        self.workers: Dict[str, Worker] = {}
        self.task_handlers: Dict[str, Callable] = {}
        self.worker_id = str(uuid.uuid4())
        self.running = False
        
        # Initialize queues
        self._init_queues()
        
        # Register default task handlers
        self._register_default_handlers()
    
    def _init_queues(self):
        """Initialize task queues"""
        queue_names = [
            "urgent", "high", "normal", "low", "scheduled", "failed", "completed"
        ]
        
        for queue_name in queue_names:
            self.queues[queue_name] = asyncio.Queue()
    
    def _register_default_handlers(self):
        """Register default task handlers"""
        self.task_handlers.update({
            "send_email": self._handle_send_email,
            "generate_content": self._handle_generate_content,
            "export_document": self._handle_export_document,
            "process_file": self._handle_process_file,
            "cleanup_temp_files": self._handle_cleanup_temp_files,
            "send_notification": self._handle_send_notification,
            "update_analytics": self._handle_update_analytics,
            "backup_data": self._handle_backup_data,
        })
    
    def register_handler(self, task_name: str, handler: Callable):
        """Register task handler"""
        self.task_handlers[task_name] = handler
        logger.info(f"Registered handler for task: {task_name}")
    
    async def enqueue_task(
        self,
        name: str,
        args: List[Any] = None,
        kwargs: Dict[str, Any] = None,
        priority: TaskPriority = TaskPriority.NORMAL,
        delay: int = 0,
        max_retries: int = 3,
        timeout: int = 300
    ) -> str:
        """Enqueue a new task"""
        try:
            # Generate task ID
            task_id = str(uuid.uuid4())
            
            # Create task
            task = Task(
                id=task_id,
                name=name,
                args=args or [],
                kwargs=kwargs or {},
                priority=priority,
                status=TaskStatus.PENDING,
                created_at=datetime.now(),
                max_retries=max_retries,
                timeout=timeout,
                scheduled_at=datetime.now() + timedelta(seconds=delay) if delay > 0 else None
            )
            
            # Store task
            await self._store_task(task)
            
            # Add to appropriate queue
            if delay > 0:
                await self._schedule_task(task)
            else:
                await self._add_to_queue(task)
            
            logger.info(f"Enqueued task: {task_id} ({name})")
            return task_id
            
        except Exception as e:
            logger.error(f"Error enqueuing task: {e}")
            raise
    
    async def dequeue_task(self, worker_id: str) -> Optional[Task]:
        """Dequeue a task for processing"""
        try:
            # Try to get task from priority queues
            for priority in [TaskPriority.URGENT, TaskPriority.HIGH, TaskPriority.NORMAL, TaskPriority.LOW]:
                queue_name = priority.name.lower()
                queue = self.queues[queue_name]
                
                try:
                    task = queue.get_nowait()
                    if task:
                        # Update task status
                        task.status = TaskStatus.RUNNING
                        task.started_at = datetime.now()
                        task.worker_id = worker_id
                        
                        # Update worker
                        if worker_id in self.workers:
                            self.workers[worker_id].current_task = task.id
                        
                        # Store updated task
                        await self._store_task(task)
                        
                        logger.info(f"Dequeued task: {task.id} for worker: {worker_id}")
                        return task
                except asyncio.QueueEmpty:
                    continue
            
            return None
            
        except Exception as e:
            logger.error(f"Error dequeuing task: {e}")
            return None
    
    async def complete_task(self, task_id: str, result: Any = None) -> bool:
        """Mark task as completed"""
        try:
            task = await self._get_task(task_id)
            if not task:
                return False
            
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()
            task.result = result
            
            # Update worker stats
            if task.worker_id and task.worker_id in self.workers:
                self.workers[task.worker_id].tasks_completed += 1
                self.workers[task.worker_id].current_task = None
            
            # Store updated task
            await self._store_task(task)
            
            # Move to completed queue
            await self.queues["completed"].put(task)
            
            logger.info(f"Completed task: {task_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error completing task: {e}")
            return False
    
    async def fail_task(self, task_id: str, error: str) -> bool:
        """Mark task as failed"""
        try:
            task = await self._get_task(task_id)
            if not task:
                return False
            
            task.retry_count += 1
            task.error = error
            
            # Update worker stats
            if task.worker_id and task.worker_id in self.workers:
                self.workers[task.worker_id].tasks_failed += 1
                self.workers[task.worker_id].current_task = None
            
            # Check if should retry
            if task.retry_count < task.max_retries:
                task.status = TaskStatus.RETRYING
                # Schedule retry with exponential backoff
                delay = min(60 * (2 ** task.retry_count), 3600)  # Max 1 hour
                task.scheduled_at = datetime.now() + timedelta(seconds=delay)
                await self._schedule_task(task)
            else:
                task.status = TaskStatus.FAILED
                task.completed_at = datetime.now()
                # Move to failed queue
                await self.queues["failed"].put(task)
            
            # Store updated task
            await self._store_task(task)
            
            logger.warning(f"Failed task: {task_id} - {error}")
            return True
            
        except Exception as e:
            logger.error(f"Error failing task: {e}")
            return False
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a task"""
        try:
            task = await self._get_task(task_id)
            if not task:
                return False
            
            if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                return False
            
            task.status = TaskStatus.CANCELLED
            task.completed_at = datetime.now()
            
            # Store updated task
            await self._store_task(task)
            
            logger.info(f"Cancelled task: {task_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error cancelling task: {e}")
            return False
    
    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task status"""
        try:
            task = await self._get_task(task_id)
            if not task:
                return None
            
            return {
                "id": task.id,
                "name": task.name,
                "status": task.status.value,
                "priority": task.priority.value,
                "created_at": task.created_at.isoformat(),
                "started_at": task.started_at.isoformat() if task.started_at else None,
                "completed_at": task.completed_at.isoformat() if task.completed_at else None,
                "retry_count": task.retry_count,
                "max_retries": task.max_retries,
                "worker_id": task.worker_id,
                "error": task.error,
                "result": task.result
            }
            
        except Exception as e:
            logger.error(f"Error getting task status: {e}")
            return None
    
    async def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics"""
        try:
            stats = {
                "queues": {},
                "workers": len(self.workers),
                "total_tasks": 0,
                "pending_tasks": 0,
                "running_tasks": 0,
                "completed_tasks": 0,
                "failed_tasks": 0
            }
            
            # Get queue sizes
            for queue_name, queue in self.queues.items():
                stats["queues"][queue_name] = queue.qsize()
            
            # Get task counts from Redis
            task_keys = self.redis.keys("task:*")
            stats["total_tasks"] = len(task_keys)
            
            for task_key in task_keys:
                task_data = self.redis.get(task_key)
                if task_data:
                    task = pickle.loads(task_data)
                    if task.status == TaskStatus.PENDING:
                        stats["pending_tasks"] += 1
                    elif task.status == TaskStatus.RUNNING:
                        stats["running_tasks"] += 1
                    elif task.status == TaskStatus.COMPLETED:
                        stats["completed_tasks"] += 1
                    elif task.status == TaskStatus.FAILED:
                        stats["failed_tasks"] += 1
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting queue stats: {e}")
            return {}
    
    async def start_worker(self, name: str) -> str:
        """Start a new worker"""
        try:
            worker_id = str(uuid.uuid4())
            
            worker = Worker(
                id=worker_id,
                name=name,
                status="running",
                last_heartbeat=datetime.now(),
                created_at=datetime.now()
            )
            
            self.workers[worker_id] = worker
            await self._store_worker(worker)
            
            logger.info(f"Started worker: {worker_id} ({name})")
            return worker_id
            
        except Exception as e:
            logger.error(f"Error starting worker: {e}")
            raise
    
    async def stop_worker(self, worker_id: str) -> bool:
        """Stop a worker"""
        try:
            if worker_id not in self.workers:
                return False
            
            worker = self.workers[worker_id]
            worker.status = "stopped"
            
            # Cancel current task if any
            if worker.current_task:
                await self.cancel_task(worker.current_task)
            
            # Remove from memory
            del self.workers[worker_id]
            
            # Remove from storage
            await self._delete_worker(worker_id)
            
            logger.info(f"Stopped worker: {worker_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping worker: {e}")
            return False
    
    async def process_tasks(self, worker_id: str):
        """Process tasks for a worker"""
        try:
            self.running = True
            
            while self.running:
                try:
                    # Get task from queue
                    task = await self.dequeue_task(worker_id)
                    
                    if task:
                        # Process task
                        await self._process_task(task, worker_id)
                    else:
                        # No tasks available, wait a bit
                        await asyncio.sleep(1)
                        
                except Exception as e:
                    logger.error(f"Error processing task: {e}")
                    await asyncio.sleep(5)
            
        except Exception as e:
            logger.error(f"Error in task processor: {e}")
        finally:
            self.running = False
    
    async def _process_task(self, task: Task, worker_id: str):
        """Process a single task"""
        try:
            # Get task handler
            handler = self.task_handlers.get(task.name)
            if not handler:
                await self.fail_task(task.id, f"No handler found for task: {task.name}")
                return
            
            # Execute task with timeout
            try:
                result = await asyncio.wait_for(
                    handler(*task.args, **task.kwargs),
                    timeout=task.timeout
                )
                await self.complete_task(task.id, result)
            except asyncio.TimeoutError:
                await self.fail_task(task.id, f"Task timeout after {task.timeout} seconds")
            except Exception as e:
                await self.fail_task(task.id, str(e))
                
        except Exception as e:
            logger.error(f"Error processing task {task.id}: {e}")
            await self.fail_task(task.id, str(e))
    
    async def _schedule_task(self, task: Task):
        """Schedule a task for later execution"""
        try:
            # Store in scheduled queue
            await self.queues["scheduled"].put(task)
            
            # Set Redis key with TTL
            key = f"scheduled_task:{task.id}"
            self.redis.setex(key, int((task.scheduled_at - datetime.now()).total_seconds()), task.id)
            
        except Exception as e:
            logger.error(f"Error scheduling task: {e}")
    
    async def _add_to_queue(self, task: Task):
        """Add task to appropriate queue"""
        try:
            queue_name = task.priority.name.lower()
            queue = self.queues[queue_name]
            await queue.put(task)
        except Exception as e:
            logger.error(f"Error adding task to queue: {e}")
    
    async def _store_task(self, task: Task):
        """Store task in Redis"""
        try:
            key = f"task:{task.id}"
            data = pickle.dumps(task)
            self.redis.setex(key, 86400 * 7, data)  # Store for 7 days
        except Exception as e:
            logger.error(f"Error storing task: {e}")
    
    async def _get_task(self, task_id: str) -> Optional[Task]:
        """Get task from Redis"""
        try:
            key = f"task:{task_id}"
            data = self.redis.get(key)
            if data:
                return pickle.loads(data)
            return None
        except Exception as e:
            logger.error(f"Error getting task: {e}")
            return None
    
    async def _store_worker(self, worker: Worker):
        """Store worker in Redis"""
        try:
            key = f"worker:{worker.id}"
            data = asdict(worker)
            self.redis.setex(key, 3600, json.dumps(data, default=str))  # Store for 1 hour
        except Exception as e:
            logger.error(f"Error storing worker: {e}")
    
    async def _delete_worker(self, worker_id: str):
        """Delete worker from Redis"""
        try:
            key = f"worker:{worker_id}"
            self.redis.delete(key)
        except Exception as e:
            logger.error(f"Error deleting worker: {e}")
    
    # Default task handlers
    async def _handle_send_email(self, to: str, subject: str, body: str, **kwargs):
        """Handle send email task"""
        # This would integrate with email service
        logger.info(f"Sending email to {to}: {subject}")
        await asyncio.sleep(1)  # Simulate email sending
        return {"status": "sent", "to": to}
    
    async def _handle_generate_content(self, prompt: str, content_type: str, **kwargs):
        """Handle generate content task"""
        # This would integrate with AI service
        logger.info(f"Generating {content_type} content: {prompt}")
        await asyncio.sleep(2)  # Simulate content generation
        return {"content": f"Generated {content_type} content", "prompt": prompt}
    
    async def _handle_export_document(self, content_id: str, format: str, **kwargs):
        """Handle export document task"""
        # This would integrate with export service
        logger.info(f"Exporting document {content_id} to {format}")
        await asyncio.sleep(3)  # Simulate export
        return {"export_url": f"/exports/{content_id}.{format}", "format": format}
    
    async def _handle_process_file(self, file_path: str, operation: str, **kwargs):
        """Handle process file task"""
        # This would integrate with file service
        logger.info(f"Processing file {file_path} with operation {operation}")
        await asyncio.sleep(1)  # Simulate file processing
        return {"processed_file": file_path, "operation": operation}
    
    async def _handle_cleanup_temp_files(self, **kwargs):
        """Handle cleanup temp files task"""
        # This would clean up temporary files
        logger.info("Cleaning up temporary files")
        await asyncio.sleep(0.5)  # Simulate cleanup
        return {"cleaned_files": 10}
    
    async def _handle_send_notification(self, user_id: str, message: str, **kwargs):
        """Handle send notification task"""
        # This would integrate with notification service
        logger.info(f"Sending notification to user {user_id}: {message}")
        await asyncio.sleep(0.5)  # Simulate notification
        return {"status": "sent", "user_id": user_id}
    
    async def _handle_update_analytics(self, event: str, data: Dict[str, Any], **kwargs):
        """Handle update analytics task"""
        # This would integrate with analytics service
        logger.info(f"Updating analytics for event {event}")
        await asyncio.sleep(0.5)  # Simulate analytics update
        return {"event": event, "updated": True}
    
    async def _handle_backup_data(self, backup_type: str, **kwargs):
        """Handle backup data task"""
        # This would perform data backup
        logger.info(f"Creating {backup_type} backup")
        await asyncio.sleep(5)  # Simulate backup
        return {"backup_id": str(uuid.uuid4()), "type": backup_type}

























