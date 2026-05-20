"""
Queue Manager
============

Advanced queue management system for bulk document generation.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import uuid
import json
from dataclasses import dataclass, asdict
from enum import Enum

from ..models.schemas import BulkGenerationRequest, TaskStatus, TaskInfo
from ..utils.redis_client import RedisClient
from ..utils.database_client import DatabaseClient

logger = logging.getLogger(__name__)

class QueuePriority(Enum):
    """Queue priority levels."""
    LOW = 1
    NORMAL = 5
    HIGH = 8
    URGENT = 10

@dataclass
class QueueTask:
    """Queue task representation."""
    task_id: str
    request: BulkGenerationRequest
    status: TaskStatus
    priority: int
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: float = 0.0
    documents_generated: int = 0
    errors: List[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.metadata is None:
            self.metadata = {}

class QueueManager:
    """
    Advanced queue management system for bulk document generation.
    
    Features:
    - Priority-based queuing
    - Task lifecycle management
    - Progress tracking
    - Error handling
    - Performance monitoring
    """
    
    def __init__(self):
        self.redis_client = RedisClient()
        self.database_client = DatabaseClient()
        self.active_tasks = {}
        self.task_queue = asyncio.PriorityQueue()
        self.max_concurrent_tasks = 5
        self.task_timeout = 3600  # 1 hour
        self.cleanup_interval = 300  # 5 minutes
        
    async def initialize(self):
        """Initialize the queue manager."""
        logger.info("Initializing Queue Manager...")
        
        try:
            await self.redis_client.initialize()
            await self.database_client.initialize()
            
            # Start background tasks
            asyncio.create_task(self._cleanup_old_tasks())
            asyncio.create_task(self._monitor_task_health())
            
            logger.info("Queue Manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Queue Manager: {str(e)}")
            raise
    
    async def create_generation_task(self, request: BulkGenerationRequest) -> str:
        """
        Create a new generation task.
        
        Args:
            request: Bulk generation request
            
        Returns:
            Task ID
        """
        try:
            task_id = str(uuid.uuid4())
            
            # Create queue task
            task = QueueTask(
                task_id=task_id,
                request=request,
                status=TaskStatus.CREATED,
                priority=request.priority,
                created_at=datetime.utcnow(),
                metadata={
                    "max_documents": request.config.max_documents,
                    "estimated_duration": request.config.estimated_duration,
                    "optimization_level": request.config.optimization_level.value
                }
            )
            
            # Store in database
            await self._store_task(task)
            
            # Add to queue
            await self._add_to_queue(task)
            
            logger.info(f"Created generation task: {task_id}")
            return task_id
            
        except Exception as e:
            logger.error(f"Failed to create generation task: {str(e)}")
            raise
    
    async def _add_to_queue(self, task: QueueTask):
        """Add task to processing queue."""
        try:
            # Calculate priority score (higher is better)
            priority_score = task.priority * 1000 - int(task.created_at.timestamp())
            
            await self.task_queue.put((priority_score, task))
            logger.info(f"Added task {task.task_id} to queue with priority {priority_score}")
            
        except Exception as e:
            logger.error(f"Failed to add task to queue: {str(e)}")
            raise
    
    async def get_next_task(self) -> Optional[QueueTask]:
        """Get next task from queue."""
        try:
            if self.task_queue.empty():
                return None
            
            # Check if we can process more tasks
            if len(self.active_tasks) >= self.max_concurrent_tasks:
                return None
            
            # Get task from queue
            priority_score, task = await self.task_queue.get()
            
            # Update task status
            task.status = TaskStatus.STARTED
            task.started_at = datetime.utcnow()
            
            # Store in active tasks
            self.active_tasks[task.task_id] = task
            
            # Update in database
            await self._update_task(task)
            
            logger.info(f"Retrieved task {task.task_id} from queue")
            return task
            
        except Exception as e:
            logger.error(f"Failed to get next task: {str(e)}")
            return None
    
    async def update_task_status(self, task_id: str, status: TaskStatus):
        """Update task status."""
        try:
            task = await self._get_task(task_id)
            if not task:
                logger.warning(f"Task {task_id} not found")
                return
            
            task.status = status
            
            if status == TaskStatus.COMPLETED:
                task.completed_at = datetime.utcnow()
                if task_id in self.active_tasks:
                    del self.active_tasks[task_id]
            
            await self._update_task(task)
            logger.info(f"Updated task {task_id} status to {status}")
            
        except Exception as e:
            logger.error(f"Failed to update task status: {str(e)}")
    
    async def update_task_progress(self, task_id: str, completed: int, total: int):
        """Update task progress."""
        try:
            task = await self._get_task(task_id)
            if not task:
                logger.warning(f"Task {task_id} not found")
                return
            
            task.documents_generated = completed
            task.progress = (completed / total) * 100 if total > 0 else 0
            
            await self._update_task(task)
            logger.debug(f"Updated task {task_id} progress: {completed}/{total} ({task.progress:.1f}%)")
            
        except Exception as e:
            logger.error(f"Failed to update task progress: {str(e)}")
    
    async def record_generation_error(self, task_id: str, error: str):
        """Record generation error."""
        try:
            task = await self._get_task(task_id)
            if not task:
                logger.warning(f"Task {task_id} not found")
                return
            
            task.errors.append(f"{datetime.utcnow().isoformat()}: {error}")
            await self._update_task(task)
            
            logger.error(f"Recorded error for task {task_id}: {error}")
            
        except Exception as e:
            logger.error(f"Failed to record generation error: {str(e)}")
    
    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task status."""
        try:
            task = await self._get_task(task_id)
            if not task:
                return None
            
            return {
                "task_id": task.task_id,
                "status": task.status.value,
                "progress": task.progress,
                "documents_generated": task.documents_generated,
                "total_documents": task.metadata.get("max_documents", 0),
                "created_at": task.created_at.isoformat(),
                "started_at": task.started_at.isoformat() if task.started_at else None,
                "completed_at": task.completed_at.isoformat() if task.completed_at else None,
                "errors": task.errors,
                "metadata": task.metadata
            }
            
        except Exception as e:
            logger.error(f"Failed to get task status: {str(e)}")
            return None
    
    async def list_tasks(
        self, 
        status: Optional[TaskStatus] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """List tasks with optional filtering."""
        try:
            tasks = await self._get_tasks(status, limit, offset)
            
            return [
                {
                    "task_id": task.task_id,
                    "status": task.status.value,
                    "progress": task.progress,
                    "documents_generated": task.documents_generated,
                    "total_documents": task.metadata.get("max_documents", 0),
                    "created_at": task.created_at.isoformat(),
                    "started_at": task.started_at.isoformat() if task.started_at else None,
                    "completed_at": task.completed_at.isoformat() if task.completed_at else None,
                    "priority": task.priority,
                    "errors_count": len(task.errors)
                }
                for task in tasks
            ]
            
        except Exception as e:
            logger.error(f"Failed to list tasks: {str(e)}")
            return []
    
    async def stop_task(self, task_id: str) -> bool:
        """Stop a task."""
        try:
            task = await self._get_task(task_id)
            if not task:
                return False
            
            if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                return False
            
            task.status = TaskStatus.CANCELLED
            task.completed_at = datetime.utcnow()
            
            if task_id in self.active_tasks:
                del self.active_tasks[task_id]
            
            await self._update_task(task)
            
            logger.info(f"Stopped task {task_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop task: {str(e)}")
            return False
    
    async def should_continue_generation(self, task_id: str) -> bool:
        """Check if generation should continue."""
        try:
            task = await self._get_task(task_id)
            if not task:
                return False
            
            # Check if task is cancelled or failed
            if task.status in [TaskStatus.CANCELLED, TaskStatus.FAILED]:
                return False
            
            # Check if task has reached max documents
            max_documents = task.metadata.get("max_documents", 0)
            if task.documents_generated >= max_documents:
                return False
            
            # Check if task has timed out
            if task.started_at:
                elapsed = datetime.utcnow() - task.started_at
                if elapsed.total_seconds() > self.task_timeout:
                    logger.warning(f"Task {task_id} timed out")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to check if generation should continue: {str(e)}")
            return False
    
    async def _store_task(self, task: QueueTask):
        """Store task in database."""
        try:
            task_data = asdict(task)
            task_data["created_at"] = task.created_at.isoformat()
            task_data["started_at"] = task.started_at.isoformat() if task.started_at else None
            task_data["completed_at"] = task.completed_at.isoformat() if task.completed_at else None
            
            await self.database_client.store_task(task.task_id, task_data)
            
        except Exception as e:
            logger.error(f"Failed to store task: {str(e)}")
            raise
    
    async def _get_task(self, task_id: str) -> Optional[QueueTask]:
        """Get task from database."""
        try:
            task_data = await self.database_client.get_task(task_id)
            if not task_data:
                return None
            
            # Convert back to QueueTask
            task = QueueTask(
                task_id=task_data["task_id"],
                request=BulkGenerationRequest(**task_data["request"]),
                status=TaskStatus(task_data["status"]),
                priority=task_data["priority"],
                created_at=datetime.fromisoformat(task_data["created_at"]),
                started_at=datetime.fromisoformat(task_data["started_at"]) if task_data.get("started_at") else None,
                completed_at=datetime.fromisoformat(task_data["completed_at"]) if task_data.get("completed_at") else None,
                progress=task_data.get("progress", 0.0),
                documents_generated=task_data.get("documents_generated", 0),
                errors=task_data.get("errors", []),
                metadata=task_data.get("metadata", {})
            )
            
            return task
            
        except Exception as e:
            logger.error(f"Failed to get task: {str(e)}")
            return None
    
    async def _update_task(self, task: QueueTask):
        """Update task in database."""
        try:
            await self._store_task(task)
            
        except Exception as e:
            logger.error(f"Failed to update task: {str(e)}")
            raise
    
    async def _get_tasks(
        self, 
        status: Optional[TaskStatus] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[QueueTask]:
        """Get tasks from database."""
        try:
            tasks_data = await self.database_client.get_tasks(status.value if status else None, limit, offset)
            
            tasks = []
            for task_data in tasks_data:
                try:
                    task = QueueTask(
                        task_id=task_data["task_id"],
                        request=BulkGenerationRequest(**task_data["request"]),
                        status=TaskStatus(task_data["status"]),
                        priority=task_data["priority"],
                        created_at=datetime.fromisoformat(task_data["created_at"]),
                        started_at=datetime.fromisoformat(task_data["started_at"]) if task_data.get("started_at") else None,
                        completed_at=datetime.fromisoformat(task_data["completed_at"]) if task_data.get("completed_at") else None,
                        progress=task_data.get("progress", 0.0),
                        documents_generated=task_data.get("documents_generated", 0),
                        errors=task_data.get("errors", []),
                        metadata=task_data.get("metadata", {})
                    )
                    tasks.append(task)
                except Exception as e:
                    logger.warning(f"Failed to parse task data: {str(e)}")
                    continue
            
            return tasks
            
        except Exception as e:
            logger.error(f"Failed to get tasks: {str(e)}")
            return []
    
    async def _cleanup_old_tasks(self):
        """Cleanup old completed tasks."""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval)
                
                # Get old completed tasks
                cutoff_time = datetime.utcnow() - timedelta(hours=24)
                old_tasks = await self.database_client.get_old_tasks(cutoff_time)
                
                for task_data in old_tasks:
                    await self.database_client.delete_task(task_data["task_id"])
                
                if old_tasks:
                    logger.info(f"Cleaned up {len(old_tasks)} old tasks")
                
            except Exception as e:
                logger.error(f"Failed to cleanup old tasks: {str(e)}")
    
    async def _monitor_task_health(self):
        """Monitor task health and handle timeouts."""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                current_time = datetime.utcnow()
                timed_out_tasks = []
                
                for task_id, task in self.active_tasks.items():
                    if task.started_at:
                        elapsed = current_time - task.started_at
                        if elapsed.total_seconds() > self.task_timeout:
                            timed_out_tasks.append(task_id)
                
                # Handle timed out tasks
                for task_id in timed_out_tasks:
                    logger.warning(f"Task {task_id} timed out")
                    await self.update_task_status(task_id, TaskStatus.FAILED)
                    await self.record_generation_error(task_id, "Task timed out")
                
            except Exception as e:
                logger.error(f"Failed to monitor task health: {str(e)}")
    
    async def get_queue_statistics(self) -> Dict[str, Any]:
        """Get queue statistics."""
        try:
            total_tasks = await self.database_client.count_tasks()
            active_tasks = len(self.active_tasks)
            queue_size = self.task_queue.qsize()
            
            return {
                "total_tasks": total_tasks,
                "active_tasks": active_tasks,
                "queue_size": queue_size,
                "max_concurrent_tasks": self.max_concurrent_tasks,
                "task_timeout": self.task_timeout
            }
            
        except Exception as e:
            logger.error(f"Failed to get queue statistics: {str(e)}")
            return {}
    
    async def cleanup(self):
        """Cleanup resources."""
        try:
            await self.redis_client.cleanup()
            await self.database_client.cleanup()
            logger.info("Queue Manager cleaned up successfully")
        except Exception as e:
            logger.error(f"Failed to cleanup Queue Manager: {str(e)}")











