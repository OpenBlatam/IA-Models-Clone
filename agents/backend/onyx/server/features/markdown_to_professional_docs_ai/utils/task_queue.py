"""Task queue for asynchronous processing"""
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import uuid
import asyncio
import logging

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Task status"""
    PENDING = "pending"
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Task:
    """Task definition"""
    task_id: str
    task_type: str
    payload: Dict[str, Any]
    status: TaskStatus
    priority: int = 0
    created_at: datetime = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


class TaskQueue:
    """Task queue for asynchronous processing"""
    
    def __init__(self, max_workers: int = 4):
        """
        Initialize task queue
        
        Args:
            max_workers: Maximum number of concurrent workers
        """
        self.max_workers = max_workers
        self.tasks: Dict[str, Task] = {}
        self.queue: asyncio.Queue = asyncio.Queue()
        self.workers: list = []
        self.handlers: Dict[str, Callable] = {}
        self._register_default_handlers()
        self._start_workers()
    
    def _register_default_handlers(self):
        """Register default task handlers"""
        self.handlers["convert"] = self._handle_convert
        self.handlers["batch_convert"] = self._handle_batch_convert
        self.handlers["export"] = self._handle_export
    
    def _start_workers(self):
        """Start worker threads"""
        for i in range(self.max_workers):
            worker = asyncio.create_task(self._worker(f"worker-{i}"))
            self.workers.append(worker)
    
    async def _worker(self, worker_id: str):
        """Worker coroutine"""
        logger.info(f"Worker {worker_id} started")
        
        while True:
            try:
                task = await self.queue.get()
                
                if task is None:  # Shutdown signal
                    break
                
                task.status = TaskStatus.PROCESSING
                task.started_at = datetime.now()
                
                # Process task
                handler = self.handlers.get(task.task_type)
                if handler:
                    try:
                        result = await handler(task.payload)
                        task.status = TaskStatus.COMPLETED
                        task.result = result
                    except Exception as e:
                        task.status = TaskStatus.FAILED
                        task.error = str(e)
                        logger.error(f"Task {task.task_id} failed: {e}")
                else:
                    task.status = TaskStatus.FAILED
                    task.error = f"Unknown task type: {task.task_type}"
                
                task.completed_at = datetime.now()
                self.queue.task_done()
                
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
    
    async def enqueue(
        self,
        task_type: str,
        payload: Dict[str, Any],
        priority: int = 0
    ) -> str:
        """
        Enqueue a task
        
        Args:
            task_type: Task type
            payload: Task payload
            priority: Task priority (higher = more priority)
            
        Returns:
            Task ID
        """
        task_id = str(uuid.uuid4())
        task = Task(
            task_id=task_id,
            task_type=task_type,
            payload=payload,
            status=TaskStatus.QUEUED,
            priority=priority
        )
        
        self.tasks[task_id] = task
        await self.queue.put(task)
        
        return task_id
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """Get task by ID"""
        return self.tasks.get(task_id)
    
    def get_tasks(
        self,
        status: Optional[TaskStatus] = None,
        limit: int = 100
    ) -> list[Task]:
        """
        Get tasks
        
        Args:
            status: Optional status filter
            limit: Maximum number of tasks
            
        Returns:
            List of tasks
        """
        tasks = list(self.tasks.values())
        
        if status:
            tasks = [t for t in tasks if t.status == status]
        
        tasks.sort(key=lambda x: x.created_at, reverse=True)
        return tasks[:limit]
    
    async def _handle_convert(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle convert task"""
        from services.converter_service import ConverterService
        from services.markdown_parser import MarkdownParser
        
        markdown_content = payload.get("markdown_content", "")
        output_format = payload.get("output_format", "pdf")
        
        parser = MarkdownParser()
        parsed_content = parser.parse(markdown_content)
        
        converter = ConverterService()
        output_path = await converter.convert(
            parsed_content=parsed_content,
            output_format=output_format
        )
        
        return {
            "output_path": output_path,
            "format": output_format
        }
    
    async def _handle_batch_convert(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle batch convert task"""
        # Placeholder
        return {"status": "not_implemented"}
    
    async def _handle_export(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle export task"""
        # Placeholder
        return {"status": "not_implemented"}
    
    def register_handler(self, task_type: str, handler: Callable):
        """Register custom task handler"""
        self.handlers[task_type] = handler
    
    async def shutdown(self):
        """Shutdown task queue"""
        # Send shutdown signal to all workers
        for _ in range(self.max_workers):
            await self.queue.put(None)
        
        # Wait for workers to finish
        await asyncio.gather(*self.workers, return_exceptions=True)


# Global task queue
_task_queue: Optional[TaskQueue] = None


def get_task_queue() -> TaskQueue:
    """Get global task queue"""
    global _task_queue
    if _task_queue is None:
        _task_queue = TaskQueue()
    return _task_queue

