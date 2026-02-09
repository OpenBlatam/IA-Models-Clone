"""
Queue Service for Social Video Transcriber AI
Priority-based job queue with concurrency control
"""

import asyncio
import heapq
import logging
from typing import Optional, Dict, Any, List, Callable, Awaitable
from dataclasses import dataclass, field
from datetime import datetime
from uuid import UUID, uuid4
from enum import IntEnum

from ..config.settings import get_settings

logger = logging.getLogger(__name__)


class Priority(IntEnum):
    """Job priority levels (lower number = higher priority)"""
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BACKGROUND = 4


@dataclass(order=True)
class QueueItem:
    """Item in the priority queue"""
    priority: int
    timestamp: float = field(compare=True)
    job_id: UUID = field(compare=False)
    job_type: str = field(compare=False, default="transcription")
    payload: Dict[str, Any] = field(compare=False, default_factory=dict)
    callback: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = field(
        compare=False, default=None
    )
    created_at: datetime = field(compare=False, default_factory=datetime.utcnow)
    started_at: Optional[datetime] = field(compare=False, default=None)
    completed_at: Optional[datetime] = field(compare=False, default=None)
    retry_count: int = field(compare=False, default=0)
    max_retries: int = field(compare=False, default=3)
    error: Optional[str] = field(compare=False, default=None)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": str(self.job_id),
            "job_type": self.job_type,
            "priority": Priority(self.priority).name,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "retry_count": self.retry_count,
            "error": self.error,
        }


class QueueStatus:
    """Queue status tracking"""
    def __init__(self):
        self.pending = 0
        self.processing = 0
        self.completed = 0
        self.failed = 0
        self.total_processed = 0
        self.average_processing_time = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "pending": self.pending,
            "processing": self.processing,
            "completed": self.completed,
            "failed": self.failed,
            "total_processed": self.total_processed,
            "average_processing_time": round(self.average_processing_time, 2),
        }


class QueueService:
    """Priority-based job queue with concurrency control"""
    
    def __init__(
        self,
        max_concurrent: int = 3,
        max_queue_size: int = 1000,
    ):
        self.settings = get_settings()
        self.max_concurrent = max_concurrent
        self.max_queue_size = max_queue_size
        
        self._queue: List[QueueItem] = []
        self._processing: Dict[UUID, QueueItem] = {}
        self._completed: Dict[UUID, QueueItem] = {}
        self._failed: Dict[UUID, QueueItem] = {}
        
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._status = QueueStatus()
        self._running = False
        self._worker_task: Optional[asyncio.Task] = None
        
        self._handlers: Dict[str, Callable[[Dict[str, Any]], Awaitable[Any]]] = {}
    
    def register_handler(
        self,
        job_type: str,
        handler: Callable[[Dict[str, Any]], Awaitable[Any]],
    ):
        """Register a handler for a job type"""
        self._handlers[job_type] = handler
        logger.info(f"Registered handler for job type: {job_type}")
    
    async def enqueue(
        self,
        job_type: str,
        payload: Dict[str, Any],
        priority: Priority = Priority.NORMAL,
        callback: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None,
        max_retries: int = 3,
    ) -> UUID:
        """
        Add a job to the queue
        
        Args:
            job_type: Type of job
            payload: Job payload
            priority: Job priority
            callback: Optional completion callback
            max_retries: Maximum retry attempts
            
        Returns:
            Job ID
        """
        if len(self._queue) >= self.max_queue_size:
            raise ValueError(f"Queue full (max {self.max_queue_size})")
        
        job_id = uuid4()
        
        item = QueueItem(
            priority=priority.value,
            timestamp=datetime.utcnow().timestamp(),
            job_id=job_id,
            job_type=job_type,
            payload=payload,
            callback=callback,
            max_retries=max_retries,
        )
        
        heapq.heappush(self._queue, item)
        self._status.pending += 1
        
        logger.info(f"Enqueued job {job_id} ({job_type}) with priority {priority.name}")
        
        return job_id
    
    async def start(self):
        """Start the queue worker"""
        if self._running:
            return
        
        self._running = True
        self._worker_task = asyncio.create_task(self._worker_loop())
        logger.info("Queue worker started")
    
    async def stop(self, wait_for_completion: bool = True):
        """Stop the queue worker"""
        self._running = False
        
        if wait_for_completion and self._processing:
            logger.info("Waiting for processing jobs to complete...")
            while self._processing:
                await asyncio.sleep(0.5)
        
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Queue worker stopped")
    
    async def _worker_loop(self):
        """Main worker loop"""
        while self._running:
            if not self._queue:
                await asyncio.sleep(0.1)
                continue
            
            async with self._semaphore:
                if not self._queue:
                    continue
                
                item = heapq.heappop(self._queue)
                self._status.pending -= 1
                
                asyncio.create_task(self._process_item(item))
    
    async def _process_item(self, item: QueueItem):
        """Process a single queue item"""
        item.started_at = datetime.utcnow()
        self._processing[item.job_id] = item
        self._status.processing += 1
        
        logger.debug(f"Processing job {item.job_id}")
        
        try:
            handler = self._handlers.get(item.job_type)
            
            if not handler:
                raise ValueError(f"No handler for job type: {item.job_type}")
            
            result = await handler(item.payload)
            
            item.completed_at = datetime.utcnow()
            
            del self._processing[item.job_id]
            self._completed[item.job_id] = item
            
            self._status.processing -= 1
            self._status.completed += 1
            self._status.total_processed += 1
            
            processing_time = (item.completed_at - item.started_at).total_seconds()
            self._update_average_time(processing_time)
            
            logger.info(f"Job {item.job_id} completed in {processing_time:.2f}s")
            
            if item.callback:
                await item.callback({"status": "completed", "result": result})
            
        except Exception as e:
            logger.error(f"Job {item.job_id} failed: {e}")
            item.error = str(e)
            
            del self._processing[item.job_id]
            self._status.processing -= 1
            
            if item.retry_count < item.max_retries:
                item.retry_count += 1
                item.started_at = None
                item.error = None
                
                heapq.heappush(self._queue, item)
                self._status.pending += 1
                
                logger.info(f"Job {item.job_id} requeued (retry {item.retry_count})")
            else:
                self._failed[item.job_id] = item
                self._status.failed += 1
                
                if item.callback:
                    await item.callback({"status": "failed", "error": str(e)})
    
    def _update_average_time(self, new_time: float):
        """Update rolling average processing time"""
        total = self._status.total_processed
        current_avg = self._status.average_processing_time
        
        self._status.average_processing_time = (
            (current_avg * (total - 1) + new_time) / total
        )
    
    def get_job_status(self, job_id: UUID) -> Optional[Dict[str, Any]]:
        """Get status of a specific job"""
        for item in self._queue:
            if item.job_id == job_id:
                return {"status": "pending", **item.to_dict()}
        
        if job_id in self._processing:
            return {"status": "processing", **self._processing[job_id].to_dict()}
        
        if job_id in self._completed:
            return {"status": "completed", **self._completed[job_id].to_dict()}
        
        if job_id in self._failed:
            return {"status": "failed", **self._failed[job_id].to_dict()}
        
        return None
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get overall queue status"""
        return {
            "running": self._running,
            "max_concurrent": self.max_concurrent,
            "max_queue_size": self.max_queue_size,
            **self._status.to_dict(),
            "queue_by_priority": self._get_queue_by_priority(),
        }
    
    def _get_queue_by_priority(self) -> Dict[str, int]:
        """Get pending jobs count by priority"""
        counts = {p.name: 0 for p in Priority}
        
        for item in self._queue:
            priority_name = Priority(item.priority).name
            counts[priority_name] += 1
        
        return counts
    
    def cancel_job(self, job_id: UUID) -> bool:
        """Cancel a pending job"""
        for i, item in enumerate(self._queue):
            if item.job_id == job_id:
                self._queue.pop(i)
                heapq.heapify(self._queue)
                self._status.pending -= 1
                logger.info(f"Job {job_id} cancelled")
                return True
        
        return False
    
    def clear_completed(self, older_than_hours: int = 24):
        """Clear completed jobs older than specified hours"""
        cutoff = datetime.utcnow().timestamp() - (older_than_hours * 3600)
        
        cleared = 0
        to_remove = [
            job_id for job_id, item in self._completed.items()
            if item.completed_at and item.completed_at.timestamp() < cutoff
        ]
        
        for job_id in to_remove:
            del self._completed[job_id]
            cleared += 1
        
        logger.info(f"Cleared {cleared} completed jobs")
        return cleared


_queue_service: Optional[QueueService] = None


def get_queue_service() -> QueueService:
    """Get queue service singleton"""
    global _queue_service
    if _queue_service is None:
        _queue_service = QueueService()
    return _queue_service












