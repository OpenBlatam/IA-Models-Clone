"""
Queue Manager Service
Manages job queues for video generation
"""

import asyncio
from typing import Dict, Optional, Callable, Any
from uuid import UUID
from enum import Enum
import logging
from datetime import datetime
from collections import deque

logger = logging.getLogger(__name__)


class QueuePriority(str, Enum):
    """Queue priority levels"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class QueueManager:
    """Manages job queues with priority support"""
    
    def __init__(self, max_workers: int = 5):
        self.max_workers = max_workers
        self.queues: Dict[QueuePriority, deque] = {
            QueuePriority.LOW: deque(),
            QueuePriority.NORMAL: deque(),
            QueuePriority.HIGH: deque(),
            QueuePriority.URGENT: deque(),
        }
        self.processing: Dict[UUID, Dict[str, Any]] = {}
        self.completed: Dict[UUID, Dict[str, Any]] = {}
        self.workers: list = []
        self.running = False
    
    def enqueue(
        self,
        job_id: UUID,
        job_data: Dict[str, Any],
        priority: QueuePriority = QueuePriority.NORMAL
    ):
        """Add job to queue"""
        job = {
            "job_id": job_id,
            "data": job_data,
            "priority": priority,
            "enqueued_at": datetime.utcnow(),
        }
        self.queues[priority].append(job)
        logger.info(f"Job {job_id} enqueued with priority {priority.value}")
    
    def dequeue(self) -> Optional[Dict[str, Any]]:
        """Get next job from queue (highest priority first)"""
        for priority in [QueuePriority.URGENT, QueuePriority.HIGH, QueuePriority.NORMAL, QueuePriority.LOW]:
            if self.queues[priority]:
                job = self.queues[priority].popleft()
                self.processing[job["job_id"]] = {
                    **job,
                    "started_at": datetime.utcnow(),
                }
                return job
        return None
    
    def mark_completed(self, job_id: UUID, result: Any):
        """Mark job as completed"""
        if job_id in self.processing:
            job = self.processing.pop(job_id)
            self.completed[job_id] = {
                **job,
                "completed_at": datetime.utcnow(),
                "result": result,
            }
            logger.info(f"Job {job_id} completed")
    
    def mark_failed(self, job_id: UUID, error: str):
        """Mark job as failed"""
        if job_id in self.processing:
            job = self.processing.pop(job_id)
            self.completed[job_id] = {
                **job,
                "failed_at": datetime.utcnow(),
                "error": error,
            }
            logger.error(f"Job {job_id} failed: {error}")
    
    def get_status(self, job_id: UUID) -> Optional[Dict[str, Any]]:
        """Get job status"""
        if job_id in self.processing:
            return {
                "status": "processing",
                **self.processing[job_id],
            }
        elif job_id in self.completed:
            completed = self.completed[job_id]
            if "error" in completed:
                return {
                    "status": "failed",
                    **completed,
                }
            else:
                return {
                    "status": "completed",
                    **completed,
                }
        else:
            # Check if in any queue
            for queue in self.queues.values():
                for job in queue:
                    if job["job_id"] == job_id:
                        return {
                            "status": "queued",
                            **job,
                        }
        return None
    
    async def start_workers(self, processor: Callable):
        """Start worker processes"""
        if self.running:
            return
        
        self.running = True
        self.workers = [
            asyncio.create_task(self._worker(processor, i))
            for i in range(self.max_workers)
        ]
        logger.info(f"Started {self.max_workers} queue workers")
    
    async def stop_workers(self):
        """Stop worker processes"""
        self.running = False
        if self.workers:
            await asyncio.gather(*self.workers, return_exceptions=True)
            self.workers.clear()
        logger.info("Stopped queue workers")
    
    async def _worker(self, processor: Callable, worker_id: int):
        """Worker process"""
        logger.info(f"Worker {worker_id} started")
        
        while self.running:
            job = self.dequeue()
            if job:
                try:
                    result = await processor(job["job_id"], job["data"])
                    self.mark_completed(job["job_id"], result)
                except Exception as e:
                    logger.error(f"Worker {worker_id} error: {str(e)}")
                    self.mark_failed(job["job_id"], str(e))
            else:
                await asyncio.sleep(1)  # Wait before checking again
        
        logger.info(f"Worker {worker_id} stopped")
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics"""
        return {
            "queued": {
                priority.value: len(queue)
                for priority, queue in self.queues.items()
            },
            "processing": len(self.processing),
            "completed": len([j for j in self.completed.values() if "error" not in j]),
            "failed": len([j for j in self.completed.values() if "error" in j]),
            "workers": len(self.workers),
            "running": self.running,
        }


_queue_manager: Optional[QueueManager] = None


def get_queue_manager(max_workers: int = 5) -> QueueManager:
    """Get queue manager instance (singleton)"""
    global _queue_manager
    if _queue_manager is None:
        _queue_manager = QueueManager(max_workers=max_workers)
    return _queue_manager

