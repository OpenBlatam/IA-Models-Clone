"""
Advanced Queue
==============

Advanced task queue system with priorities, scheduling, and persistence.
"""

import asyncio
import logging
import heapq
from typing import Dict, Any, Optional, List, Callable, Awaitable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from collections import deque

logger = logging.getLogger(__name__)


class QueuePriority(Enum):
    """Queue priority."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    URGENT = 3
    CRITICAL = 4


class QueueStatus(Enum):
    """Queue item status."""
    PENDING = "pending"
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class QueueItem:
    """Queue item."""
    id: str
    data: Any
    priority: QueuePriority = QueuePriority.NORMAL
    scheduled_at: Optional[datetime] = None
    status: QueueStatus = QueueStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Any = None
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __lt__(self, other):
        """Compare by priority and scheduled time."""
        if self.priority.value != other.priority.value:
            return self.priority.value > other.priority.value
        
        if self.scheduled_at and other.scheduled_at:
            return self.scheduled_at < other.scheduled_at
        
        return self.created_at < other.created_at


class AdvancedQueue:
    """Advanced task queue."""
    
    def __init__(self, name: str = "Queue"):
        """
        Initialize advanced queue.
        
        Args:
            name: Queue name
        """
        self.name = name
        self.queue: List[QueueItem] = []
        self.processing: Dict[str, QueueItem] = {}
        self.completed: deque = deque(maxlen=1000)
        self.failed: deque = deque(maxlen=1000)
        self._lock = asyncio.Lock()
        self._condition = asyncio.Condition(self._lock)
        self.stats = {
            "total_enqueued": 0,
            "total_processed": 0,
            "total_completed": 0,
            "total_failed": 0
        }
    
    async def enqueue(
        self,
        item_id: str,
        data: Any,
        priority: QueuePriority = QueuePriority.NORMAL,
        scheduled_at: Optional[datetime] = None,
        max_retries: int = 0,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Enqueue an item.
        
        Args:
            item_id: Item ID
            data: Item data
            priority: Item priority
            scheduled_at: Optional scheduled time
            max_retries: Maximum retry count
            metadata: Optional metadata
        """
        async with self._lock:
            item = QueueItem(
                id=item_id,
                data=data,
                priority=priority,
                scheduled_at=scheduled_at,
                max_retries=max_retries,
                metadata=metadata or {}
            )
            
            heapq.heappush(self.queue, item)
            item.status = QueueStatus.QUEUED
            self.stats["total_enqueued"] += 1
            self._condition.notify_all()
            
            logger.info(f"Enqueued item {item_id} with priority {priority.value}")
    
    async def dequeue(self, timeout: Optional[float] = None) -> Optional[QueueItem]:
        """
        Dequeue an item (highest priority first).
        
        Args:
            timeout: Optional timeout in seconds
            
        Returns:
            Queue item or None
        """
        async with self._condition:
            while True:
                # Check for available items
                now = datetime.now()
                available_items = [
                    item for item in self.queue
                    if item.scheduled_at is None or item.scheduled_at <= now
                ]
                
                if available_items:
                    # Get highest priority item
                    item = min(available_items, key=lambda x: (x.priority.value, x.created_at))
                    self.queue.remove(item)
                    heapq.heapify(self.queue)
                    
                    item.status = QueueStatus.PROCESSING
                    item.started_at = datetime.now()
                    self.processing[item.id] = item
                    self.stats["total_processed"] += 1
                    return item
                
                # Wait for items
                if timeout:
                    try:
                        await asyncio.wait_for(self._condition.wait(), timeout=timeout)
                    except asyncio.TimeoutError:
                        return None
                else:
                    await self._condition.wait()
    
    async def complete(self, item_id: str, result: Any = None):
        """
        Mark item as completed.
        
        Args:
            item_id: Item ID
            result: Optional result
        """
        async with self._lock:
            if item_id in self.processing:
                item = self.processing.pop(item_id)
                item.status = QueueStatus.COMPLETED
                item.completed_at = datetime.now()
                item.result = result
                self.completed.append(item)
                self.stats["total_completed"] += 1
                logger.info(f"Completed item {item_id}")
    
    async def fail(self, item_id: str, error: str, retry: bool = False):
        """
        Mark item as failed.
        
        Args:
            item_id: Item ID
            error: Error message
            retry: Whether to retry
        """
        async with self._lock:
            if item_id in self.processing:
                item = self.processing.pop(item_id)
                item.error = error
                
                if retry and item.retry_count < item.max_retries:
                    item.retry_count += 1
                    item.status = QueueStatus.QUEUED
                    item.started_at = None
                    heapq.heappush(self.queue, item)
                    logger.info(f"Retrying item {item_id} (attempt {item.retry_count}/{item.max_retries})")
                else:
                    item.status = QueueStatus.FAILED
                    item.completed_at = datetime.now()
                    self.failed.append(item)
                    self.stats["total_failed"] += 1
                    logger.warning(f"Failed item {item_id}: {error}")
    
    async def cancel(self, item_id: str) -> bool:
        """
        Cancel an item.
        
        Args:
            item_id: Item ID
            
        Returns:
            True if cancelled
        """
        async with self._lock:
            # Check in queue
            for item in self.queue:
                if item.id == item_id:
                    self.queue.remove(item)
                    heapq.heapify(self.queue)
                    item.status = QueueStatus.CANCELLED
                    return True
            
            # Check in processing
            if item_id in self.processing:
                item = self.processing.pop(item_id)
                item.status = QueueStatus.CANCELLED
                return True
            
            return False
    
    def get_size(self) -> int:
        """Get queue size."""
        return len(self.queue)
    
    def get_processing_count(self) -> int:
        """Get number of items being processed."""
        return len(self.processing)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        return {
            "name": self.name,
            "queue_size": len(self.queue),
            "processing": len(self.processing),
            "completed": len(self.completed),
            "failed": len(self.failed),
            **self.stats
        }
    
    def get_item(self, item_id: str) -> Optional[QueueItem]:
        """Get item by ID."""
        # Check in queue
        for item in self.queue:
            if item.id == item_id:
                return item
        
        # Check in processing
        if item_id in self.processing:
            return self.processing[item_id]
        
        # Check in completed
        for item in self.completed:
            if item.id == item_id:
                return item
        
        # Check in failed
        for item in self.failed:
            if item.id == item_id:
                return item
        
        return None

