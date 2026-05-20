"""
Advanced Queue System
=====================

Advanced queue system with priorities, scheduling, and persistence.
"""

import asyncio
import logging
import heapq
from typing import Dict, Any, Optional, List, Callable, Awaitable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


class QueuePriority(Enum):
    """Queue priorities."""
    LOW = 0
    NORMAL = 5
    HIGH = 10
    CRITICAL = 20


class QueueItemStatus(Enum):
    """Queue item status."""
    PENDING = "pending"
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
    max_retries: int = 0
    retry_count: int = 0
    status: QueueItemStatus = QueueItemStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __lt__(self, other):
        """Compare for priority queue."""
        if not isinstance(other, QueueItem):
            return NotImplemented
        
        # Higher priority first
        if self.priority.value != other.priority.value:
            return self.priority.value > other.priority.value
        
        # Earlier scheduled_at first
        if self.scheduled_at and other.scheduled_at:
            return self.scheduled_at < other.scheduled_at
        
        # Earlier created_at first
        return self.created_at < other.created_at


class AdvancedQueue:
    """Advanced queue with priorities and scheduling."""
    
    def __init__(self, name: str, max_size: Optional[int] = None):
        """
        Initialize advanced queue.
        
        Args:
            name: Queue name
            max_size: Optional maximum queue size
        """
        self.name = name
        self.max_size = max_size
        self.items: List[QueueItem] = []
        self.items_by_id: Dict[str, QueueItem] = {}
        self.lock = asyncio.Lock()
        self.processing: Dict[str, QueueItem] = {}
        self.completed: List[QueueItem] = []
        self.failed: List[QueueItem] = []
    
    async def enqueue(
        self,
        item_id: str,
        data: Any,
        priority: QueuePriority = QueuePriority.NORMAL,
        scheduled_at: Optional[datetime] = None,
        max_retries: int = 0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> QueueItem:
        """
        Enqueue item.
        
        Args:
            item_id: Item ID
            data: Item data
            priority: Item priority
            scheduled_at: Optional scheduled time
            max_retries: Maximum retry count
            metadata: Optional metadata
            
        Returns:
            Queue item
        """
        async with self.lock:
            # Check max size
            if self.max_size and len(self.items) >= self.max_size:
                raise ValueError(f"Queue {self.name} is full (max_size: {self.max_size})")
            
            # Check if item already exists
            if item_id in self.items_by_id:
                raise ValueError(f"Item {item_id} already exists in queue")
            
            item = QueueItem(
                id=item_id,
                data=data,
                priority=priority,
                scheduled_at=scheduled_at,
                max_retries=max_retries,
                metadata=metadata or {}
            )
            
            heapq.heappush(self.items, item)
            self.items_by_id[item_id] = item
            
            logger.info(f"Enqueued item {item_id} in queue {self.name}")
            return item
    
    async def dequeue(self, wait: bool = False) -> Optional[QueueItem]:
        """
        Dequeue item.
        
        Args:
            wait: Whether to wait if queue is empty
            
        Returns:
            Queue item or None
        """
        async with self.lock:
            while True:
                # Remove items that are scheduled for future
                now = datetime.now()
                ready_items = []
                
                while self.items:
                    item = heapq.heappop(self.items)
                    if item.scheduled_at and item.scheduled_at > now:
                        # Not ready yet, put back
                        heapq.heappush(self.items, item)
                        break
                    ready_items.append(item)
                
                # Put back items that weren't ready
                for item in ready_items:
                    heapq.heappush(self.items, item)
                
                if ready_items:
                    # Get highest priority item
                    item = heapq.heappop(self.items)
                    del self.items_by_id[item.id]
                    item.status = QueueItemStatus.PROCESSING
                    item.started_at = datetime.now()
                    self.processing[item.id] = item
                    return item
                
                if not wait:
                    return None
                
                # Wait a bit before checking again
                await asyncio.sleep(0.1)
    
    async def complete(self, item_id: str, result: Any = None):
        """
        Mark item as completed.
        
        Args:
            item_id: Item ID
            result: Optional result
        """
        async with self.lock:
            if item_id in self.processing:
                item = self.processing.pop(item_id)
                item.status = QueueItemStatus.COMPLETED
                item.completed_at = datetime.now()
                if result is not None:
                    item.metadata['result'] = result
                self.completed.append(item)
                logger.info(f"Completed item {item_id} in queue {self.name}")
    
    async def fail(self, item_id: str, error: str):
        """
        Mark item as failed.
        
        Args:
            item_id: Item ID
            error: Error message
        """
        async with self.lock:
            if item_id in self.processing:
                item = self.processing.pop(item_id)
                item.error = error
                
                # Check if should retry
                if item.retry_count < item.max_retries:
                    item.retry_count += 1
                    item.status = QueueItemStatus.PENDING
                    item.started_at = None
                    # Reschedule with delay
                    item.scheduled_at = datetime.now() + timedelta(seconds=2 ** item.retry_count)
                    heapq.heappush(self.items, item)
                    self.items_by_id[item.id] = item
                    logger.info(f"Retrying item {item_id} (attempt {item.retry_count}/{item.max_retries})")
                else:
                    item.status = QueueItemStatus.FAILED
                    item.completed_at = datetime.now()
                    self.failed.append(item)
                    logger.warning(f"Failed item {item_id} in queue {self.name}: {error}")
    
    async def cancel(self, item_id: str):
        """
        Cancel item.
        
        Args:
            item_id: Item ID
        """
        async with self.lock:
            # Remove from queue
            if item_id in self.items_by_id:
                item = self.items_by_id.pop(item_id)
                # Rebuild heap without this item
                self.items = [i for i in self.items if i.id != item_id]
                heapq.heapify(self.items)
                item.status = QueueItemStatus.CANCELLED
                logger.info(f"Cancelled item {item_id} in queue {self.name}")
            
            # Remove from processing
            if item_id in self.processing:
                item = self.processing.pop(item_id)
                item.status = QueueItemStatus.CANCELLED
                logger.info(f"Cancelled processing item {item_id} in queue {self.name}")
    
    def get_item(self, item_id: str) -> Optional[QueueItem]:
        """
        Get item by ID.
        
        Args:
            item_id: Item ID
            
        Returns:
            Queue item or None
        """
        return self.items_by_id.get(item_id) or self.processing.get(item_id)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get queue statistics.
        
        Returns:
            Statistics dictionary
        """
        return {
            "name": self.name,
            "pending": len(self.items),
            "processing": len(self.processing),
            "completed": len(self.completed),
            "failed": len(self.failed),
            "total": len(self.items) + len(self.processing) + len(self.completed) + len(self.failed),
            "max_size": self.max_size
        }



