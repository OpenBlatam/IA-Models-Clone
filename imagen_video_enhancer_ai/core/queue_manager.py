"""
Queue Management System
======================

Advanced queue management with priorities and scheduling.
"""

import asyncio
import logging
import heapq
from typing import Dict, Any, Optional, List, Callable, Awaitable
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class QueuePriority(Enum):
    """Queue priority levels."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    URGENT = 3


@dataclass
class QueueItem:
    """Queue item with priority."""
    id: str
    data: Any
    priority: QueuePriority = QueuePriority.NORMAL
    created_at: datetime = field(default_factory=datetime.now)
    scheduled_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __lt__(self, other):
        """Compare by priority and scheduled time."""
        if self.priority.value != other.priority.value:
            return self.priority.value > other.priority.value  # Higher priority first
        
        if self.scheduled_at and other.scheduled_at:
            return self.scheduled_at < other.scheduled_at
        
        return self.created_at < other.created_at


class QueueManager:
    """Advanced queue manager with priorities."""
    
    def __init__(self, name: str = "QueueManager"):
        """
        Initialize queue manager.
        
        Args:
            name: Queue name
        """
        self.name = name
        self.queue: List[QueueItem] = []
        self.processing: Dict[str, QueueItem] = {}
        self.completed: List[QueueItem] = []
        self.failed: List[QueueItem] = []
        self._lock = asyncio.Lock()
        self._condition = asyncio.Condition(self._lock)
        self.max_completed = 1000
    
    async def enqueue(
        self,
        item_id: str,
        data: Any,
        priority: QueuePriority = QueuePriority.NORMAL,
        scheduled_at: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Enqueue an item.
        
        Args:
            item_id: Item ID
            data: Item data
            priority: Item priority
            scheduled_at: Optional scheduled time
            metadata: Optional metadata
        """
        async with self._lock:
            item = QueueItem(
                id=item_id,
                data=data,
                priority=priority,
                scheduled_at=scheduled_at,
                metadata=metadata or {}
            )
            
            heapq.heappush(self.queue, item)
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
                    
                    self.processing[item.id] = item
                    return item
                
                # Wait for items
                if timeout:
                    try:
                        await asyncio.wait_for(self._condition.wait(), timeout=timeout)
                    except asyncio.TimeoutError:
                        return None
                else:
                    await self._condition.wait()
    
    async def complete(self, item_id: str):
        """
        Mark item as completed.
        
        Args:
            item_id: Item ID
        """
        async with self._lock:
            if item_id in self.processing:
                item = self.processing.pop(item_id)
                self.completed.append(item)
                
                # Limit completed items
                if len(self.completed) > self.max_completed:
                    self.completed = self.completed[-self.max_completed:]
                
                logger.info(f"Completed item {item_id}")
    
    async def fail(self, item_id: str, error: Optional[str] = None):
        """
        Mark item as failed.
        
        Args:
            item_id: Item ID
            error: Optional error message
        """
        async with self._lock:
            if item_id in self.processing:
                item = self.processing.pop(item_id)
                if error:
                    item.metadata["error"] = error
                self.failed.append(item)
                logger.warning(f"Failed item {item_id}: {error}")
    
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
            "failed": len(self.failed)
        }




