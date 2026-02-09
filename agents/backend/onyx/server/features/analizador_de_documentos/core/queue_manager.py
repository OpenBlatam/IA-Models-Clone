"""
Queue Manager for Document Analyzer
====================================

Advanced queue management for async task processing.
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
from collections import deque

logger = logging.getLogger(__name__)

class QueuePriority(Enum):
    """Queue priorities"""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3

@dataclass
class QueueItem:
    """Queue item"""
    item_id: str
    data: Any
    priority: QueuePriority = QueuePriority.NORMAL
    created_at: datetime = field(default_factory=datetime.now)
    attempts: int = 0
    max_attempts: int = 3

class QueueManager:
    """Advanced queue manager"""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.queues: Dict[str, deque] = {}
        self.workers: Dict[str, asyncio.Task] = {}
        self.is_processing: Dict[str, bool] = {}
        logger.info(f"QueueManager initialized. Max size: {max_size}")
    
    def create_queue(self, queue_name: str):
        """Create a new queue"""
        if queue_name not in self.queues:
            self.queues[queue_name] = deque(maxlen=self.max_size)
            self.is_processing[queue_name] = False
            logger.info(f"Created queue: {queue_name}")
    
    async def enqueue(
        self,
        queue_name: str,
        data: Any,
        priority: QueuePriority = QueuePriority.NORMAL,
        item_id: Optional[str] = None
    ) -> str:
        """Enqueue an item"""
        if queue_name not in self.queues:
            self.create_queue(queue_name)
        
        if len(self.queues[queue_name]) >= self.max_size:
            raise RuntimeError(f"Queue {queue_name} is full")
        
        item = QueueItem(
            item_id=item_id or f"item_{int(time.time())}_{len(self.queues[queue_name])}",
            data=data,
            priority=priority
        )
        
        # Insert based on priority
        queue = self.queues[queue_name]
        inserted = False
        for i, existing_item in enumerate(queue):
            if priority.value > existing_item.priority.value:
                queue.insert(i, item)
                inserted = True
                break
        
        if not inserted:
            queue.append(item)
        
        logger.debug(f"Enqueued item {item.item_id} to {queue_name}")
        return item.item_id
    
    async def dequeue(self, queue_name: str) -> Optional[QueueItem]:
        """Dequeue an item"""
        if queue_name not in self.queues or len(self.queues[queue_name]) == 0:
            return None
        
        return self.queues[queue_name].popleft()
    
    async def start_worker(
        self,
        queue_name: str,
        processor: Callable,
        concurrency: int = 1
    ):
        """Start a worker for a queue"""
        if queue_name in self.workers:
            logger.warning(f"Worker for {queue_name} already running")
            return
        
        self.is_processing[queue_name] = True
        
        async def worker_loop():
            semaphore = asyncio.Semaphore(concurrency)
            
            while self.is_processing[queue_name]:
                item = await self.dequeue(queue_name)
                
                if item is None:
                    await asyncio.sleep(0.1)
                    continue
                
                async with semaphore:
                    try:
                        if asyncio.iscoroutinefunction(processor):
                            await processor(item.data)
                        else:
                            await asyncio.to_thread(processor, item.data)
                    except Exception as e:
                        logger.error(f"Error processing item {item.item_id}: {e}")
                        item.attempts += 1
                        if item.attempts < item.max_attempts:
                            await self.enqueue(queue_name, item.data, item.priority, item.item_id)
        
        self.workers[queue_name] = asyncio.create_task(worker_loop())
        logger.info(f"Started worker for {queue_name}")
    
    def stop_worker(self, queue_name: str):
        """Stop a worker"""
        if queue_name in self.workers:
            self.is_processing[queue_name] = False
            self.workers[queue_name].cancel()
            del self.workers[queue_name]
            logger.info(f"Stopped worker for {queue_name}")
    
    def get_queue_size(self, queue_name: str) -> int:
        """Get queue size"""
        if queue_name not in self.queues:
            return 0
        return len(self.queues[queue_name])
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics"""
        return {
            queue_name: {
                "size": len(queue),
                "worker_running": queue_name in self.workers,
                "is_processing": self.is_processing.get(queue_name, False)
            }
            for queue_name, queue in self.queues.items()
        }

# Global instance
queue_manager = QueueManager()
















