"""
Queue Manager
=============

Advanced message queue management.
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class MessagePriority(Enum):
    """Message priority."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class Message:
    """Message definition."""
    id: str
    queue: str
    payload: Any
    priority: MessagePriority = MessagePriority.NORMAL
    created_at: datetime = None
    attempts: int = 0
    max_attempts: int = 3
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.metadata is None:
            self.metadata = {}


class QueueManager:
    """Advanced message queue manager."""
    
    def __init__(self):
        self._queues: Dict[str, asyncio.Queue] = {}
        self._consumers: Dict[str, List[asyncio.Task]] = {}
        self._handlers: Dict[str, Callable] = {}
        self._processing: Dict[str, bool] = {}
    
    def create_queue(self, queue_name: str, maxsize: int = 0):
        """Create queue."""
        self._queues[queue_name] = asyncio.Queue(maxsize=maxsize)
        self._consumers[queue_name] = []
        self._processing[queue_name] = False
        logger.info(f"Created queue: {queue_name}")
    
    def register_handler(self, queue_name: str, handler: Callable):
        """Register message handler."""
        self._handlers[queue_name] = handler
        logger.info(f"Registered handler for queue: {queue_name}")
    
    async def publish(
        self,
        queue_name: str,
        payload: Any,
        priority: MessagePriority = MessagePriority.NORMAL,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Publish message to queue."""
        if queue_name not in self._queues:
            self.create_queue(queue_name)
        
        import uuid
        message_id = str(uuid.uuid4())
        
        message = Message(
            id=message_id,
            queue=queue_name,
            payload=payload,
            priority=priority,
            metadata=metadata or {}
        )
        
        await self._queues[queue_name].put(message)
        logger.debug(f"Published message {message_id} to {queue_name}")
        return message_id
    
    async def consume(self, queue_name: str, timeout: Optional[float] = None) -> Optional[Message]:
        """Consume message from queue."""
        if queue_name not in self._queues:
            return None
        
        try:
            message = await asyncio.wait_for(
                self._queues[queue_name].get(),
                timeout=timeout
            )
            return message
        except asyncio.TimeoutError:
            return None
    
    def start_consuming(self, queue_name: str, concurrency: int = 1):
        """Start consuming messages."""
        if queue_name not in self._queues:
            raise ValueError(f"Queue {queue_name} not found")
        
        if queue_name not in self._handlers:
            raise ValueError(f"No handler registered for queue {queue_name}")
        
        if self._processing[queue_name]:
            return
        
        self._processing[queue_name] = True
        handler = self._handlers[queue_name]
        
        async def consumer():
            while self._processing[queue_name]:
                message = await self.consume(queue_name)
                if message:
                    try:
                        if asyncio.iscoroutinefunction(handler):
                            await handler(message)
                        else:
                            handler(message)
                    except Exception as e:
                        logger.error(f"Error processing message {message.id}: {e}")
                        message.attempts += 1
                        if message.attempts < message.max_attempts:
                            # Retry
                            await self._queues[queue_name].put(message)
        
        for _ in range(concurrency):
            task = asyncio.create_task(consumer())
            self._consumers[queue_name].append(task)
        
        logger.info(f"Started consuming from {queue_name} with {concurrency} workers")
    
    def stop_consuming(self, queue_name: str):
        """Stop consuming messages."""
        self._processing[queue_name] = False
        
        # Cancel consumer tasks
        for task in self._consumers.get(queue_name, []):
            task.cancel()
        
        self._consumers[queue_name] = []
        logger.info(f"Stopped consuming from {queue_name}")
    
    def get_queue_size(self, queue_name: str) -> int:
        """Get queue size."""
        if queue_name not in self._queues:
            return 0
        return self._queues[queue_name].qsize()
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        return {
            "total_queues": len(self._queues),
            "queues": {
                queue_name: {
                    "size": self.get_queue_size(queue_name),
                    "processing": self._processing.get(queue_name, False),
                    "consumers": len(self._consumers.get(queue_name, []))
                }
                for queue_name in self._queues.keys()
            }
        }















