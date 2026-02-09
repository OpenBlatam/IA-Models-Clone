"""
Message Queue Service
=====================
Service for message queue with pub/sub and topic support
"""

import asyncio
import logging
import time
import uuid
from typing import Dict, Any, List, Optional, Callable, Awaitable
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict, deque
from enum import Enum

logger = logging.getLogger(__name__)


class MessagePriority(Enum):
    """Message priority"""
    LOW = 3
    NORMAL = 2
    HIGH = 1
    CRITICAL = 0


class MessageStatus(Enum):
    """Message status"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    DEAD_LETTER = "dead_letter"


@dataclass
class Message:
    """Message in queue"""
    id: str
    topic: str
    payload: Dict[str, Any]
    priority: MessagePriority = MessagePriority.NORMAL
    status: MessageStatus = MessageStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    processed_at: Optional[datetime] = None
    retry_count: int = 0
    max_retries: int = 3
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __lt__(self, other):
        """Compare by priority"""
        if self.priority.value != other.priority.value:
            return self.priority.value < other.priority.value
        return self.created_at < other.created_at


class MessageQueueService:
    """
    Service for message queue with topics and priorities.
    
    Features:
    - Topic-based queues
    - Priority support
    - Multiple consumers per topic
    - Retry logic
    - Dead letter queue
    - Statistics
    """
    
    def __init__(self):
        """Initialize message queue service"""
        self._queues: Dict[str, List[Message]] = defaultdict(list)  # topic -> messages
        self._consumers: Dict[str, List[Callable[[Message], Awaitable[None]]]] = defaultdict(list)
        self._dead_letter_queue: List[Message] = []
        self._processing: Dict[str, Message] = {}  # message_id -> message
        self._workers: Dict[str, List[asyncio.Task]] = defaultdict(list)
        self._running = False
        self._stats = {
            'published': 0,
            'consumed': 0,
            'failed': 0,
            'dead_lettered': 0
        }
    
    def publish(
        self,
        topic: str,
        payload: Dict[str, Any],
        priority: MessagePriority = MessagePriority.NORMAL,
        message_id: Optional[str] = None,
        max_retries: int = 3
    ) -> Message:
        """
        Publish message to topic.
        
        Args:
            topic: Topic name
            payload: Message payload
            priority: Message priority
            message_id: Optional message ID
            max_retries: Maximum retry attempts
        
        Returns:
            Message object
        """
        if message_id is None:
            message_id = f"msg_{uuid.uuid4().hex[:12]}"
        
        message = Message(
            id=message_id,
            topic=topic,
            payload=payload,
            priority=priority,
            max_retries=max_retries
        )
        
        # Add to queue (maintain priority order)
        queue = self._queues[topic]
        # Simple insertion sort by priority
        inserted = False
        for i, existing in enumerate(queue):
            if message < existing:
                queue.insert(i, message)
                inserted = True
                break
        
        if not inserted:
            queue.append(message)
        
        self._stats['published'] += 1
        logger.info(f"Message published to topic '{topic}': {message_id}")
        
        # Notify workers if consumers exist
        if topic in self._consumers and self._consumers[topic]:
            self._notify_workers(topic)
        
        return message
    
    def subscribe(
        self,
        topic: str,
        handler: Callable[[Message], Awaitable[None]],
        num_workers: int = 1
    ):
        """
        Subscribe to topic with handler.
        
        Args:
            topic: Topic name
            handler: Async handler function
            num_workers: Number of worker tasks
        """
        self._consumers[topic].append(handler)
        
        # Start workers if service is running
        if self._running:
            for i in range(num_workers):
                worker = asyncio.create_task(self._worker(topic, handler, f"worker-{i}"))
                self._workers[topic].append(worker)
        
        logger.info(f"Subscribed to topic '{topic}' with {num_workers} worker(s)")
    
    def unsubscribe(self, topic: str, handler: Optional[Callable] = None):
        """
        Unsubscribe from topic.
        
        Args:
            topic: Topic name
            handler: Optional specific handler to remove
        """
        if handler:
            if handler in self._consumers[topic]:
                self._consumers[topic].remove(handler)
                logger.info(f"Unsubscribed handler from topic '{topic}'")
        else:
            # Remove all handlers
            if topic in self._consumers:
                del self._consumers[topic]
            # Stop workers
            for worker in self._workers.get(topic, []):
                worker.cancel()
            if topic in self._workers:
                del self._workers[topic]
            logger.info(f"Unsubscribed from topic '{topic}'")
    
    async def start(self):
        """Start message queue workers"""
        if self._running:
            return
        
        self._running = True
        
        # Start workers for all subscribed topics
        for topic, handlers in self._consumers.items():
            for handler in handlers:
                worker = asyncio.create_task(self._worker(topic, handler, f"worker-{topic}"))
                self._workers[topic].append(worker)
        
        logger.info("Message queue service started")
    
    async def stop(self):
        """Stop message queue workers"""
        if not self._running:
            return
        
        self._running = False
        
        # Cancel all workers
        all_workers = []
        for workers in self._workers.values():
            all_workers.extend(workers)
        
        if all_workers:
            await asyncio.gather(*all_workers, return_exceptions=True)
        
        self._workers.clear()
        logger.info("Message queue service stopped")
    
    def _notify_workers(self, topic: str):
        """Notify workers that messages are available"""
        # Workers check queue automatically
        pass
    
    async def _worker(
        self,
        topic: str,
        handler: Callable[[Message], Awaitable[None]],
        worker_name: str
    ):
        """Worker that processes messages"""
        logger.info(f"Message queue worker '{worker_name}' started for topic '{topic}'")
        
        while self._running:
            message = None
            
            # Get message from queue
            if topic in self._queues and self._queues[topic]:
                message = self._queues[topic].pop(0)
            
            if message is None:
                await asyncio.sleep(0.1)
                continue
            
            # Process message
            message.status = MessageStatus.PROCESSING
            self._processing[message.id] = message
            
            try:
                await handler(message)
                message.status = MessageStatus.COMPLETED
                message.processed_at = datetime.now()
                self._stats['consumed'] += 1
                logger.debug(f"Message {message.id} processed by {worker_name}")
            
            except Exception as e:
                message.error = str(e)
                message.retry_count += 1
                
                if message.retry_count < message.max_retries:
                    # Re-queue for retry
                    message.status = MessageStatus.PENDING
                    queue = self._queues[topic]
                    inserted = False
                    for i, existing in enumerate(queue):
                        if message < existing:
                            queue.insert(i, message)
                            inserted = True
                            break
                    if not inserted:
                        queue.append(message)
                    logger.warning(f"Message {message.id} failed, retrying ({message.retry_count}/{message.max_retries})")
                else:
                    # Move to dead letter queue
                    message.status = MessageStatus.DEAD_LETTER
                    self._dead_letter_queue.append(message)
                    self._stats['dead_lettered'] += 1
                    logger.error(f"Message {message.id} failed after {message.retry_count} attempts, moved to DLQ")
            
            finally:
                if message.id in self._processing:
                    del self._processing[message.id]
    
    def get_queue_size(self, topic: Optional[str] = None) -> int:
        """Get queue size for topic or total"""
        if topic:
            return len(self._queues.get(topic, []))
        return sum(len(queue) for queue in self._queues.values())
    
    def get_dead_letter_queue(self) -> List[Message]:
        """Get dead letter queue"""
        return self._dead_letter_queue.copy()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get message queue statistics"""
        return {
            'total_queues': len(self._queues),
            'total_messages': self.get_queue_size(),
            'processing': len(self._processing),
            'dead_letter_count': len(self._dead_letter_queue),
            'published': self._stats['published'],
            'consumed': self._stats['consumed'],
            'failed': self._stats['failed'],
            'dead_lettered': self._stats['dead_lettered'],
            'topics': list(self._queues.keys()),
            'subscribed_topics': list(self._consumers.keys()),
            'running': self._running
        }


# Global message queue service instance
_message_queue_service: Optional[MessageQueueService] = None


def get_message_queue_service() -> MessageQueueService:
    """Get or create message queue service instance"""
    global _message_queue_service
    if _message_queue_service is None:
        _message_queue_service = MessageQueueService()
    return _message_queue_service

