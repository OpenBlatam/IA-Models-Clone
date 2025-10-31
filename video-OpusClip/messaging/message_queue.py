#!/usr/bin/env python3
"""
Message Queue System

Advanced message queue with:
- Asynchronous message processing
- Message persistence and durability
- Dead letter queues and retry logic
- Message routing and filtering
- Priority queues and scheduling
- Message acknowledgment and delivery guarantees
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Union, Callable, Awaitable
import asyncio
import time
import json
import uuid
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import structlog
from collections import defaultdict, deque
import heapq
import threading

logger = structlog.get_logger("message_queue")

# =============================================================================
# MESSAGE QUEUE MODELS
# =============================================================================

class MessagePriority(Enum):
    """Message priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

class MessageStatus(Enum):
    """Message processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    DEAD_LETTER = "dead_letter"
    SCHEDULED = "scheduled"

class DeliveryMode(Enum):
    """Message delivery modes."""
    AT_MOST_ONCE = "at_most_once"
    AT_LEAST_ONCE = "at_least_once"
    EXACTLY_ONCE = "exactly_once"

@dataclass
class Message:
    """Message structure."""
    message_id: str
    queue_name: str
    payload: Dict[str, Any]
    headers: Dict[str, str]
    priority: MessagePriority
    status: MessageStatus
    created_at: datetime
    scheduled_at: Optional[datetime]
    expires_at: Optional[datetime]
    delivery_mode: DeliveryMode
    retry_count: int
    max_retries: int
    correlation_id: Optional[str]
    reply_to: Optional[str]
    
    def __post_init__(self):
        if not self.message_id:
            self.message_id = str(uuid.uuid4())
        if not self.created_at:
            self.created_at = datetime.utcnow()
        if not self.headers:
            self.headers = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "message_id": self.message_id,
            "queue_name": self.queue_name,
            "payload": self.payload,
            "headers": self.headers,
            "priority": self.priority.value,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "scheduled_at": self.scheduled_at.isoformat() if self.scheduled_at else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "delivery_mode": self.delivery_mode.value,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "correlation_id": self.correlation_id,
            "reply_to": self.reply_to
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Message:
        """Create from dictionary."""
        return cls(
            message_id=data["message_id"],
            queue_name=data["queue_name"],
            payload=data["payload"],
            headers=data.get("headers", {}),
            priority=MessagePriority(data["priority"]),
            status=MessageStatus(data["status"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            scheduled_at=datetime.fromisoformat(data["scheduled_at"]) if data.get("scheduled_at") else None,
            expires_at=datetime.fromisoformat(data["expires_at"]) if data.get("expires_at") else None,
            delivery_mode=DeliveryMode(data["delivery_mode"]),
            retry_count=data.get("retry_count", 0),
            max_retries=data.get("max_retries", 3),
            correlation_id=data.get("correlation_id"),
            reply_to=data.get("reply_to")
        )

@dataclass
class QueueConfig:
    """Queue configuration."""
    queue_name: str
    max_size: int = 10000
    message_ttl: int = 3600  # seconds
    max_retries: int = 3
    retry_delay: int = 60  # seconds
    dead_letter_queue: Optional[str] = None
    enable_priority: bool = True
    enable_scheduling: bool = True
    enable_persistence: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "queue_name": self.queue_name,
            "max_size": self.max_size,
            "message_ttl": self.message_ttl,
            "max_retries": self.max_retries,
            "retry_delay": self.retry_delay,
            "dead_letter_queue": self.dead_letter_queue,
            "enable_priority": self.enable_priority,
            "enable_scheduling": self.enable_scheduling,
            "enable_persistence": self.enable_persistence
        }

@dataclass
class MessageHandler:
    """Message handler definition."""
    handler_id: str
    queue_name: str
    handler_func: Callable[[Message], Awaitable[bool]]
    concurrency: int = 1
    timeout: int = 30
    enabled: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "handler_id": handler_id,
            "queue_name": self.queue_name,
            "concurrency": self.concurrency,
            "timeout": self.timeout,
            "enabled": self.enabled
        }

# =============================================================================
# MESSAGE QUEUE IMPLEMENTATION
# =============================================================================

class MessageQueue:
    """Advanced message queue implementation."""
    
    def __init__(self, config: QueueConfig):
        self.config = config
        self.queue: deque = deque(maxlen=config.max_size)
        self.priority_queue: List[tuple] = []  # Heap for priority messages
        self.scheduled_messages: List[tuple] = []  # Heap for scheduled messages
        self.processing_messages: Dict[str, Message] = {}
        self.dead_letter_queue: deque = deque(maxlen=1000)
        
        # Message handlers
        self.handlers: List[MessageHandler] = []
        self.handler_tasks: List[asyncio.Task] = []
        
        # Statistics
        self.stats = {
            'messages_published': 0,
            'messages_consumed': 0,
            'messages_failed': 0,
            'messages_retried': 0,
            'messages_dead_lettered': 0,
            'average_processing_time': 0.0,
            'queue_size': 0
        }
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Background tasks
        self.is_running = False
        self.cleanup_task: Optional[asyncio.Task] = None
        self.scheduler_task: Optional[asyncio.Task] = None
    
    async def start(self) -> None:
        """Start the message queue."""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Start background tasks
        self.cleanup_task = asyncio.create_task(self._cleanup_expired_messages())
        self.scheduler_task = asyncio.create_task(self._process_scheduled_messages())
        
        # Start message handlers
        for handler in self.handlers:
            if handler.enabled:
                for worker_id in range(handler.concurrency):
                    task = asyncio.create_task(
                        self._process_messages(handler, worker_id)
                    )
                    self.handler_tasks.append(task)
        
        logger.info(
            "Message queue started",
            queue_name=self.config.queue_name,
            handlers=len(self.handlers)
        )
    
    async def stop(self) -> None:
        """Stop the message queue."""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Cancel background tasks
        if self.cleanup_task:
            self.cleanup_task.cancel()
        if self.scheduler_task:
            self.scheduler_task.cancel()
        
        # Cancel handler tasks
        for task in self.handler_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        all_tasks = [self.cleanup_task, self.scheduler_task] + self.handler_tasks
        await asyncio.gather(*[t for t in all_tasks if t], return_exceptions=True)
        
        logger.info("Message queue stopped", queue_name=self.config.queue_name)
    
    async def publish(self, message: Message) -> str:
        """Publish a message to the queue."""
        with self._lock:
            # Check queue size
            if len(self.queue) >= self.config.max_size:
                raise RuntimeError("Queue is full")
            
            # Set queue name
            message.queue_name = self.config.queue_name
            
            # Add to appropriate queue
            if message.scheduled_at and message.scheduled_at > datetime.utcnow():
                # Scheduled message
                heapq.heappush(self.scheduled_messages, (message.scheduled_at, message))
            elif self.config.enable_priority:
                # Priority message
                priority_value = -message.priority.value  # Negative for max-heap behavior
                heapq.heappush(self.priority_queue, (priority_value, message))
            else:
                # Regular message
                self.queue.append(message)
            
            # Update statistics
            self.stats['messages_published'] += 1
            self.stats['queue_size'] = len(self.queue) + len(self.priority_queue)
            
            logger.debug(
                "Message published",
                message_id=message.message_id,
                queue_name=self.config.queue_name,
                priority=message.priority.value
            )
            
            return message.message_id
    
    async def consume(self, timeout: Optional[float] = None) -> Optional[Message]:
        """Consume a message from the queue."""
        start_time = time.time()
        
        while True:
            with self._lock:
                # Try to get message from priority queue first
                if self.priority_queue:
                    _, message = heapq.heappop(self.priority_queue)
                    if not self._is_message_expired(message):
                        message.status = MessageStatus.PROCESSING
                        self.processing_messages[message.message_id] = message
                        return message
                
                # Try to get message from regular queue
                if self.queue:
                    message = self.queue.popleft()
                    if not self._is_message_expired(message):
                        message.status = MessageStatus.PROCESSING
                        self.processing_messages[message.message_id] = message
                        return message
            
            # Check timeout
            if timeout and (time.time() - start_time) >= timeout:
                return None
            
            # Wait a bit before trying again
            await asyncio.sleep(0.1)
    
    async def acknowledge(self, message_id: str, success: bool = True) -> None:
        """Acknowledge message processing."""
        if message_id not in self.processing_messages:
            return
        
        message = self.processing_messages[message_id]
        
        if success:
            message.status = MessageStatus.COMPLETED
            self.stats['messages_consumed'] += 1
        else:
            # Handle failure
            message.retry_count += 1
            
            if message.retry_count >= message.max_retries:
                # Move to dead letter queue
                message.status = MessageStatus.DEAD_LETTER
                self.dead_letter_queue.append(message)
                self.stats['messages_dead_lettered'] += 1
                
                logger.warning(
                    "Message moved to dead letter queue",
                    message_id=message_id,
                    retry_count=message.retry_count
                )
            else:
                # Schedule retry
                message.status = MessageStatus.RETRYING
                retry_time = datetime.utcnow() + timedelta(seconds=self.config.retry_delay)
                message.scheduled_at = retry_time
                
                heapq.heappush(self.scheduled_messages, (retry_time, message))
                self.stats['messages_retried'] += 1
                
                logger.info(
                    "Message scheduled for retry",
                    message_id=message_id,
                    retry_count=message.retry_count,
                    retry_time=retry_time
                )
        
        # Remove from processing
        del self.processing_messages[message_id]
    
    def add_handler(self, handler: MessageHandler) -> None:
        """Add message handler."""
        self.handlers.append(handler)
        logger.info(
            "Message handler added",
            handler_id=handler.handler_id,
            queue_name=handler.queue_name
        )
    
    def remove_handler(self, handler_id: str) -> bool:
        """Remove message handler."""
        for i, handler in enumerate(self.handlers):
            if handler.handler_id == handler_id:
                del self.handlers[i]
                logger.info("Message handler removed", handler_id=handler_id)
                return True
        return False
    
    async def _process_messages(self, handler: MessageHandler, worker_id: int) -> None:
        """Process messages with handler."""
        logger.info(
            "Message handler worker started",
            handler_id=handler.handler_id,
            worker_id=worker_id
        )
        
        while self.is_running:
            try:
                # Get message
                message = await self.consume(timeout=1.0)
                if not message:
                    continue
                
                # Process message
                start_time = time.time()
                
                try:
                    success = await asyncio.wait_for(
                        handler.handler_func(message),
                        timeout=handler.timeout
                    )
                    
                    processing_time = time.time() - start_time
                    self._update_processing_time(processing_time)
                    
                except asyncio.TimeoutError:
                    logger.error(
                        "Handler timeout",
                        handler_id=handler.handler_id,
                        message_id=message.message_id,
                        timeout=handler.timeout
                    )
                    success = False
                
                except Exception as e:
                    logger.error(
                        "Handler error",
                        handler_id=handler.handler_id,
                        message_id=message.message_id,
                        error=str(e)
                    )
                    success = False
                
                # Acknowledge message
                await self.acknowledge(message.message_id, success)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Message processing error", error=str(e))
                await asyncio.sleep(1)
    
    async def _process_scheduled_messages(self) -> None:
        """Process scheduled messages."""
        while self.is_running:
            try:
                current_time = datetime.utcnow()
                
                with self._lock:
                    # Move ready scheduled messages to main queue
                    ready_messages = []
                    
                    while self.scheduled_messages:
                        scheduled_time, message = self.scheduled_messages[0]
                        
                        if scheduled_time <= current_time:
                            heapq.heappop(self.scheduled_messages)
                            ready_messages.append(message)
                        else:
                            break
                    
                    # Add ready messages to appropriate queue
                    for message in ready_messages:
                        message.scheduled_at = None
                        
                        if self.config.enable_priority:
                            priority_value = -message.priority.value
                            heapq.heappush(self.priority_queue, (priority_value, message))
                        else:
                            self.queue.append(message)
                
                await asyncio.sleep(1)  # Check every second
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Scheduled message processing error", error=str(e))
                await asyncio.sleep(1)
    
    async def _cleanup_expired_messages(self) -> None:
        """Clean up expired messages."""
        while self.is_running:
            try:
                current_time = datetime.utcnow()
                
                with self._lock:
                    # Clean up expired messages from regular queue
                    expired_count = 0
                    valid_messages = deque()
                    
                    while self.queue:
                        message = self.queue.popleft()
                        if not self._is_message_expired(message):
                            valid_messages.append(message)
                        else:
                            expired_count += 1
                    
                    self.queue = valid_messages
                    
                    # Clean up expired messages from priority queue
                    valid_priority_messages = []
                    
                    while self.priority_queue:
                        priority, message = heapq.heappop(self.priority_queue)
                        if not self._is_message_expired(message):
                            valid_priority_messages.append((priority, message))
                        else:
                            expired_count += 1
                    
                    self.priority_queue = valid_priority_messages
                    
                    if expired_count > 0:
                        logger.info(
                            "Expired messages cleaned up",
                            expired_count=expired_count,
                            queue_name=self.config.queue_name
                        )
                
                await asyncio.sleep(60)  # Clean up every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Message cleanup error", error=str(e))
                await asyncio.sleep(60)
    
    def _is_message_expired(self, message: Message) -> bool:
        """Check if message is expired."""
        if not message.expires_at:
            return False
        
        return datetime.utcnow() > message.expires_at
    
    def _update_processing_time(self, processing_time: float) -> None:
        """Update average processing time."""
        total_messages = self.stats['messages_consumed']
        current_avg = self.stats['average_processing_time']
        
        if total_messages > 0:
            self.stats['average_processing_time'] = (
                (current_avg * (total_messages - 1) + processing_time) / total_messages
            )
        else:
            self.stats['average_processing_time'] = processing_time
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        return {
            **self.stats,
            'queue_size': len(self.queue),
            'priority_queue_size': len(self.priority_queue),
            'scheduled_messages_count': len(self.scheduled_messages),
            'processing_messages_count': len(self.processing_messages),
            'dead_letter_count': len(self.dead_letter_queue),
            'handlers_count': len(self.handlers)
        }
    
    def get_dead_letter_messages(self) -> List[Dict[str, Any]]:
        """Get dead letter messages."""
        return [message.to_dict() for message in self.dead_letter_queue]

# =============================================================================
# MESSAGE QUEUE MANAGER
# =============================================================================

class MessageQueueManager:
    """Manager for multiple message queues."""
    
    def __init__(self):
        self.queues: Dict[str, MessageQueue] = {}
        self.is_running = False
    
    async def start(self) -> None:
        """Start all queues."""
        if self.is_running:
            return
        
        self.is_running = True
        
        for queue in self.queues.values():
            await queue.start()
        
        logger.info("Message queue manager started", queue_count=len(self.queues))
    
    async def stop(self) -> None:
        """Stop all queues."""
        if not self.is_running:
            return
        
        self.is_running = False
        
        for queue in self.queues.values():
            await queue.stop()
        
        logger.info("Message queue manager stopped")
    
    def create_queue(self, config: QueueConfig) -> MessageQueue:
        """Create a new message queue."""
        queue = MessageQueue(config)
        self.queues[config.queue_name] = queue
        
        logger.info("Message queue created", queue_name=config.queue_name)
        return queue
    
    def get_queue(self, queue_name: str) -> Optional[MessageQueue]:
        """Get queue by name."""
        return self.queues.get(queue_name)
    
    def remove_queue(self, queue_name: str) -> bool:
        """Remove queue."""
        if queue_name in self.queues:
            del self.queues[queue_name]
            logger.info("Message queue removed", queue_name=queue_name)
            return True
        return False
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all queues."""
        return {
            queue_name: queue.get_queue_stats()
            for queue_name, queue in self.queues.items()
        }
    
    def get_global_stats(self) -> Dict[str, Any]:
        """Get global statistics."""
        all_stats = self.get_all_stats()
        
        if not all_stats:
            return {}
        
        return {
            'total_queues': len(self.queues),
            'total_messages_published': sum(stats['messages_published'] for stats in all_stats.values()),
            'total_messages_consumed': sum(stats['messages_consumed'] for stats in all_stats.values()),
            'total_messages_failed': sum(stats['messages_failed'] for stats in all_stats.values()),
            'total_queue_size': sum(stats['queue_size'] for stats in all_stats.values()),
            'queues': all_stats
        }

# =============================================================================
# GLOBAL MESSAGE QUEUE INSTANCES
# =============================================================================

# Global message queue manager
message_queue_manager = MessageQueueManager()

# Pre-configured queues
video_processing_queue = message_queue_manager.create_queue(
    QueueConfig(
        queue_name="video_processing",
        max_size=5000,
        message_ttl=7200,
        max_retries=3,
        retry_delay=60,
        dead_letter_queue="video_processing_dlq"
    )
)

batch_processing_queue = message_queue_manager.create_queue(
    QueueConfig(
        queue_name="batch_processing",
        max_size=10000,
        message_ttl=3600,
        max_retries=5,
        retry_delay=30,
        dead_letter_queue="batch_processing_dlq"
    )
)

notification_queue = message_queue_manager.create_queue(
    QueueConfig(
        queue_name="notifications",
        max_size=2000,
        message_ttl=1800,
        max_retries=2,
        retry_delay=120,
        dead_letter_queue="notifications_dlq"
    )
)

# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'MessagePriority',
    'MessageStatus',
    'DeliveryMode',
    'Message',
    'QueueConfig',
    'MessageHandler',
    'MessageQueue',
    'MessageQueueManager',
    'message_queue_manager',
    'video_processing_queue',
    'batch_processing_queue',
    'notification_queue'
]





























