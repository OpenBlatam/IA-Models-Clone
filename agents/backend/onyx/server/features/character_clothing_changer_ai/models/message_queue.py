"""
Message Queue for Flux2 Clothing Changer
========================================

Advanced message queue system.
"""

import time
import uuid
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum
from collections import deque
import logging

logger = logging.getLogger(__name__)


class MessagePriority(Enum):
    """Message priority."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4


@dataclass
class Message:
    """Message information."""
    message_id: str
    queue_name: str
    payload: Dict[str, Any]
    priority: MessagePriority = MessagePriority.NORMAL
    created_at: float = time.time()
    processed_at: Optional[float] = None
    attempts: int = 0
    max_attempts: int = 3


class MessageQueue:
    """Advanced message queue system."""
    
    def __init__(
        self,
        max_queue_size: int = 10000,
    ):
        """
        Initialize message queue.
        
        Args:
            max_queue_size: Maximum queue size
        """
        self.max_queue_size = max_queue_size
        self.queues: Dict[str, deque] = {}
        self.consumers: Dict[str, List[Callable]] = {}
        self.message_history: deque = deque(maxlen=10000)
    
    def create_queue(self, queue_name: str) -> None:
        """
        Create message queue.
        
        Args:
            queue_name: Queue name
        """
        if queue_name not in self.queues:
            self.queues[queue_name] = deque(maxlen=self.max_queue_size)
            logger.info(f"Created queue: {queue_name}")
    
    def publish(
        self,
        queue_name: str,
        payload: Dict[str, Any],
        priority: MessagePriority = MessagePriority.NORMAL,
    ) -> str:
        """
        Publish message to queue.
        
        Args:
            queue_name: Queue name
            payload: Message payload
            priority: Message priority
            
        Returns:
            Message ID
        """
        if queue_name not in self.queues:
            self.create_queue(queue_name)
        
        message_id = str(uuid.uuid4())
        
        message = Message(
            message_id=message_id,
            queue_name=queue_name,
            payload=payload,
            priority=priority,
        )
        
        # Insert based on priority
        queue = self.queues[queue_name]
        
        if priority == MessagePriority.URGENT:
            queue.appendleft(message)
        elif priority == MessagePriority.HIGH:
            # Insert after urgent messages
            urgent_count = sum(1 for m in queue if m.priority == MessagePriority.URGENT)
            queue.insert(urgent_count, message)
        else:
            queue.append(message)
        
        logger.debug(f"Published message {message_id} to {queue_name}")
        return message_id
    
    def subscribe(
        self,
        queue_name: str,
        handler: Callable[[Message], bool],
    ) -> None:
        """
        Subscribe to queue.
        
        Args:
            queue_name: Queue name
            handler: Message handler function
        """
        if queue_name not in self.consumers:
            self.consumers[queue_name] = []
        
        self.consumers[queue_name].append(handler)
        logger.info(f"Subscribed to queue: {queue_name}")
    
    def process_messages(
        self,
        queue_name: str,
        batch_size: int = 10,
    ) -> int:
        """
        Process messages from queue.
        
        Args:
            queue_name: Queue name
            batch_size: Batch size
            
        Returns:
            Number of messages processed
        """
        if queue_name not in self.queues:
            return 0
        
        if queue_name not in self.consumers:
            return 0
        
        queue = self.queues[queue_name]
        handlers = self.consumers[queue_name]
        processed = 0
        
        for _ in range(min(batch_size, len(queue))):
            if not queue:
                break
            
            message = queue.popleft()
            message.attempts += 1
            
            success = False
            for handler in handlers:
                try:
                    if handler(message):
                        success = True
                        break
                except Exception as e:
                    logger.error(f"Error processing message {message.message_id}: {e}")
            
            if success:
                message.processed_at = time.time()
                processed += 1
            elif message.attempts < message.max_attempts:
                # Re-queue for retry
                queue.append(message)
            else:
                logger.warning(f"Message {message.message_id} failed after {message.max_attempts} attempts")
            
            self.message_history.append(message)
        
        return processed
    
    def get_queue_size(self, queue_name: str) -> int:
        """Get queue size."""
        if queue_name not in self.queues:
            return 0
        return len(self.queues[queue_name])
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get queue statistics."""
        return {
            "total_queues": len(self.queues),
            "queue_sizes": {
                name: len(queue) for name, queue in self.queues.items()
            },
            "total_consumers": sum(len(consumers) for consumers in self.consumers.values()),
        }


