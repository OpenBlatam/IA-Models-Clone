"""
Dead Letter Queue
=================

Advanced dead letter queue for failed messages.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from collections import deque
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class DeadLetterMessage:
    """Dead letter message."""
    message_id: str
    original_message: Any
    error: str
    error_type: str
    timestamp: datetime
    retry_count: int
    metadata: Dict[str, Any]

class DeadLetterQueue:
    """Advanced dead letter queue."""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.queue: deque = deque(maxlen=max_size)
        self.stats = {
            "total_messages": 0,
            "processed_messages": 0,
            "failed_messages": 0
        }
    
    def add(
        self,
        message_id: str,
        original_message: Any,
        error: str,
        error_type: str,
        retry_count: int = 0,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Add message to dead letter queue."""
        dlq_message = DeadLetterMessage(
            message_id=message_id,
            original_message=original_message,
            error=error,
            error_type=error_type,
            timestamp=datetime.now(),
            retry_count=retry_count,
            metadata=metadata or {}
        )
        
        self.queue.append(dlq_message)
        self.stats["failed_messages"] += 1
        logger.warning(f"Message {message_id} added to DLQ: {error}")
    
    def get_all(self, limit: Optional[int] = None) -> List[DeadLetterMessage]:
        """Get all messages from DLQ."""
        messages = list(self.queue)
        if limit:
            return messages[-limit:]
        return messages
    
    def get_by_error_type(self, error_type: str) -> List[DeadLetterMessage]:
        """Get messages by error type."""
        return [
            msg for msg in self.queue
            if msg.error_type == error_type
        ]
    
    def remove(self, message_id: str) -> bool:
        """Remove message from DLQ."""
        for i, msg in enumerate(self.queue):
            if msg.message_id == message_id:
                del self.queue[i]
                return True
        return False
    
    def clear(self):
        """Clear all messages."""
        self.queue.clear()
        logger.info("Dead letter queue cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get DLQ statistics."""
        error_types = {}
        for msg in self.queue:
            error_types[msg.error_type] = error_types.get(msg.error_type, 0) + 1
        
        return {
            "queue_size": len(self.queue),
            "max_size": self.max_size,
            "total_messages": self.stats["total_messages"],
            "processed_messages": self.stats["processed_messages"],
            "failed_messages": self.stats["failed_messages"],
            "error_types": error_types,
            "oldest_message": self.queue[0].timestamp.isoformat() if self.queue else None,
            "newest_message": self.queue[-1].timestamp.isoformat() if self.queue else None
        }
    
    async def retry_message(
        self,
        message_id: str,
        processor: Callable,
        max_retries: int = 3
    ) -> bool:
        """Retry processing a message from DLQ."""
        message = None
        for msg in self.queue:
            if msg.message_id == message_id:
                message = msg
                break
        
        if not message:
            return False
        
        if message.retry_count >= max_retries:
            logger.warning(f"Message {message_id} exceeded max retries")
            return False
        
        try:
            # Process message
            if asyncio.iscoroutinefunction(processor):
                await processor(message.original_message)
            else:
                processor(message.original_message)
            
            # Remove from DLQ on success
            self.remove(message_id)
            self.stats["processed_messages"] += 1
            logger.info(f"Message {message_id} successfully reprocessed")
            return True
            
        except Exception as e:
            # Increment retry count
            message.retry_count += 1
            message.error = str(e)
            logger.error(f"Retry failed for message {message_id}: {e}")
            return False

# Global instance
dead_letter_queue = DeadLetterQueue()
































