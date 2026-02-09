"""
Dead Letter Queue
=================

Dead letter queue management.
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class DeadLetterMessage:
    """Dead letter message."""
    original_message_id: str
    original_queue: str
    payload: Any
    failure_reason: str
    failed_at: datetime
    attempts: int
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class DeadLetterQueue:
    """Dead letter queue manager."""
    
    def __init__(self):
        self._dlq: List[DeadLetterMessage] = []
        self._max_size: int = 10000
    
    def add_message(
        self,
        original_message_id: str,
        original_queue: str,
        payload: Any,
        failure_reason: str,
        attempts: int
    ):
        """Add message to DLQ."""
        dlq_message = DeadLetterMessage(
            original_message_id=original_message_id,
            original_queue=original_queue,
            payload=payload,
            failure_reason=failure_reason,
            failed_at=datetime.now(),
            attempts=attempts
        )
        
        self._dlq.append(dlq_message)
        
        # Limit size
        if len(self._dlq) > self._max_size:
            self._dlq.pop(0)
        
        logger.warning(f"Added message to DLQ: {original_message_id} - {failure_reason}")
    
    def get_messages(
        self,
        queue_name: Optional[str] = None,
        limit: int = 100
    ) -> List[DeadLetterMessage]:
        """Get messages from DLQ."""
        messages = self._dlq
        
        if queue_name:
            messages = [m for m in messages if m.original_queue == queue_name]
        
        return messages[-limit:]
    
    def reprocess_message(self, message_id: str) -> bool:
        """Reprocess message from DLQ."""
        # In production, implement actual reprocessing
        # This is a placeholder
        for message in self._dlq:
            if message.original_message_id == message_id:
                logger.info(f"Reprocessing message: {message_id}")
                return True
        return False
    
    def get_dlq_stats(self) -> Dict[str, Any]:
        """Get DLQ statistics."""
        by_queue = {}
        for message in self._dlq:
            queue = message.original_queue
            if queue not in by_queue:
                by_queue[queue] = 0
            by_queue[queue] += 1
        
        return {
            "total_messages": len(self._dlq),
            "by_queue": by_queue,
            "max_size": self._max_size
        }















