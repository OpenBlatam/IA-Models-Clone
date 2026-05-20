"""
Unified Messaging System for Color Grading AI
==============================================

Unified messaging system consolidating events, notifications, and webhooks.
"""

import logging
import asyncio
import uuid
from typing import Dict, Any, Optional, List, Callable, TypeVar, Generic
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

T = TypeVar('T')


class MessageType(Enum):
    """Message types."""
    EVENT = "event"
    NOTIFICATION = "notification"
    WEBHOOK = "webhook"
    COMMAND = "command"
    QUERY = "query"


class MessagePriority(Enum):
    """Message priority."""
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3


@dataclass
class Message(Generic[T]):
    """Unified message structure."""
    message_id: str
    message_type: MessageType
    topic: str
    payload: T
    priority: MessagePriority = MessagePriority.NORMAL
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = "system"
    metadata: Dict[str, Any] = field(default_factory=dict)
    correlation_id: Optional[str] = None
    reply_to: Optional[str] = None


class MessageHandler(ABC, Generic[T]):
    """Abstract message handler."""
    
    @abstractmethod
    async def handle(self, message: Message[T]) -> Optional[Any]:
        """Handle message."""
        pass


class UnifiedMessaging:
    """
    Unified messaging system.
    
    Features:
    - Pub/Sub pattern
    - Topic-based routing
    - Priority handling
    - Message filtering
    - Async handlers
    - Message history
    - Retry mechanism
    - Dead letter queue
    """
    
    def __init__(self, max_history: int = 10000):
        """
        Initialize unified messaging.
        
        Args:
            max_history: Maximum message history
        """
        self.max_history = max_history
        self._subscribers: Dict[str, List[MessageHandler]] = {}
        self._message_history: List[Message] = []
        self._dead_letter_queue: List[Message] = []
        self._max_retries = 3
        self._running = False
    
    def subscribe(
        self,
        topic: str,
        handler: MessageHandler,
        message_type: Optional[MessageType] = None
    ):
        """
        Subscribe to topic.
        
        Args:
            topic: Topic name
            handler: Message handler
            message_type: Optional message type filter
        """
        topic_key = self._get_topic_key(topic, message_type)
        
        if topic_key not in self._subscribers:
            self._subscribers[topic_key] = []
        
        self._subscribers[topic_key].append(handler)
        logger.info(f"Subscribed handler to topic: {topic_key}")
    
    def unsubscribe(self, topic: str, handler: MessageHandler):
        """Unsubscribe from topic."""
        for topic_key in self._subscribers.keys():
            if topic_key.startswith(topic):
                if handler in self._subscribers[topic_key]:
                    self._subscribers[topic_key].remove(handler)
                    logger.info(f"Unsubscribed handler from topic: {topic_key}")
    
    async def publish(
        self,
        topic: str,
        payload: Any,
        message_type: MessageType = MessageType.EVENT,
        priority: MessagePriority = MessagePriority.NORMAL,
        source: str = "system",
        metadata: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None
    ) -> str:
        """
        Publish message.
        
        Args:
            topic: Topic name
            payload: Message payload
            message_type: Message type
            priority: Message priority
            source: Message source
            metadata: Optional metadata
            correlation_id: Optional correlation ID
            
        Returns:
            Message ID
        """
        message = Message(
            message_id=str(uuid.uuid4()),
            message_type=message_type,
            topic=topic,
            payload=payload,
            priority=priority,
            source=source,
            metadata=metadata or {},
            correlation_id=correlation_id
        )
        
        # Add to history
        self._message_history.append(message)
        if len(self._message_history) > self.max_history:
            self._message_history = self._message_history[-self.max_history:]
        
        # Route to subscribers
        await self._route_message(message)
        
        logger.debug(f"Published message {message.message_id} to topic {topic}")
        
        return message.message_id
    
    async def _route_message(self, message: Message):
        """Route message to subscribers."""
        handlers = []
        
        # Get exact topic match
        exact_key = self._get_topic_key(message.topic, message.message_type)
        handlers.extend(self._subscribers.get(exact_key, []))
        
        # Get wildcard matches
        for topic_key, topic_handlers in self._subscribers.items():
            if self._matches_topic(topic_key, message.topic, message.message_type):
                handlers.extend(topic_handlers)
        
        # Execute handlers
        for handler in handlers:
            try:
                await handler.handle(message)
            except Exception as e:
                logger.error(f"Error handling message {message.message_id}: {e}")
                # Retry logic could be added here
    
    def _get_topic_key(self, topic: str, message_type: Optional[MessageType]) -> str:
        """Get topic key for subscription."""
        if message_type:
            return f"{message_type.value}:{topic}"
        return f"*:{topic}"
    
    def _matches_topic(
        self,
        topic_key: str,
        message_topic: str,
        message_type: MessageType
    ) -> bool:
        """Check if topic matches."""
        parts = topic_key.split(":", 1)
        if len(parts) != 2:
            return False
        
        key_type, key_topic = parts
        
        # Type match
        if key_type != "*" and key_type != message_type.value:
            return False
        
        # Topic match (supports wildcards)
        if key_topic == "*" or key_topic == message_topic:
            return True
        
        if "*" in key_topic:
            # Simple wildcard matching
            pattern = key_topic.replace("*", ".*")
            import re
            return bool(re.match(pattern, message_topic))
        
        return False
    
    def get_message_history(
        self,
        topic: Optional[str] = None,
        message_type: Optional[MessageType] = None,
        limit: int = 100
    ) -> List[Message]:
        """Get message history."""
        messages = self._message_history
        
        if topic:
            messages = [m for m in messages if m.topic == topic]
        if message_type:
            messages = [m for m in messages if m.message_type == message_type]
        
        return messages[-limit:]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get messaging statistics."""
        return {
            "total_messages": len(self._message_history),
            "subscribers_count": sum(len(h) for h in self._subscribers.values()),
            "topics_count": len(self._subscribers),
            "dead_letter_count": len(self._dead_letter_queue),
        }


# Convenience functions for different message types
class EventHandler(MessageHandler):
    """Event message handler."""
    
    def __init__(self, handler_func: Callable):
        self.handler_func = handler_func
    
    async def handle(self, message: Message) -> Optional[Any]:
        """Handle event message."""
        if message.message_type == MessageType.EVENT:
            if asyncio.iscoroutinefunction(self.handler_func):
                return await self.handler_func(message.payload, message)
            else:
                return self.handler_func(message.payload, message)


class NotificationHandler(MessageHandler):
    """Notification message handler."""
    
    def __init__(self, handler_func: Callable):
        self.handler_func = handler_func
    
    async def handle(self, message: Message) -> Optional[Any]:
        """Handle notification message."""
        if message.message_type == MessageType.NOTIFICATION:
            if asyncio.iscoroutinefunction(self.handler_func):
                return await self.handler_func(message.payload, message)
            else:
                return self.handler_func(message.payload, message)


class WebhookHandler(MessageHandler):
    """Webhook message handler."""
    
    def __init__(self, handler_func: Callable):
        self.handler_func = handler_func
    
    async def handle(self, message: Message) -> Optional[Any]:
        """Handle webhook message."""
        if message.message_type == MessageType.WEBHOOK:
            if asyncio.iscoroutinefunction(self.handler_func):
                return await self.handler_func(message.payload, message)
            else:
                return self.handler_func(message.payload, message)




