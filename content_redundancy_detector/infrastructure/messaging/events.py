"""
Event-Driven Architecture
Domain events and event publishing/subscribing
"""

import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Type
from enum import Enum

from .brokers import MessageBroker


@dataclass(frozen=True)
class DomainEvent:
    """Base domain event"""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: str = ""
    aggregate_id: str = ""
    timestamp: float = field(default_factory=time.time)
    version: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "aggregate_id": self.aggregate_id,
            "timestamp": self.timestamp,
            "version": self.version,
            "metadata": self.metadata,
            **self.__dict__
        }


# Domain Events
@dataclass(frozen=True)
class AnalysisCompletedEvent(DomainEvent):
    """Event emitted when analysis is completed"""
    content_hash: str
    redundancy_score: float
    analysis_result: Dict[str, Any]
    
    def __post_init__(self):
        if not self.event_type:
            object.__setattr__(self, 'event_type', 'analysis.completed')


@dataclass(frozen=True)
class SimilarityCompletedEvent(DomainEvent):
    """Event emitted when similarity check is completed"""
    text1_hash: str
    text2_hash: str
    similarity_score: float
    is_similar: bool
    
    def __post_init__(self):
        if not self.event_type:
            object.__setattr__(self, 'event_type', 'similarity.completed')


@dataclass(frozen=True)
class BatchProcessingCompletedEvent(DomainEvent):
    """Event emitted when batch processing is completed"""
    batch_id: str
    total_jobs: int
    completed_jobs: int
    failed_jobs: int
    
    def __post_init__(self):
        object.__setattr__(self, 'event_type', 'batch.completed')


@dataclass(frozen=True)
class WebhookDeliveredEvent(DomainEvent):
    """Event emitted when webhook is delivered"""
    endpoint_id: str
    delivery_id: str
    status: str
    
    def __post_init__(self):
        object.__setattr__(self, 'event_type', 'webhook.delivered')


class EventPublisher:
    """Publishes domain events to message broker"""
    
    def __init__(self, broker: MessageBroker):
        self.broker = broker
        self._connected = False
    
    async def connect(self):
        """Connect to broker"""
        if not self._connected:
            await self.broker.connect()
            self._connected = True
    
    async def publish(self, event: DomainEvent, topic: Optional[str] = None) -> None:
        """
        Publish domain event
        
        Args:
            event: Domain event to publish
            topic: Optional topic override (defaults to event_type)
        """
        if not self._connected:
            await self.connect()
        
        topic_name = topic or event.event_type.replace('.', '_')
        
        await self.broker.publish(
            topic=topic_name,
            message=event.to_dict(),
            key=event.aggregate_id
        )
    
    async def publish_batch(self, events: list[DomainEvent]) -> None:
        """Publish multiple events"""
        for event in events:
            await self.publish(event)


class EventSubscriber:
    """Subscribes to domain events from message broker"""
    
    def __init__(self, broker: MessageBroker, handler_registry: Dict[str, callable]):
        self.broker = broker
        self.handlers = handler_registry
        self._connected = False
    
    async def connect(self):
        """Connect to broker"""
        if not self._connected:
            await self.broker.connect()
            self._connected = True
    
    async def subscribe(self, topic: str, consumer_group: Optional[str] = None) -> None:
        """
        Subscribe to topic and handle events
        
        Args:
            topic: Topic to subscribe to
            consumer_group: Optional consumer group for load balancing
        """
        if not self._connected:
            await self.connect()
        
        async def handle_message(message: Dict[str, Any]):
            event_type = message.get('event_type', '')
            
            if event_type in self.handlers:
                handler = self.handlers[event_type]
                await handler(message)
            else:
                logger.warning(f"No handler registered for event type: {event_type}")
        
        await self.broker.subscribe(topic, handle_message, consumer_group)
    
    async def subscribe_all(self, consumer_group: Optional[str] = None) -> None:
        """Subscribe to all registered event types"""
        for event_type in self.handlers.keys():
            topic = event_type.replace('.', '_')
            await self.subscribe(topic, consumer_group)


# Event Handler Registry
class EventHandlerRegistry:
    """Registry for event handlers"""
    
    def __init__(self):
        self._handlers: Dict[str, list[callable]] = {}
    
    def register(self, event_type: str, handler: callable):
        """Register event handler"""
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)
    
    def get_handlers(self, event_type: str) -> list[callable]:
        """Get handlers for event type"""
        return self._handlers.get(event_type, [])
    
    def get_all_handlers(self) -> Dict[str, list[callable]]:
        """Get all registered handlers"""
        return self._handlers.copy()

