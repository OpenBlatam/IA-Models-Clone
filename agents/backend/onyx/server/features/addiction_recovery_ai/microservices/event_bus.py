"""
Event Bus
Event-driven communication for microservices
"""

import logging
import json
from typing import Dict, Any, List, Callable, Optional
from datetime import datetime
from enum import Enum
from dataclasses import dataclass

from core.interfaces import IEventPublisher, IEventHandler
from aws.aws_services import SNSService, SQSService
from microservices.service_discovery import ServiceRegistry, get_service_registry

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Event type enumeration"""
    USER_CREATED = "user.created"
    USER_UPDATED = "user.updated"
    MILESTONE_ACHIEVED = "milestone.achieved"
    PROGRESS_UPDATED = "progress.updated"
    RELAPSE_DETECTED = "relapse.detected"
    NOTIFICATION_SENT = "notification.sent"


@dataclass
class Event:
    """Event data structure"""
    event_type: str
    source: str
    data: Dict[str, Any]
    timestamp: datetime = None
    event_id: str = None
    correlation_id: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
        if self.event_id is None:
            import uuid
            self.event_id = str(uuid.uuid4())


class EventBus:
    """
    Event bus for event-driven architecture
    
    Features:
    - Event publishing
    - Event subscription
    - Event routing
    - Event filtering
    - Dead letter queue
    """
    
    def __init__(self):
        self._subscribers: Dict[str, List[IEventHandler]] = {}
        self._publishers: Dict[str, IEventPublisher] = {}
        self.registry = get_service_registry()
        self.sns = SNSService() if self._has_sns() else None
        self.sqs = SQSService() if self._has_sqs() else None
    
    def _has_sns(self) -> bool:
        """Check if SNS is available"""
        try:
            from config.aws_settings import get_aws_settings
            return bool(get_aws_settings().sns_topic_arn)
        except:
            return False
    
    def _has_sqs(self) -> bool:
        """Check if SQS is available"""
        try:
            from config.aws_settings import get_aws_settings
            return bool(get_aws_settings().sqs_queue_url)
        except:
            return False
    
    def subscribe(self, event_type: str, handler: IEventHandler) -> None:
        """Subscribe to event type"""
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        
        self._subscribers[event_type].append(handler)
        logger.info(f"Subscribed handler to event: {event_type}")
    
    def unsubscribe(self, event_type: str, handler: IEventHandler) -> bool:
        """Unsubscribe from event type"""
        if event_type not in self._subscribers:
            return False
        
        try:
            self._subscribers[event_type].remove(handler)
            return True
        except ValueError:
            return False
    
    async def publish(self, event: Event) -> str:
        """
        Publish event
        
        Args:
            event: Event to publish
            
        Returns:
            Event ID
        """
        # Publish to SNS if available
        if self.sns:
            try:
                await self._publish_to_sns(event)
            except Exception as e:
                logger.error(f"Failed to publish to SNS: {str(e)}")
        
        # Publish to local subscribers
        await self._publish_local(event)
        
        return event.event_id
    
    async def _publish_to_sns(self, event: Event) -> None:
        """Publish event to SNS"""
        if not self.sns:
            return
        
        message = {
            "event_type": event.event_type,
            "source": event.source,
            "data": event.data,
            "timestamp": event.timestamp.isoformat(),
            "event_id": event.event_id,
            "correlation_id": event.correlation_id
        }
        
        self.sns.publish(
            message=json.dumps(message),
            subject=f"Event: {event.event_type}"
        )
    
    async def _publish_local(self, event: Event) -> None:
        """Publish event to local subscribers"""
        handlers = self._subscribers.get(event.event_type, [])
        
        for handler in handlers:
            try:
                if hasattr(handler, "handle"):
                    await handler.handle(event.data)
                elif callable(handler):
                    await handler(event.data)
            except Exception as e:
                logger.error(f"Error handling event {event.event_type}: {str(e)}")
    
    async def publish_event(
        self,
        event_type: str,
        source: str,
        data: Dict[str, Any],
        correlation_id: Optional[str] = None
    ) -> str:
        """Publish event (convenience method)"""
        event = Event(
            event_type=event_type,
            source=source,
            data=data,
            correlation_id=correlation_id
        )
        return await self.publish(event)
    
    def list_subscribers(self, event_type: str) -> List[str]:
        """List subscribers for event type"""
        handlers = self._subscribers.get(event_type, [])
        return [type(h).__name__ for h in handlers]


# Global event bus instance
_event_bus: Optional[EventBus] = None


def get_event_bus() -> EventBus:
    """Get global event bus"""
    global _event_bus
    if _event_bus is None:
        _event_bus = EventBus()
    return _event_bus

