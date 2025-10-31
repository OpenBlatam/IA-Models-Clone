"""
PDF Variantes - Event System
Pub/Sub event system for decoupled communication
"""

from typing import Dict, List, Callable, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import asyncio
import logging
from enum import Enum


class EventType(str, Enum):
    """Event types in the system"""
    # PDF Events
    PDF_UPLOADED = "pdf.uploaded"
    PDF_PROCESSED = "pdf.processed"
    PDF_DELETED = "pdf.deleted"
    
    # Variant Events
    VARIANT_GENERATED = "variant.generated"
    VARIANT_GENERATION_STARTED = "variant.generation_started"
    VARIANT_GENERATION_STOPPED = "variant.generation_stopped"
    
    # Topic Events
    TOPICS_EXTRACTED = "topics.extracted"
    
    # Brainstorm Events
    BRAINSTORM_GENERATED = "brainstorm.generated"
    
    # Collaboration Events
    COLLABORATOR_JOINED = "collaboration.joined"
    COLLABORATOR_LEFT = "collaboration.left"
    
    # Export Events
    EXPORT_COMPLETED = "export.completed"
    
    # System Events
    SERVICE_STARTED = "system.service_started"
    SERVICE_STOPPED = "system.service_stopped"
    ERROR_OCCURRED = "system.error"


@dataclass
class Event:
    """Event data structure"""
    type: EventType
    payload: Dict[str, Any]
    timestamp: datetime
    source: str
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class EventBus:
    """Central event bus for pub/sub"""
    
    def __init__(self):
        self._subscribers: Dict[EventType, List[Callable]] = {}
        self.logger = logging.getLogger(__name__)
        self._event_history: List[Event] = []
        self._max_history = 1000
    
    def subscribe(self, event_type: EventType, handler: Callable):
        """Subscribe to an event type"""
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(handler)
        self.logger.debug(f"Subscribed {handler} to {event_type}")
    
    def unsubscribe(self, event_type: EventType, handler: Callable):
        """Unsubscribe from an event type"""
        if event_type in self._subscribers:
            try:
                self._subscribers[event_type].remove(handler)
            except ValueError:
                pass
    
    async def publish(self, event: Event):
        """Publish an event to all subscribers"""
        handlers = self._subscribers.get(event.type, [])
        
        # Store in history
        self._event_history.append(event)
        if len(self._event_history) > self._max_history:
            self._event_history.pop(0)
        
        # Call all handlers
        tasks = []
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    tasks.append(handler(event))
                else:
                    handler(event)
            except Exception as e:
                self.logger.error(f"Error in event handler for {event.type}: {e}")
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def emit(
        self,
        event_type: EventType,
        payload: Dict[str, Any],
        source: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Convenience method to emit an event"""
        event = Event(
            type=event_type,
            payload=payload,
            timestamp=datetime.utcnow(),
            source=source,
            metadata=metadata
        )
        await self.publish(event)
    
    def get_history(self, event_type: Optional[EventType] = None, limit: int = 100) -> List[Event]:
        """Get event history"""
        history = self._event_history
        if event_type:
            history = [e for e in history if e.type == event_type]
        return history[-limit:]
    
    def clear_history(self):
        """Clear event history"""
        self._event_history.clear()


# Global event bus instance
_event_bus: Optional[EventBus] = None


def get_event_bus() -> EventBus:
    """Get global event bus instance"""
    global _event_bus
    if _event_bus is None:
        _event_bus = EventBus()
    return _event_bus


def reset_event_bus():
    """Reset global event bus (for testing)"""
    global _event_bus
    _event_bus = None






