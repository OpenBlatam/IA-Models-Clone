"""
Shared Events
Event system shared across modules
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Type, List
from datetime import datetime
from uuid import uuid4
import asyncio


class BaseEvent(ABC):
    """Base event class"""
    
    def __init__(self, source: str, metadata: Dict[str, Any] = None):
        self.event_id = str(uuid4())
        self.occurred_at = datetime.utcnow()
        self.source = source
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "event_id": self.event_id,
            "event_type": self.__class__.__name__,
            "source": self.source,
            "occurred_at": self.occurred_at.isoformat(),
            "metadata": self.metadata
        }


class EventHandler(ABC):
    """Base event handler"""
    
    @abstractmethod
    async def handle(self, event: BaseEvent) -> None:
        """Handle event"""
        pass
    
    @property
    @abstractmethod
    def handles(self) -> List[Type[BaseEvent]]:
        """List of event types this handler handles"""
        pass


class EventBus(ABC):
    """Event bus interface"""
    
    @abstractmethod
    async def publish(self, event: BaseEvent) -> None:
        """Publish event"""
        pass
    
    @abstractmethod
    async def subscribe(
        self,
        event_type: Type[BaseEvent],
        handler: EventHandler
    ) -> None:
        """Subscribe handler to event type"""
        pass


class InMemoryEventBus(EventBus):
    """In-memory event bus implementation"""
    
    def __init__(self):
        self._handlers: Dict[Type[BaseEvent], List[EventHandler]] = {}
    
    async def publish(self, event: BaseEvent) -> None:
        """Publish event to all subscribers"""
        event_type = type(event)
        handlers = self._handlers.get(event_type, [])
        
        if handlers:
            tasks = [handler.handle(event) for handler in handlers]
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def subscribe(
        self,
        event_type: Type[BaseEvent],
        handler: EventHandler
    ) -> None:
        """Subscribe handler to event type"""
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        
        if handler not in self._handlers[event_type]:
            self._handlers[event_type].append(handler)






