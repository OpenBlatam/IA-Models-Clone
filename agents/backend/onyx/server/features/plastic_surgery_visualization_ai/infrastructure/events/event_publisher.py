"""Event publisher implementation."""

from typing import Dict, List, Callable, Any
import asyncio

from core.interfaces import IEventPublisher, IEventSubscriber
from domain.events import DomainEvent
from utils.logger import get_logger

logger = get_logger(__name__)


class SimpleEventPublisher(IEventPublisher, IEventSubscriber):
    """Simple in-memory event publisher/subscriber."""
    
    def __init__(self):
        self._subscribers: Dict[str, List[Callable]] = {}
    
    async def publish(self, event: DomainEvent) -> None:
        """
        Publish a domain event.
        
        Args:
            event: Domain event to publish
        """
        event_type = event.event_type
        subscribers = self._subscribers.get(event_type, [])
        
        logger.debug(f"Publishing event {event_type} to {len(subscribers)} subscribers")
        
        # Execute all subscribers asynchronously
        tasks = [handler(event) for handler in subscribers]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def subscribe(self, event_type: str, handler: Callable) -> None:
        """
        Subscribe to an event type.
        
        Args:
            event_type: Type of event to subscribe to
            handler: Async function to handle the event
        """
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        
        self._subscribers[event_type].append(handler)
        logger.debug(f"Subscribed handler to event type: {event_type}")
    
    def subscribe_sync(self, event_type: str, handler: Callable) -> None:
        """
        Subscribe to an event type (synchronous version).
        
        Args:
            event_type: Type of event to subscribe to
            handler: Async function to handle the event
        """
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        
        self._subscribers[event_type].append(handler)
        logger.debug(f"Subscribed handler to event type: {event_type}")


class NullEventPublisher(IEventPublisher):
    """Null object pattern for event publishing (no-op)."""
    
    async def publish(self, event: DomainEvent) -> None:
        """No-op publish."""
        pass

