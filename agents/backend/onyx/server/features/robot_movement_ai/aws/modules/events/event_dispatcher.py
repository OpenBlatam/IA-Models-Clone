"""
Event Dispatcher
================

Advanced event dispatcher with routing and filtering.
"""

import logging
from typing import Dict, List, Optional, Callable
from aws.modules.events.event_bus import EventBus, Event

logger = logging.getLogger(__name__)


class EventDispatcher:
    """Event dispatcher with routing capabilities."""
    
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self._routes: Dict[str, List[str]] = {}
        self._filters: Dict[str, Callable[[Event], bool]] = {}
    
    def route(self, event_type: str, target_types: List[str]):
        """Route event to other event types."""
        self._routes[event_type] = target_types
        logger.info(f"Routed {event_type} to {target_types}")
    
    def filter(self, event_type: str, filter_func: Callable[[Event], bool]):
        """Add filter for event type."""
        self._filters[event_type] = filter_func
    
    async def dispatch(self, event: Event):
        """Dispatch event with routing and filtering."""
        # Apply filter
        if event.event_type in self._filters:
            if not self._filters[event.event_type](event):
                logger.debug(f"Event {event.event_type} filtered out")
                return
        
        # Publish original event
        await self.event_bus.publish(event)
        
        # Route to other event types
        if event.event_type in self._routes:
            for target_type in self._routes[event.event_type]:
                routed_event = Event(
                    event_type=target_type,
                    payload=event.payload,
                    source=event.event_type,
                    correlation_id=event.event_id
                )
                await self.event_bus.publish(routed_event)















