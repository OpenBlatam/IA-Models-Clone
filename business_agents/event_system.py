"""Event system for internal event publishing."""
from typing import Dict, List, Callable, Any, Awaitable
from enum import Enum
import asyncio
import logging

logger = logging.getLogger(__name__)


class EventType(str, Enum):
    """System event types."""
    TASK_CREATED = "task.created"
    TASK_COMPLETED = "task.completed"
    TASK_FAILED = "task.failed"
    WEBHOOK_TRIGGERED = "webhook.triggered"
    BATCH_STARTED = "batch.started"
    BATCH_COMPLETED = "batch.completed"
    SYSTEM_ERROR = "system.error"


class EventBus:
    """Simple event bus for internal event handling."""
    
    def __init__(self):
        self.subscribers: Dict[str, List[Callable[[Dict[str, Any]], Awaitable[None]]]] = {}
    
    def subscribe(self, event_type: str, handler: Callable[[Dict[str, Any]], Awaitable[None]]):
        """Subscribe to an event type."""
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(handler)
    
    async def publish(self, event_type: str, payload: Dict[str, Any]):
        """Publish an event to all subscribers."""
        handlers = self.subscribers.get(event_type, [])
        handlers_all = self.subscribers.get("*", [])  # Wildcard subscribers
        
        for handler in handlers + handlers_all:
            try:
                await handler(payload)
            except Exception as e:
                logger.error(f"Event handler failed for {event_type}: {e}", exc_info=True)


# Global event bus instance
event_bus = EventBus()


