"""
Event-Driven Architecture Package
=================================

Event-driven architecture with message queues and event handling.
"""

from .manager import EventManager, EventBus
from .types import Event, EventType, EventHandler, EventSubscription
from .handlers import (
    AgentEventHandler, WorkflowEventHandler, 
    DocumentEventHandler, SystemEventHandler
)
from .queue import EventQueue, RedisEventQueue, InMemoryEventQueue

__all__ = [
    "EventManager",
    "EventBus",
    "Event",
    "EventType", 
    "EventHandler",
    "EventSubscription",
    "AgentEventHandler",
    "WorkflowEventHandler",
    "DocumentEventHandler", 
    "SystemEventHandler",
    "EventQueue",
    "RedisEventQueue",
    "InMemoryEventQueue"
]
