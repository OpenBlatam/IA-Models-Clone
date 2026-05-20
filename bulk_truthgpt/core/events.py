"""
Event System
============

Ultra-modular event system with advanced patterns.
"""

import logging
import threading
import asyncio
from typing import Dict, Any, Optional, Type, List, Callable, Union, get_type_hints
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
import inspect
import functools
import weakref
from collections import defaultdict

logger = logging.getLogger(__name__)

class EventPriority(int, Enum):
    """Event priorities."""
    LOWEST = 0
    LOW = 1
    NORMAL = 2
    HIGH = 3
    HIGHEST = 4

class EventScope(str, Enum):
    """Event scopes."""
    GLOBAL = "global"
    LOCAL = "local"
    SESSION = "session"
    REQUEST = "request"

@dataclass
class Event:
    """Event data structure."""
    name: str
    data: Dict[str, Any]
    source: str
    timestamp: float = field(default_factory=lambda: __import__('time').time())
    scope: EventScope = EventScope.GLOBAL
    priority: EventPriority = EventPriority.NORMAL
    metadata: Dict[str, Any] = field(default_factory=dict)
    propagation_stopped: bool = False
    default_prevented: bool = False

@dataclass
class EventHandler:
    """Event handler information."""
    name: str
    handler: Callable
    priority: EventPriority
    scope: EventScope
    async_handler: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

class EventBus:
    """Event bus for managing events."""
    
    def __init__(self):
        self._handlers: Dict[str, List[EventHandler]] = defaultdict(list)
        self._global_handlers: List[EventHandler] = []
        self._lock = threading.RLock()
        self._event_history: List[Event] = []
        self._max_history: int = 1000
        self._enabled: bool = True
    
    def subscribe(self, event_name: str, handler: Callable, priority: EventPriority = EventPriority.NORMAL,
                  scope: EventScope = EventScope.GLOBAL, async_handler: bool = False, 
                  metadata: Dict[str, Any] = None) -> str:
        """Subscribe to an event."""
        try:
            with self._lock:
                handler_id = f"{event_name}_{id(handler)}_{id(threading.current_thread())}"
                
                event_handler = EventHandler(
                    name=handler_id,
                    handler=handler,
                    priority=priority,
                    scope=scope,
                    async_handler=async_handler,
                    metadata=metadata or {}
                )
                
                if event_name == "*":
                    self._global_handlers.append(event_handler)
                else:
                    self._handlers[event_name].append(event_handler)
                    # Sort by priority
                    self._handlers[event_name].sort(key=lambda h: h.priority, reverse=True)
                
                logger.info(f"Handler {handler_id} subscribed to {event_name}")
                return handler_id
                
        except Exception as e:
            logger.error(f"Failed to subscribe to {event_name}: {str(e)}")
            raise
    
    def unsubscribe(self, event_name: str, handler_id: str) -> None:
        """Unsubscribe from an event."""
        try:
            with self._lock:
                if event_name == "*":
                    self._global_handlers = [h for h in self._global_handlers if h.name != handler_id]
                else:
                    if event_name in self._handlers:
                        self._handlers[event_name] = [h for h in self._handlers[event_name] if h.name != handler_id]
                
                logger.info(f"Handler {handler_id} unsubscribed from {event_name}")
        except Exception as e:
            logger.error(f"Failed to unsubscribe from {event_name}: {str(e)}")
            raise
    
    def publish(self, event: Event) -> None:
        """Publish an event."""
        try:
            if not self._enabled:
                return
            
            with self._lock:
                # Add to history
                self._event_history.append(event)
                if len(self._event_history) > self._max_history:
                    self._event_history.pop(0)
                
                # Get handlers for this event
                handlers = self._handlers.get(event.name, [])
                
                # Add global handlers
                handlers.extend(self._global_handlers)
                
                # Sort by priority
                handlers.sort(key=lambda h: h.priority, reverse=True)
                
                # Execute handlers
                for handler in handlers:
                    if event.propagation_stopped:
                        break
                    
                    try:
                        if handler.async_handler:
                            # Schedule async handler
                            asyncio.create_task(self._execute_async_handler(handler, event))
                        else:
                            # Execute sync handler
                            handler.handler(event)
                    except Exception as e:
                        logger.error(f"Handler {handler.name} failed: {str(e)}")
                
                logger.debug(f"Event {event.name} published")
                
        except Exception as e:
            logger.error(f"Failed to publish event {event.name}: {str(e)}")
            raise
    
    async def publish_async(self, event: Event) -> None:
        """Publish an event asynchronously."""
        try:
            if not self._enabled:
                return
            
            with self._lock:
                # Add to history
                self._event_history.append(event)
                if len(self._event_history) > self._max_history:
                    self._event_history.pop(0)
                
                # Get handlers for this event
                handlers = self._handlers.get(event.name, [])
                
                # Add global handlers
                handlers.extend(self._global_handlers)
                
                # Sort by priority
                handlers.sort(key=lambda h: h.priority, reverse=True)
                
                # Execute handlers
                for handler in handlers:
                    if event.propagation_stopped:
                        break
                    
                    try:
                        if handler.async_handler:
                            await handler.handler(event)
                        else:
                            # Execute sync handler in thread pool
                            loop = asyncio.get_event_loop()
                            await loop.run_in_executor(None, handler.handler, event)
                    except Exception as e:
                        logger.error(f"Handler {handler.name} failed: {str(e)}")
                
                logger.debug(f"Event {event.name} published asynchronously")
                
        except Exception as e:
            logger.error(f"Failed to publish async event {event.name}: {str(e)}")
            raise
    
    async def _execute_async_handler(self, handler: EventHandler, event: Event) -> None:
        """Execute async handler."""
        try:
            await handler.handler(event)
        except Exception as e:
            logger.error(f"Async handler {handler.name} failed: {str(e)}")
    
    def get_event_history(self, event_name: str = None, limit: int = 100) -> List[Event]:
        """Get event history."""
        try:
            with self._lock:
                if event_name:
                    events = [e for e in self._event_history if e.name == event_name]
                else:
                    events = self._event_history.copy()
                
                return events[-limit:] if limit else events
        except Exception as e:
            logger.error(f"Failed to get event history: {str(e)}")
            return []
    
    def clear_history(self) -> None:
        """Clear event history."""
        try:
            with self._lock:
                self._event_history.clear()
                logger.info("Event history cleared")
        except Exception as e:
            logger.error(f"Failed to clear event history: {str(e)}")
            raise
    
    def enable(self) -> None:
        """Enable event bus."""
        try:
            with self._lock:
                self._enabled = True
                logger.info("Event bus enabled")
        except Exception as e:
            logger.error(f"Failed to enable event bus: {str(e)}")
            raise
    
    def disable(self) -> None:
        """Disable event bus."""
        try:
            with self._lock:
                self._enabled = False
                logger.info("Event bus disabled")
        except Exception as e:
            logger.error(f"Failed to disable event bus: {str(e)}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Get event bus statistics."""
        try:
            with self._lock:
                return {
                    'enabled': self._enabled,
                    'total_handlers': sum(len(handlers) for handlers in self._handlers.values()) + len(self._global_handlers),
                    'event_types': len(self._handlers),
                    'history_size': len(self._event_history),
                    'handlers_by_event': {name: len(handlers) for name, handlers in self._handlers.items()}
                }
        except Exception as e:
            logger.error(f"Failed to get event bus stats: {str(e)}")
            return {}

class EventDispatcher:
    """Event dispatcher for specific event types."""
    
    def __init__(self, event_bus: EventBus, event_name: str):
        self.event_bus = event_bus
        self.event_name = event_name
    
    def dispatch(self, data: Dict[str, Any], source: str = "system", 
                 scope: EventScope = EventScope.GLOBAL, priority: EventPriority = EventPriority.NORMAL,
                 metadata: Dict[str, Any] = None) -> Event:
        """Dispatch an event."""
        try:
            event = Event(
                name=self.event_name,
                data=data,
                source=source,
                scope=scope,
                priority=priority,
                metadata=metadata or {}
            )
            
            self.event_bus.publish(event)
            return event
            
        except Exception as e:
            logger.error(f"Failed to dispatch event {self.event_name}: {str(e)}")
            raise
    
    async def dispatch_async(self, data: Dict[str, Any], source: str = "system",
                           scope: EventScope = EventScope.GLOBAL, priority: EventPriority = EventPriority.NORMAL,
                           metadata: Dict[str, Any] = None) -> Event:
        """Dispatch an event asynchronously."""
        try:
            event = Event(
                name=self.event_name,
                data=data,
                source=source,
                scope=scope,
                priority=priority,
                metadata=metadata or {}
            )
            
            await self.event_bus.publish_async(event)
            return event
            
        except Exception as e:
            logger.error(f"Failed to dispatch async event {self.event_name}: {str(e)}")
            raise

# Global event bus
event_bus = EventBus()

# Event decorators
def event_handler(event_name: str, priority: EventPriority = EventPriority.NORMAL,
                 scope: EventScope = EventScope.GLOBAL, async_handler: bool = False,
                 metadata: Dict[str, Any] = None):
    """Decorator to register an event handler."""
    def decorator(func):
        handler_id = event_bus.subscribe(
            event_name, func, priority, scope, async_handler, metadata
        )
        
        # Store handler ID for cleanup
        if not hasattr(func, '_event_handler_ids'):
            func._event_handler_ids = []
        func._event_handler_ids.append((event_name, handler_id))
        
        return func
    return decorator

def async_event_handler(event_name: str, priority: EventPriority = EventPriority.NORMAL,
                       scope: EventScope = EventScope.GLOBAL, metadata: Dict[str, Any] = None):
    """Decorator to register an async event handler."""
    return event_handler(event_name, priority, scope, True, metadata)

def global_event_handler(priority: EventPriority = EventPriority.NORMAL,
                         scope: EventScope = EventScope.GLOBAL, async_handler: bool = False,
                         metadata: Dict[str, Any] = None):
    """Decorator to register a global event handler."""
    return event_handler("*", priority, scope, async_handler, metadata)

# Event creation helpers
def create_event(name: str, data: Dict[str, Any], source: str = "system",
                scope: EventScope = EventScope.GLOBAL, priority: EventPriority = EventPriority.NORMAL,
                metadata: Dict[str, Any] = None) -> Event:
    """Create an event."""
    return Event(
        name=name,
        data=data,
        source=source,
        scope=scope,
        priority=priority,
        metadata=metadata or {}
    )

def publish_event(name: str, data: Dict[str, Any], source: str = "system",
                 scope: EventScope = EventScope.GLOBAL, priority: EventPriority = EventPriority.NORMAL,
                 metadata: Dict[str, Any] = None) -> Event:
    """Publish an event."""
    event = create_event(name, data, source, scope, priority, metadata)
    event_bus.publish(event)
    return event

async def publish_event_async(name: str, data: Dict[str, Any], source: str = "system",
                             scope: EventScope = EventScope.GLOBAL, priority: EventPriority = EventPriority.NORMAL,
                             metadata: Dict[str, Any] = None) -> Event:
    """Publish an event asynchronously."""
    event = create_event(name, data, source, scope, priority, metadata)
    await event_bus.publish_async(event)
    return event

# Event dispatchers for common events
user_events = EventDispatcher(event_bus, "user")
optimization_events = EventDispatcher(event_bus, "optimization")
performance_events = EventDispatcher(event_bus, "performance")
security_events = EventDispatcher(event_bus, "security")
system_events = EventDispatcher(event_bus, "system")









