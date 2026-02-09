"""Event utilities and observer pattern."""

from typing import Callable, Dict, List, Any, Optional
from collections import defaultdict
from abc import ABC, abstractmethod
import asyncio


class EventEmitter:
    """Simple event emitter."""
    
    def __init__(self):
        self._listeners: Dict[str, List[Callable]] = defaultdict(list)
    
    def on(self, event: str, callback: Callable) -> None:
        """
        Register event listener.
        
        Args:
            event: Event name
            callback: Callback function
        """
        self._listeners[event].append(callback)
    
    def off(self, event: str, callback: Callable) -> None:
        """
        Remove event listener.
        
        Args:
            event: Event name
            callback: Callback function
        """
        if event in self._listeners:
            self._listeners[event].remove(callback)
    
    def emit(self, event: str, *args, **kwargs) -> None:
        """
        Emit event.
        
        Args:
            event: Event name
            *args: Positional arguments
            **kwargs: Keyword arguments
        """
        for callback in self._listeners.get(event, []):
            callback(*args, **kwargs)
    
    async def emit_async(self, event: str, *args, **kwargs) -> None:
        """
        Emit event asynchronously.
        
        Args:
            event: Event name
            *args: Positional arguments
            **kwargs: Keyword arguments
        """
        tasks = []
        for callback in self._listeners.get(event, []):
            if asyncio.iscoroutinefunction(callback):
                tasks.append(callback(*args, **kwargs))
            else:
                callback(*args, **kwargs)
        
        if tasks:
            await asyncio.gather(*tasks)
    
    def once(self, event: str, callback: Callable) -> None:
        """
        Register one-time event listener.
        
        Args:
            event: Event name
            callback: Callback function
        """
        def wrapper(*args, **kwargs):
            callback(*args, **kwargs)
            self.off(event, wrapper)
        
        self.on(event, wrapper)
    
    def remove_all_listeners(self, event: Optional[str] = None) -> None:
        """
        Remove all listeners for event or all events.
        
        Args:
            event: Event name (None for all events)
        """
        if event:
            self._listeners.pop(event, None)
        else:
            self._listeners.clear()


class EventBus:
    """Global event bus."""
    
    def __init__(self):
        self._emitter = EventEmitter()
    
    def subscribe(self, event: str, callback: Callable) -> None:
        """Subscribe to event."""
        self._emitter.on(event, callback)
    
    def unsubscribe(self, event: str, callback: Callable) -> None:
        """Unsubscribe from event."""
        self._emitter.off(event, callback)
    
    def publish(self, event: str, *args, **kwargs) -> None:
        """Publish event."""
        self._emitter.emit(event, *args, **kwargs)
    
    async def publish_async(self, event: str, *args, **kwargs) -> None:
        """Publish event asynchronously."""
        await self._emitter.emit_async(event, *args, **kwargs)


# Global event bus
_event_bus = EventBus()


def subscribe(event: str, callback: Callable) -> None:
    """Subscribe to event."""
    _event_bus.subscribe(event, callback)


def unsubscribe(event: str, callback: Callable) -> None:
    """Unsubscribe from event."""
    _event_bus.unsubscribe(event, callback)


def publish(event: str, *args, **kwargs) -> None:
    """Publish event."""
    _event_bus.publish(event, *args, **kwargs)


async def publish_async(event: str, *args, **kwargs) -> None:
    """Publish event asynchronously."""
    await _event_bus.publish_async(event, *args, **kwargs)

