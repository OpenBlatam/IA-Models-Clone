"""Mediator pattern utilities."""

from typing import Any, Dict, Callable, Optional, Set
from abc import ABC, abstractmethod
from datetime import datetime


class Mediator(ABC):
    """Base mediator interface."""
    
    @abstractmethod
    def notify(self, sender: Any, event: str, data: Any = None):
        """Notify mediator of event."""
        pass


class SimpleMediator(Mediator):
    """Simple mediator implementation."""
    
    def __init__(self):
        self.colleagues: Dict[str, Any] = {}
        self.handlers: Dict[str, list] = {}
    
    def register(self, name: str, colleague: Any):
        """Register colleague."""
        self.colleagues[name] = colleague
    
    def unregister(self, name: str):
        """Unregister colleague."""
        self.colleagues.pop(name, None)
    
    def subscribe(self, event: str, handler: Callable):
        """Subscribe to event."""
        if event not in self.handlers:
            self.handlers[event] = []
        self.handlers[event].append(handler)
    
    def unsubscribe(self, event: str, handler: Callable):
        """Unsubscribe from event."""
        if event in self.handlers:
            self.handlers[event].remove(handler)
    
    def notify(self, sender: Any, event: str, data: Any = None):
        """Notify all handlers of event."""
        if event in self.handlers:
            for handler in self.handlers[event]:
                try:
                    handler(sender, event, data)
                except Exception as e:
                    print(f"Error in handler for {event}: {e}")


class EventMediator(Mediator):
    """Event-based mediator."""
    
    def __init__(self):
        self.events: Dict[str, list] = {}
        self.event_history: list = []
    
    def on(self, event: str, handler: Callable):
        """Register event handler."""
        if event not in self.events:
            self.events[event] = []
        self.events[event].append(handler)
    
    def off(self, event: str, handler: Optional[Callable] = None):
        """Unregister event handler."""
        if event in self.events:
            if handler:
                self.events[event].remove(handler)
            else:
                self.events[event].clear()
    
    def emit(self, event: str, data: Any = None):
        """Emit event."""
        self.event_history.append({
            'event': event,
            'data': data,
            'timestamp': datetime.utcnow().isoformat()
        })
        
        if event in self.events:
            for handler in self.events[event]:
                try:
                    handler(data)
                except Exception as e:
                    print(f"Error in handler for {event}: {e}")
    
    def notify(self, sender: Any, event: str, data: Any = None):
        """Notify (alias for emit)."""
        self.emit(event, data)
    
    def get_history(self, event: Optional[str] = None) -> list:
        """Get event history."""
        if event:
            return [e for e in self.event_history if e['event'] == event]
        return self.event_history.copy()
    
    def clear_history(self):
        """Clear event history."""
        self.event_history.clear()


class Colleague:
    """Base colleague class."""
    
    def __init__(self, mediator: Mediator, name: str):
        self.mediator = mediator
        self.name = name
    
    def send(self, event: str, data: Any = None):
        """Send event through mediator."""
        self.mediator.notify(self, event, data)
    
    def receive(self, sender: Any, event: str, data: Any = None):
        """Receive event from mediator."""
        pass


class AsyncMediator(Mediator):
    """Async mediator for async operations."""
    
    def __init__(self):
        self.handlers: Dict[str, list] = {}
    
    def subscribe(self, event: str, handler: Callable):
        """Subscribe to event."""
        if event not in self.handlers:
            self.handlers[event] = []
        self.handlers[event].append(handler)
    
    async def notify(self, sender: Any, event: str, data: Any = None):
        """Notify handlers asynchronously."""
        if event in self.handlers:
            import asyncio
            tasks = []
            for handler in self.handlers[event]:
                if asyncio.iscoroutinefunction(handler):
                    tasks.append(handler(sender, event, data))
                else:
                    handler(sender, event, data)
            
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
    
    async def emit(self, event: str, data: Any = None):
        """Emit event asynchronously."""
        await self.notify(None, event, data)


class PriorityMediator(SimpleMediator):
    """Mediator with priority-based handling."""
    
    def __init__(self):
        super().__init__()
        self.priorities: Dict[str, int] = {}
    
    def subscribe(self, event: str, handler: Callable, priority: int = 0):
        """Subscribe with priority."""
        if event not in self.handlers:
            self.handlers[event] = []
        
        handler_id = id(handler)
        self.priorities[handler_id] = priority
        self.handlers[event].append(handler)
        
        self.handlers[event].sort(
            key=lambda h: self.priorities.get(id(h), 0),
            reverse=True
        )
    
    def notify(self, sender: Any, event: str, data: Any = None):
        """Notify handlers in priority order."""
        super().notify(sender, event, data)


