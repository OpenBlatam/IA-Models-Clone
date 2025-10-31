"""
Observer Pattern Implementation

Event-driven communication pattern for loose coupling
between components in the AI History Comparison System.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Callable, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import weakref

logger = logging.getLogger(__name__)


class EventPriority(Enum):
    """Event priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class Event:
    """Base event class"""
    name: str
    data: Any = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    priority: EventPriority = EventPriority.NORMAL
    source: Optional[str] = None
    correlation_id: Optional[str] = None
    
    def __post_init__(self):
        if not self.correlation_id:
            self.correlation_id = f"{self.name}_{self.timestamp.timestamp()}"


class Observer(ABC):
    """Abstract observer interface"""
    
    @abstractmethod
    async def update(self, event: Event) -> None:
        """Handle event notification"""
        pass
    
    @abstractmethod
    def can_handle(self, event: Event) -> bool:
        """Check if observer can handle the event"""
        pass
    
    def get_priority(self) -> EventPriority:
        """Get observer priority (higher priority observers are called first)"""
        return EventPriority.NORMAL


class AsyncObserver(Observer):
    """Async observer implementation"""
    
    def __init__(self, handler: Callable[[Event], Any], 
                 event_filter: Optional[Callable[[Event], bool]] = None,
                 priority: EventPriority = EventPriority.NORMAL):
        self._handler = handler
        self._event_filter = event_filter
        self._priority = priority
        self._enabled = True
    
    async def update(self, event: Event) -> None:
        """Handle event notification"""
        if not self._enabled:
            return
        
        try:
            if self.can_handle(event):
                if asyncio.iscoroutinefunction(self._handler):
                    await self._handler(event)
                else:
                    self._handler(event)
        except Exception as e:
            logger.error(f"Error in observer {self.__class__.__name__}: {e}")
    
    def can_handle(self, event: Event) -> bool:
        """Check if observer can handle the event"""
        if not self._enabled:
            return False
        
        if self._event_filter:
            return self._event_filter(event)
        
        return True
    
    def get_priority(self) -> EventPriority:
        """Get observer priority"""
        return self._priority
    
    def enable(self):
        """Enable observer"""
        self._enabled = True
    
    def disable(self):
        """Disable observer"""
        self._enabled = False


class Subject:
    """Subject implementation for observer pattern"""
    
    def __init__(self):
        self._observers: List[Observer] = []
        self._observer_refs: Set[weakref.ref] = set()
        self._lock = asyncio.Lock()
        self._event_queue: asyncio.Queue = asyncio.Queue()
        self._processing_task: Optional[asyncio.Task] = None
        self._running = False
    
    async def start(self):
        """Start event processing"""
        if not self._running:
            self._running = True
            self._processing_task = asyncio.create_task(self._process_events())
    
    async def stop(self):
        """Stop event processing"""
        self._running = False
        if self._processing_task:
            self._processing_task.cancel()
            try:
                await self._processing_task
            except asyncio.CancelledError:
                pass
    
    def attach(self, observer: Observer) -> None:
        """Attach observer to subject"""
        if observer not in self._observers:
            self._observers.append(observer)
            # Sort by priority (higher priority first)
            self._observers.sort(key=lambda o: o.get_priority().value, reverse=True)
    
    def detach(self, observer: Observer) -> None:
        """Detach observer from subject"""
        if observer in self._observers:
            self._observers.remove(observer)
    
    def attach_weak(self, observer: Observer) -> None:
        """Attach observer using weak reference"""
        ref = weakref.ref(observer, self._remove_weak_ref)
        self._observer_refs.add(ref)
        self.attach(observer)
    
    def _remove_weak_ref(self, ref: weakref.ref):
        """Remove weak reference when observer is garbage collected"""
        self._observer_refs.discard(ref)
    
    async def notify(self, event: Event) -> None:
        """Notify all observers of an event"""
        await self._event_queue.put(event)
    
    async def notify_sync(self, event: Event) -> None:
        """Notify observers synchronously (immediate processing)"""
        await self._process_event(event)
    
    async def _process_events(self):
        """Process events from queue"""
        while self._running:
            try:
                event = await asyncio.wait_for(self._event_queue.get(), timeout=1.0)
                await self._process_event(event)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error processing event: {e}")
    
    async def _process_event(self, event: Event) -> None:
        """Process a single event"""
        # Clean up dead weak references
        dead_refs = [ref for ref in self._observer_refs if ref() is None]
        for ref in dead_refs:
            self._observer_refs.discard(ref)
        
        # Notify observers
        for observer in self._observers[:]:  # Copy to avoid modification during iteration
            try:
                if observer.can_handle(event):
                    await observer.update(event)
            except Exception as e:
                logger.error(f"Error notifying observer {observer.__class__.__name__}: {e}")
    
    def get_observer_count(self) -> int:
        """Get number of attached observers"""
        return len(self._observers)
    
    def get_observers(self) -> List[Observer]:
        """Get list of attached observers"""
        return self._observers.copy()


class EventBus:
    """Global event bus for system-wide event communication"""
    
    def __init__(self):
        self._subjects: Dict[str, Subject] = {}
        self._global_subject = Subject()
        self._lock = asyncio.Lock()
    
    async def start(self):
        """Start event bus"""
        await self._global_subject.start()
        for subject in self._subjects.values():
            await subject.start()
    
    async def stop(self):
        """Stop event bus"""
        await self._global_subject.stop()
        for subject in self._subjects.values():
            await subject.stop()
    
    async def subscribe(self, event_name: str, observer: Observer) -> None:
        """Subscribe to specific event"""
        async with self._lock:
            if event_name not in self._subjects:
                self._subjects[event_name] = Subject()
                await self._subjects[event_name].start()
            
            self._subjects[event_name].attach(observer)
    
    async def unsubscribe(self, event_name: str, observer: Observer) -> None:
        """Unsubscribe from specific event"""
        async with self._lock:
            if event_name in self._subjects:
                self._subjects[event_name].detach(observer)
    
    async def subscribe_global(self, observer: Observer) -> None:
        """Subscribe to all events"""
        self._global_subject.attach(observer)
    
    async def unsubscribe_global(self, observer: Observer) -> None:
        """Unsubscribe from all events"""
        self._global_subject.detach(observer)
    
    async def publish(self, event: Event) -> None:
        """Publish event to subscribers"""
        # Publish to specific subscribers
        if event.name in self._subjects:
            await self._subjects[event.name].notify(event)
        
        # Publish to global subscribers
        await self._global_subject.notify(event)
    
    async def publish_sync(self, event: Event) -> None:
        """Publish event synchronously"""
        # Publish to specific subscribers
        if event.name in self._subjects:
            await self._subjects[event.name].notify_sync(event)
        
        # Publish to global subscribers
        await self._global_subject.notify_sync(event)
    
    def get_subject(self, event_name: str) -> Optional[Subject]:
        """Get subject for specific event"""
        return self._subjects.get(event_name)
    
    def get_event_names(self) -> List[str]:
        """Get list of event names with subscribers"""
        return list(self._subjects.keys())


# Global event bus instance
event_bus = EventBus()


# Convenience functions
async def subscribe(event_name: str, observer: Observer):
    """Subscribe to event"""
    await event_bus.subscribe(event_name, observer)


async def unsubscribe(event_name: str, observer: Observer):
    """Unsubscribe from event"""
    await event_bus.unsubscribe(event_name, observer)


async def publish(event: Event):
    """Publish event"""
    await event_bus.publish(event)


async def publish_sync(event: Event):
    """Publish event synchronously"""
    await event_bus.publish_sync(event)





















