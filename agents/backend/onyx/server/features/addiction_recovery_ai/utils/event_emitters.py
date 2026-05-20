"""
Event emitter utilities
Functional event emitter patterns
"""

from typing import Callable, Dict, List, Any, Optional, TypeVar
from collections import defaultdict

T = TypeVar('T')


class EventEmitter:
    """
    Event emitter for pub/sub pattern
    """
    
    def __init__(self):
        self._listeners: Dict[str, List[Callable]] = defaultdict(list)
        self._once_listeners: Dict[str, List[Callable]] = defaultdict(list)
    
    def on(self, event: str, callback: Callable) -> Callable:
        """
        Subscribe to event
        
        Args:
            event: Event name
            callback: Callback function
        
        Returns:
            Unsubscribe function
        """
        self._listeners[event].append(callback)
        
        def unsubscribe():
            if callback in self._listeners[event]:
                self._listeners[event].remove(callback)
        
        return unsubscribe
    
    def once(self, event: str, callback: Callable) -> None:
        """Subscribe to event once"""
        self._once_listeners[event].append(callback)
    
    def emit(self, event: str, *args, **kwargs) -> None:
        """
        Emit event
        
        Args:
            event: Event name
            *args: Positional arguments
            **kwargs: Keyword arguments
        """
        # Call regular listeners
        for callback in self._listeners[event]:
            try:
                callback(*args, **kwargs)
            except Exception:
                pass
        
        # Call once listeners and remove
        for callback in self._once_listeners[event]:
            try:
                callback(*args, **kwargs)
            except Exception:
                pass
        
        self._once_listeners[event].clear()
    
    def off(self, event: str, callback: Optional[Callable] = None) -> None:
        """
        Unsubscribe from event
        
        Args:
            event: Event name
            callback: Optional specific callback to remove
        """
        if callback:
            if callback in self._listeners[event]:
                self._listeners[event].remove(callback)
        else:
            self._listeners[event].clear()
    
    def remove_all_listeners(self, event: Optional[str] = None) -> None:
        """Remove all listeners for event or all events"""
        if event:
            self._listeners[event].clear()
            self._once_listeners[event].clear()
        else:
            self._listeners.clear()
            self._once_listeners.clear()


def create_event_emitter() -> EventEmitter:
    """Create new event emitter"""
    return EventEmitter()

