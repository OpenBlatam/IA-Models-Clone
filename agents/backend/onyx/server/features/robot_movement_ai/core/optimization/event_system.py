"""
Event system for trajectory optimization
"""

from enum import Enum
from typing import Callable, Dict, List, Any
import threading


class EventType(str, Enum):
    """Event types"""
    TRAJECTORY_START = "trajectory_start"
    TRAJECTORY_END = "trajectory_end"
    OPTIMIZATION_START = "optimization_start"
    OPTIMIZATION_END = "optimization_end"
    COLLISION_DETECTED = "collision_detected"
    PATH_FOUND = "path_found"


class EventEmitter:
    """Simple event emitter"""
    def __init__(self):
        self._listeners: Dict[str, List[Callable]] = {}
        self._lock = threading.Lock()
    
    def on(self, event: str, callback: Callable):
        """Register event listener"""
        with self._lock:
            if event not in self._listeners:
                self._listeners[event] = []
            self._listeners[event].append(callback)
    
    def emit(self, event: str, *args, **kwargs):
        """Emit an event"""
        with self._lock:
            if event in self._listeners:
                for callback in self._listeners[event]:
                    try:
                        callback(*args, **kwargs)
                    except Exception:
                        pass


_emitter = EventEmitter()


def get_event_emitter() -> EventEmitter:
    """Get the global event emitter"""
    return _emitter



