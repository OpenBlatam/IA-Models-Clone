"""
Optimizer Callbacks
===================

Manages event callbacks for optimizer operations.
Encapsulates callback registration and notification logic.
"""

import logging
import threading
from typing import Callable, Dict, Any, List

logger = logging.getLogger(__name__)


class OptimizerCallbackManager:
    """
    Manages callbacks for optimizer events.
    
    Responsibilities:
    - Register callback functions
    - Notify callbacks of events
    - Handle callback errors gracefully
    """
    
    def __init__(self):
        """Initialize callback manager."""
        self._callbacks: List[Callable[[str, Dict[str, Any]], None]] = []
        self._lock = threading.Lock()
    
    def register(self, callback: Callable[[str, Dict[str, Any]], None]) -> None:
        """
        Register a callback function.
        
        Args:
            callback: Function that takes (event_type, data) as arguments
        """
        with self._lock:
            self._callbacks.append(callback)
            logger.debug(f"✅ Registered optimizer callback: {callback.__name__}")
    
    def notify(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Notify all registered callbacks of an event.
        
        Args:
            event_type: Type of event (e.g., 'optimizer_created')
            data: Event data dictionary
        """
        with self._lock:
            callbacks = self._callbacks.copy()
        
        for callback in callbacks:
            try:
                callback(event_type, data)
            except Exception as e:
                logger.warning(f"⚠️ Callback {callback.__name__} failed: {e}")
    
    def clear(self) -> None:
        """Clear all registered callbacks."""
        with self._lock:
            self._callbacks.clear()

