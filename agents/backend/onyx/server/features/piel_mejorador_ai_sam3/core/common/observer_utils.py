"""
Observer Pattern Utilities for Piel Mejorador AI SAM3
=====================================================

Unified observer pattern implementation utilities.
"""

import asyncio
import logging
from typing import Callable, Any, Dict, List, Optional, TypeVar, Awaitable
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class Observer:
    """Observer definition."""
    callback: Callable[[Any], Any]
    name: Optional[str] = None
    priority: int = 0
    enabled: bool = True


class Observable:
    """Observable object with observer pattern."""
    
    def __init__(self):
        """Initialize observable."""
        self._observers: Dict[str, List[Observer]] = defaultdict(list)
        self._lock = asyncio.Lock()
    
    def subscribe(
        self,
        event_type: str,
        callback: Callable[[Any], Any],
        name: Optional[str] = None,
        priority: int = 0
    ) -> Observer:
        """
        Subscribe to event type.
        
        Args:
            event_type: Event type
            callback: Callback function
            name: Optional observer name
            priority: Observer priority (higher = called first)
            
        Returns:
            Observer object
        """
        observer = Observer(
            callback=callback,
            name=name,
            priority=priority
        )
        
        self._observers[event_type].append(observer)
        # Sort by priority (higher first)
        self._observers[event_type].sort(key=lambda o: o.priority, reverse=True)
        
        logger.debug(f"Subscribed observer to {event_type}: {name or callback.__name__}")
        return observer
    
    def unsubscribe(self, event_type: str, observer: Observer):
        """
        Unsubscribe observer.
        
        Args:
            event_type: Event type
            observer: Observer to remove
        """
        if event_type in self._observers:
            try:
                self._observers[event_type].remove(observer)
            except ValueError:
                pass
    
    async def notify(
        self,
        event_type: str,
        data: Any,
        await_handlers: bool = True
    ):
        """
        Notify observers of event.
        
        Args:
            event_type: Event type
            data: Event data
            await_handlers: Whether to await async handlers
        """
        observers = self._observers.get(event_type, [])
        enabled_observers = [o for o in observers if o.enabled]
        
        if not enabled_observers:
            return
        
        if await_handlers:
            tasks = []
            for observer in enabled_observers:
                try:
                    if asyncio.iscoroutinefunction(observer.callback):
                        tasks.append(observer.callback(data))
                    else:
                        result = observer.callback(data)
                        if asyncio.iscoroutine(result):
                            tasks.append(result)
                except Exception as e:
                    logger.error(f"Error in observer {observer.name}: {e}")
            
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
        else:
            for observer in enabled_observers:
                try:
                    observer.callback(data)
                except Exception as e:
                    logger.error(f"Error in observer {observer.name}: {e}")
    
    def get_observers(self, event_type: Optional[str] = None) -> Dict[str, List[Observer]]:
        """
        Get observers.
        
        Args:
            event_type: Optional event type filter
            
        Returns:
            Dictionary of observers
        """
        if event_type:
            return {event_type: self._observers.get(event_type, [])}
        return dict(self._observers)
    
    def clear_observers(self, event_type: Optional[str] = None):
        """
        Clear observers.
        
        Args:
            event_type: Optional event type to clear (all if None)
        """
        if event_type:
            self._observers.pop(event_type, None)
        else:
            self._observers.clear()


class ObserverUtils:
    """Unified observer pattern utilities."""
    
    @staticmethod
    def create_observable() -> Observable:
        """
        Create observable object.
        
        Returns:
            Observable instance
        """
        return Observable()
    
    @staticmethod
    def create_observer(
        callback: Callable[[Any], Any],
        name: Optional[str] = None,
        priority: int = 0
    ) -> Observer:
        """
        Create observer.
        
        Args:
            callback: Callback function
            name: Optional observer name
            priority: Observer priority
            
        Returns:
            Observer object
        """
        return Observer(
            callback=callback,
            name=name,
            priority=priority
        )


# Convenience functions
def create_observable() -> Observable:
    """Create observable."""
    return ObserverUtils.create_observable()


def create_observer(callback: Callable[[Any], Any], **kwargs) -> Observer:
    """Create observer."""
    return ObserverUtils.create_observer(callback, **kwargs)




