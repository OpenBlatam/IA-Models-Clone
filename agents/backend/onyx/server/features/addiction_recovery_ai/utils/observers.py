"""
Observer pattern utilities
Functional observer implementations
"""

from typing import Callable, List, TypeVar, Any
from functools import wraps

T = TypeVar('T')


class Observer:
    """
    Observer for reactive programming
    """
    
    def __init__(self):
        self._subscribers: List[Callable[[T], None]] = []
    
    def subscribe(self, callback: Callable[[T], None]) -> Callable[[], None]:
        """
        Subscribe to observer
        
        Args:
            callback: Function to call on updates
        
        Returns:
            Unsubscribe function
        """
        self._subscribers.append(callback)
        
        def unsubscribe():
            if callback in self._subscribers:
                self._subscribers.remove(callback)
        
        return unsubscribe
    
    def notify(self, value: T) -> None:
        """
        Notify all subscribers
        
        Args:
            value: Value to notify
        """
        for callback in self._subscribers:
            try:
                callback(value)
            except Exception:
                pass  # Ignore errors in subscribers
    
    def map(self, func: Callable[[T], Any]) -> 'Observer':
        """
        Map observer to new observer
        
        Args:
            func: Function to map values
        
        Returns:
            New observer with mapped values
        """
        new_observer = Observer()
        
        def mapped_callback(value: T):
            new_observer.notify(func(value))
        
        self.subscribe(mapped_callback)
        return new_observer
    
    def filter(self, predicate: Callable[[T], bool]) -> 'Observer':
        """
        Filter observer to new observer
        
        Args:
            predicate: Function to filter values
        
        Returns:
            New observer with filtered values
        """
        new_observer = Observer()
        
        def filtered_callback(value: T):
            if predicate(value):
                new_observer.notify(value)
        
        self.subscribe(filtered_callback)
        return new_observer


def create_observer() -> Observer:
    """Create new observer"""
    return Observer()

