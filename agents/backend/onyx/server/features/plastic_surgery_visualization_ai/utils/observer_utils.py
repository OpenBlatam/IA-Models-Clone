"""Observer pattern utilities."""

from typing import Callable, List, Any, Optional
from abc import ABC, abstractmethod


class Observer(ABC):
    """Observer interface."""
    
    @abstractmethod
    def update(self, subject: Any, *args, **kwargs) -> None:
        """Update observer."""
        pass


class Subject:
    """Subject for observer pattern."""
    
    def __init__(self):
        self._observers: List[Observer] = []
    
    def attach(self, observer: Observer) -> None:
        """Attach observer."""
        if observer not in self._observers:
            self._observers.append(observer)
    
    def detach(self, observer: Observer) -> None:
        """Detach observer."""
        if observer in self._observers:
            self._observers.remove(observer)
    
    def notify(self, *args, **kwargs) -> None:
        """Notify all observers."""
        for observer in self._observers:
            observer.update(self, *args, **kwargs)


class CallbackObserver(Observer):
    """Observer using callback function."""
    
    def __init__(self, callback: Callable):
        self.callback = callback
    
    def update(self, subject: Any, *args, **kwargs) -> None:
        """Update observer."""
        self.callback(subject, *args, **kwargs)


class ObservableProperty:
    """Property that notifies observers on change."""
    
    def __init__(self, initial_value: Any = None):
        self._value = initial_value
        self._observers: List[Callable] = []
    
    def get(self) -> Any:
        """Get value."""
        return self._value
    
    def set(self, value: Any) -> None:
        """Set value and notify observers."""
        old_value = self._value
        self._value = value
        self._notify(old_value, value)
    
    def subscribe(self, callback: Callable[[Any, Any], None]) -> None:
        """Subscribe to changes."""
        self._observers.append(callback)
    
    def unsubscribe(self, callback: Callable) -> None:
        """Unsubscribe from changes."""
        if callback in self._observers:
            self._observers.remove(callback)
    
    def _notify(self, old_value: Any, new_value: Any) -> None:
        """Notify observers."""
        for callback in self._observers:
            try:
                callback(old_value, new_value)
            except Exception:
                pass



