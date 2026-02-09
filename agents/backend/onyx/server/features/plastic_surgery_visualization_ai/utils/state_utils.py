"""State management utilities."""

from typing import Any, Dict, Optional, Callable
from threading import Lock
import copy


class StateManager:
    """Thread-safe state manager."""
    
    def __init__(self, initial_state: Optional[Dict] = None):
        self._state = initial_state or {}
        self._lock = Lock()
        self._subscribers: list = []
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get state value.
        
        Args:
            key: State key
            default: Default value
            
        Returns:
            State value
        """
        with self._lock:
            return self._state.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """
        Set state value.
        
        Args:
            key: State key
            value: Value to set
        """
        with self._lock:
            old_value = self._state.get(key)
            self._state[key] = value
            self._notify(key, old_value, value)
    
    def update(self, updates: Dict[str, Any]) -> None:
        """
        Update multiple state values.
        
        Args:
            updates: Dictionary of updates
        """
        with self._lock:
            old_state = copy.deepcopy(self._state)
            self._state.update(updates)
            for key, value in updates.items():
                self._notify(key, old_state.get(key), value)
    
    def delete(self, key: str) -> None:
        """
        Delete state key.
        
        Args:
            key: State key
        """
        with self._lock:
            if key in self._state:
                old_value = self._state.pop(key)
                self._notify(key, old_value, None)
    
    def clear(self) -> None:
        """Clear all state."""
        with self._lock:
            old_state = copy.deepcopy(self._state)
            self._state.clear()
            for key, value in old_state.items():
                self._notify(key, value, None)
    
    def get_all(self) -> Dict[str, Any]:
        """
        Get all state.
        
        Returns:
            Copy of all state
        """
        with self._lock:
            return copy.deepcopy(self._state)
    
    def subscribe(self, callback: Callable[[str, Any, Any], None]) -> None:
        """
        Subscribe to state changes.
        
        Args:
            callback: Callback(key, old_value, new_value)
        """
        with self._lock:
            self._subscribers.append(callback)
    
    def _notify(self, key: str, old_value: Any, new_value: Any) -> None:
        """Notify subscribers of state change."""
        for callback in self._subscribers:
            try:
                callback(key, old_value, new_value)
            except Exception:
                pass  # Ignore subscriber errors


class ReactiveState:
    """Reactive state with computed values."""
    
    def __init__(self, initial_state: Optional[Dict] = None):
        self._state = StateManager(initial_state)
        self._computations: Dict[str, Callable] = {}
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get state value or computed value."""
        if key in self._computations:
            return self._computations[key](self._state.get_all())
        return self._state.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set state value."""
        self._state.set(key, value)
    
    def computed(self, key: str, compute_func: Callable[[Dict], Any]) -> None:
        """
        Register computed value.
        
        Args:
            key: Computed key
            compute_func: Function(state) -> value
        """
        self._computations[key] = compute_func
    
    def get_all(self) -> Dict[str, Any]:
        """Get all state including computed values."""
        state = self._state.get_all()
        for key, compute_func in self._computations.items():
            state[key] = compute_func(state)
        return state

