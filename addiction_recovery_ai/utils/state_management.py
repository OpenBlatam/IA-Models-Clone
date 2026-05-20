"""
State management utilities
Functional state management patterns
"""

from typing import TypeVar, Callable, Dict, Any, Optional
from copy import deepcopy

T = TypeVar('T')
U = TypeVar('U')


class State:
    """
    Immutable state container
    """
    
    def __init__(self, data: Dict[str, Any]):
        self._data = deepcopy(data)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get value from state"""
        return self._data.get(key, default)
    
    def set(self, key: str, value: Any) -> 'State':
        """Set value in state (immutable)"""
        new_data = deepcopy(self._data)
        new_data[key] = value
        return State(new_data)
    
    def update(self, updates: Dict[str, Any]) -> 'State':
        """Update multiple values (immutable)"""
        new_data = deepcopy(self._data)
        new_data.update(updates)
        return State(new_data)
    
    def remove(self, key: str) -> 'State':
        """Remove key from state (immutable)"""
        new_data = deepcopy(self._data)
        new_data.pop(key, None)
        return State(new_data)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary"""
        return deepcopy(self._data)
    
    def map(self, func: Callable[[Dict[str, Any]], Dict[str, Any]]) -> 'State':
        """Map over state data"""
        new_data = func(deepcopy(self._data))
        return State(new_data)


def create_state(initial_data: Optional[Dict[str, Any]] = None) -> State:
    """Create new state"""
    return State(initial_data or {})


def state_reducer(
    state: State,
    action: Dict[str, Any]
) -> State:
    """
    Generic state reducer
    
    Args:
        state: Current state
        action: Action to apply
    
    Returns:
        New state
    """
    action_type = action.get("type")
    payload = action.get("payload", {})
    
    if action_type == "SET":
        return state.set(payload.get("key"), payload.get("value"))
    
    if action_type == "UPDATE":
        return state.update(payload)
    
    if action_type == "REMOVE":
        return state.remove(payload.get("key"))
    
    return state

