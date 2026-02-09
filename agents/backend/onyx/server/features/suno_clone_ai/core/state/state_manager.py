"""
State Manager

Utilities for state management.
"""

import logging
from typing import Dict, Any, Optional
from threading import Lock

logger = logging.getLogger(__name__)


class StateManager:
    """Manage application state."""
    
    def __init__(self):
        """Initialize state manager."""
        self.state: Dict[str, Any] = {}
        self.lock = Lock()
    
    def get(
        self,
        key: str,
        default: Any = None
    ) -> Any:
        """
        Get state value.
        
        Args:
            key: State key
            default: Default value
            
        Returns:
            State value
        """
        with self.lock:
            return self.state.get(key, default)
    
    def set(
        self,
        key: str,
        value: Any
    ) -> None:
        """
        Set state value.
        
        Args:
            key: State key
            value: State value
        """
        with self.lock:
            self.state[key] = value
            logger.debug(f"State updated: {key}")
    
    def update(
        self,
        updates: Dict[str, Any]
    ) -> None:
        """
        Update multiple state values.
        
        Args:
            updates: Dictionary of updates
        """
        with self.lock:
            self.state.update(updates)
            logger.debug(f"State updated: {list(updates.keys())}")
    
    def delete(self, key: str) -> None:
        """
        Delete state value.
        
        Args:
            key: State key
        """
        with self.lock:
            if key in self.state:
                del self.state[key]
                logger.debug(f"State deleted: {key}")
    
    def clear(self) -> None:
        """Clear all state."""
        with self.lock:
            self.state.clear()
            logger.info("State cleared")
    
    def get_all(self) -> Dict[str, Any]:
        """Get all state."""
        with self.lock:
            return self.state.copy()


def create_state_manager() -> StateManager:
    """Create state manager."""
    return StateManager()


def get_state(
    manager: StateManager,
    key: str,
    default: Any = None
) -> Any:
    """Get state value."""
    return manager.get(key, default)


def set_state(
    manager: StateManager,
    key: str,
    value: Any
) -> None:
    """Set state value."""
    manager.set(key, value)



