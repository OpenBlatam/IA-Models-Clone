"""
State Store

Persistent state storage.
"""

import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional
from core.state.state_manager import StateManager

logger = logging.getLogger(__name__)


class StateStore(StateManager):
    """State manager with persistence."""
    
    def __init__(self, store_path: str = "./state.json"):
        """
        Initialize state store.
        
        Args:
            store_path: Path to state file
        """
        super().__init__()
        self.store_path = Path(store_path)
        self._load()
    
    def _load(self) -> None:
        """Load state from file."""
        if self.store_path.exists():
            try:
                with open(self.store_path, 'r') as f:
                    self.state = json.load(f)
                logger.info(f"Loaded state from: {self.store_path}")
            except Exception as e:
                logger.error(f"Error loading state: {e}")
    
    def _save(self) -> None:
        """Save state to file."""
        try:
            self.store_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.store_path, 'w') as f:
                json.dump(self.state, f, indent=2)
            logger.debug(f"Saved state to: {self.store_path}")
        except Exception as e:
            logger.error(f"Error saving state: {e}")
    
    def set(
        self,
        key: str,
        value: Any
    ) -> None:
        """Set state value and save."""
        super().set(key, value)
        self._save()
    
    def update(
        self,
        updates: Dict[str, Any]
    ) -> None:
        """Update state and save."""
        super().update(updates)
        self._save()
    
    def delete(self, key: str) -> None:
        """Delete state value and save."""
        super().delete(key)
        self._save()
    
    def clear(self) -> None:
        """Clear state and save."""
        super().clear()
        self._save()


def create_state_store(store_path: str = "./state.json") -> StateStore:
    """Create state store."""
    return StateStore(store_path)



