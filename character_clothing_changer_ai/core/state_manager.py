"""
State Management
================

System for managing application state.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Callable, Awaitable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from collections import defaultdict

logger = logging.getLogger(__name__)


class StateEvent(Enum):
    """State event type."""
    CREATED = "created"
    UPDATED = "updated"
    DELETED = "deleted"
    CHANGED = "changed"


@dataclass
class StateChange:
    """State change record."""
    key: str
    old_value: Any
    new_value: Any
    event: StateEvent
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class StateManager:
    """Application state manager."""
    
    def __init__(self):
        """Initialize state manager."""
        self.state: Dict[str, Any] = {}
        self.history: List[StateChange] = []
        self.watchers: Dict[str, List[Callable[[StateChange], Awaitable[None]]]] = defaultdict(list)
        self.max_history = 10000
        self._lock = asyncio.Lock()
    
    async def set(self, key: str, value: Any, metadata: Optional[Dict[str, Any]] = None):
        """
        Set state value.
        
        Args:
            key: State key
            value: State value
            metadata: Optional metadata
        """
        async with self._lock:
            old_value = self.state.get(key)
            self.state[key] = value
            
            # Record change
            change = StateChange(
                key=key,
                old_value=old_value,
                new_value=value,
                event=StateEvent.UPDATED if old_value is not None else StateEvent.CREATED,
                metadata=metadata or {}
            )
            
            self.history.append(change)
            if len(self.history) > self.max_history:
                self.history = self.history[-self.max_history:]
            
            # Notify watchers
            await self._notify_watchers(key, change)
    
    async def get(self, key: str, default: Any = None) -> Any:
        """
        Get state value.
        
        Args:
            key: State key
            default: Default value
            
        Returns:
            State value
        """
        async with self._lock:
            return self.state.get(key, default)
    
    async def delete(self, key: str):
        """
        Delete state value.
        
        Args:
            key: State key
        """
        async with self._lock:
            if key in self.state:
                old_value = self.state.pop(key)
                
                # Record change
                change = StateChange(
                    key=key,
                    old_value=old_value,
                    new_value=None,
                    event=StateEvent.DELETED
                )
                
                self.history.append(change)
                if len(self.history) > self.max_history:
                    self.history = self.history[-self.max_history:]
                
                # Notify watchers
                await self._notify_watchers(key, change)
    
    async def update(self, key: str, updater: Callable[[Any], Any]):
        """
        Update state value with function.
        
        Args:
            key: State key
            updater: Update function
        """
        async with self._lock:
            old_value = self.state.get(key)
            new_value = updater(old_value)
            self.state[key] = new_value
            
            # Record change
            change = StateChange(
                key=key,
                old_value=old_value,
                new_value=new_value,
                event=StateEvent.UPDATED
            )
            
            self.history.append(change)
            if len(self.history) > self.max_history:
                self.history = self.history[-self.max_history:]
            
            # Notify watchers
            await self._notify_watchers(key, change)
    
    async def watch(self, key: str, callback: Callable[[StateChange], Awaitable[None]]):
        """
        Watch for state changes.
        
        Args:
            key: State key (use "*" for all keys)
            callback: Callback function
        """
        self.watchers[key].append(callback)
        logger.debug(f"Added watcher for {key}")
    
    async def _notify_watchers(self, key: str, change: StateChange):
        """Notify watchers of state change."""
        # Notify specific watchers
        for callback in self.watchers.get(key, []):
            try:
                await callback(change)
            except Exception as e:
                logger.error(f"Error in watcher callback: {e}")
        
        # Notify wildcard watchers
        for callback in self.watchers.get("*", []):
            try:
                await callback(change)
            except Exception as e:
                logger.error(f"Error in wildcard watcher callback: {e}")
    
    async def get_all(self) -> Dict[str, Any]:
        """Get all state."""
        async with self._lock:
            return self.state.copy()
    
    async def get_history(self, key: Optional[str] = None, limit: int = 100) -> List[StateChange]:
        """
        Get state change history.
        
        Args:
            key: Optional key filter
            limit: Maximum number of changes
            
        Returns:
            List of state changes
        """
        async with self._lock:
            history = self.history
            
            if key:
                history = [c for c in history if c.key == key]
            
            return history[-limit:]
    
    async def clear(self, key: Optional[str] = None):
        """
        Clear state.
        
        Args:
            key: Optional key (clears all if not provided)
        """
        async with self._lock:
            if key:
                if key in self.state:
                    await self.delete(key)
            else:
                self.state.clear()
                logger.info("State cleared")

