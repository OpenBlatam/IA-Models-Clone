"""
Lifecycle management for KV Cache.

Provides hooks and lifecycle management.
"""
from __future__ import annotations

import logging
from typing import Callable, Any

from kv_cache.types import StatsDict

logger = logging.getLogger(__name__)


class LifecycleManager:
    """
    Manages cache lifecycle events.
    
    Provides hooks for initialization, cleanup, and state transitions.
    """
    
    def __init__(self) -> None:
        """Initialize lifecycle manager."""
        self._on_init_hooks: list[Callable[[Any], None]] = []
        self._on_clear_hooks: list[Callable[[Any], None]] = []
        self._on_evict_hooks: list[Callable[[Any, list[int]], None]] = []
        self._is_initialized = False
    
    def register_init_hook(self, hook: Callable[[Any], None]) -> None:
        """
        Register hook for initialization.
        
        Args:
            hook: Function to call on init (receives cache instance)
        """
        self._on_init_hooks.append(hook)
    
    def register_clear_hook(self, hook: Callable[[Any], None]) -> None:
        """
        Register hook for cache clear.
        
        Args:
            hook: Function to call on clear (receives cache instance)
        """
        self._on_clear_hooks.append(hook)
    
    def register_evict_hook(self, hook: Callable[[Any, list[int]], None]) -> None:
        """
        Register hook for eviction.
        
        Args:
            hook: Function to call on eviction (receives cache instance and positions)
        """
        self._on_evict_hooks.append(hook)
    
    def trigger_init(self, cache: Any) -> None:
        """
        Trigger initialization hooks.
        
        Args:
            cache: Cache instance
        """
        if self._is_initialized:
            return
        
        for hook in self._on_init_hooks:
            try:
                hook(cache)
            except Exception as e:
                logger.warning(f"Init hook failed: {e}")
        
        self._is_initialized = True
    
    def trigger_clear(self, cache: Any) -> None:
        """
        Trigger clear hooks.
        
        Args:
            cache: Cache instance
        """
        for hook in self._on_clear_hooks:
            try:
                hook(cache)
            except Exception as e:
                logger.warning(f"Clear hook failed: {e}")
    
    def trigger_evict(self, cache: Any, positions: list[int]) -> None:
        """
        Trigger eviction hooks.
        
        Args:
            cache: Cache instance
            positions: Positions being evicted
        """
        for hook in self._on_evict_hooks:
            try:
                hook(cache, positions)
            except Exception as e:
                logger.warning(f"Evict hook failed: {e}")


class CacheState:
    """
    Manages cache state and transitions.
    
    Tracks cache lifecycle state.
    """
    
    def __init__(self) -> None:
        """Initialize cache state."""
        self._state = "uninitialized"
        self._transitions: list[tuple[str, str, float]] = []  # (from, to, timestamp)
    
    def transition(self, new_state: str) -> None:
        """
        Transition to new state.
        
        Args:
            new_state: New state name
        """
        import time
        old_state = self._state
        self._state = new_state
        self._transitions.append((old_state, new_state, time.time()))
        logger.debug(f"State transition: {old_state} -> {new_state}")
    
    def get_state(self) -> str:
        """Get current state."""
        return self._state
    
    def get_transitions(self) -> list[tuple[str, str, float]]:
        """Get state transition history."""
        return list(self._transitions)
    
    def is_state(self, state: str) -> bool:
        """
        Check if in specific state.
        
        Args:
            state: State to check
            
        Returns:
            True if current state matches
        """
        return self._state == state



