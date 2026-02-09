"""
State Machine
=============

State machine implementation.
"""

import logging
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


@dataclass
class State:
    """State definition."""
    name: str
    on_enter: Optional[Callable] = None
    on_exit: Optional[Callable] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class Transition:
    """State transition definition."""
    from_state: str
    to_state: str
    condition: Optional[Callable] = None
    action: Optional[Callable] = None


class StateMachine:
    """State machine."""
    
    def __init__(self, initial_state: str):
        self._states: Dict[str, State] = {}
        self._transitions: List[Transition] = []
        self._current_state: str = initial_state
        self._history: List[str] = []
    
    def add_state(self, state: State):
        """Add state."""
        self._states[state.name] = state
        logger.info(f"Added state: {state.name}")
    
    def add_transition(self, transition: Transition):
        """Add transition."""
        self._transitions.append(transition)
        logger.info(f"Added transition: {transition.from_state} -> {transition.to_state}")
    
    async def transition(self, to_state: str, context: Any = None) -> bool:
        """Transition to state."""
        if to_state not in self._states:
            logger.error(f"State {to_state} not found")
            return False
        
        # Find valid transition
        valid_transition = None
        for transition in self._transitions:
            if transition.from_state == self._current_state and transition.to_state == to_state:
                if transition.condition is None or transition.condition(context):
                    valid_transition = transition
                    break
        
        if not valid_transition:
            logger.warning(f"No valid transition from {self._current_state} to {to_state}")
            return False
        
        # Exit current state
        current_state_obj = self._states[self._current_state]
        if current_state_obj.on_exit:
            if asyncio.iscoroutinefunction(current_state_obj.on_exit):
                await current_state_obj.on_exit(context)
            else:
                current_state_obj.on_exit(context)
        
        # Execute transition action
        if valid_transition.action:
            if asyncio.iscoroutinefunction(valid_transition.action):
                await valid_transition.action(context)
            else:
                valid_transition.action(context)
        
        # Enter new state
        new_state_obj = self._states[to_state]
        if new_state_obj.on_enter:
            if asyncio.iscoroutinefunction(new_state_obj.on_enter):
                await new_state_obj.on_enter(context)
            else:
                new_state_obj.on_enter(context)
        
        # Update state
        self._history.append(self._current_state)
        self._current_state = to_state
        
        logger.info(f"Transitioned from {self._history[-1]} to {self._current_state}")
        return True
    
    def get_current_state(self) -> str:
        """Get current state."""
        return self._current_state
    
    def get_history(self) -> List[str]:
        """Get state history."""
        return self._history.copy()
    
    def get_available_transitions(self) -> List[str]:
        """Get available transitions from current state."""
        return [
            transition.to_state
            for transition in self._transitions
            if transition.from_state == self._current_state
        ]


# Import asyncio
import asyncio















