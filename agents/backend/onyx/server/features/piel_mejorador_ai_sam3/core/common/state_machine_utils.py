"""
State Machine Utilities for Piel Mejorador AI SAM3
===================================================

Unified state machine implementation utilities.
"""

import logging
from typing import TypeVar, Dict, Set, Optional, Callable, Any, List
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)

S = TypeVar('S', bound=Enum)


@dataclass
class Transition:
    """State transition definition."""
    from_state: Enum
    to_state: Enum
    condition: Optional[Callable[[Any], bool]] = None
    action: Optional[Callable[[Any], Any]] = None
    name: Optional[str] = None


@dataclass
class StateInfo:
    """State information."""
    state: Enum
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "state": self.state.value if isinstance(self.state, Enum) else str(self.state),
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


class StateMachine:
    """State machine implementation."""
    
    def __init__(
        self,
        initial_state: Enum,
        allowed_transitions: Optional[Dict[Enum, Set[Enum]]] = None
    ):
        """
        Initialize state machine.
        
        Args:
            initial_state: Initial state
            allowed_transitions: Optional dictionary of allowed transitions
        """
        self._state = initial_state
        self._history: List[StateInfo] = [StateInfo(initial_state)]
        self._transitions: List[Transition] = []
        self._allowed_transitions: Dict[Enum, Set[Enum]] = allowed_transitions or {}
        self._callbacks: Dict[Enum, List[Callable[[StateInfo], None]]] = {}
    
    @property
    def state(self) -> Enum:
        """Get current state."""
        return self._state
    
    @property
    def history(self) -> List[StateInfo]:
        """Get state history."""
        return self._history.copy()
    
    @property
    def current_info(self) -> StateInfo:
        """Get current state info."""
        return self._history[-1] if self._history else StateInfo(self._state)
    
    def add_transition(
        self,
        from_state: Enum,
        to_state: Enum,
        condition: Optional[Callable[[Any], bool]] = None,
        action: Optional[Callable[[Any], Any]] = None,
        name: Optional[str] = None
    ):
        """
        Add transition.
        
        Args:
            from_state: Source state
            to_state: Target state
            condition: Optional condition function
            action: Optional action function
            name: Optional transition name
        """
        transition = Transition(
            from_state=from_state,
            to_state=to_state,
            condition=condition,
            action=action,
            name=name
        )
        self._transitions.append(transition)
        logger.debug(f"Added transition: {from_state} -> {to_state}")
    
    def can_transition(self, to_state: Enum, context: Any = None) -> bool:
        """
        Check if transition is allowed.
        
        Args:
            to_state: Target state
            context: Optional context for condition check
            
        Returns:
            True if transition is allowed
        """
        # Check allowed transitions
        if self._allowed_transitions:
            allowed = self._allowed_transitions.get(self._state, set())
            if to_state not in allowed:
                return False
        
        # Check transition conditions
        for transition in self._transitions:
            if transition.from_state == self._state and transition.to_state == to_state:
                if transition.condition:
                    if not transition.condition(context):
                        return False
                return True
        
        # If no explicit transitions defined, allow if in allowed_transitions
        if self._allowed_transitions:
            allowed = self._allowed_transitions.get(self._state, set())
            return to_state in allowed
        
        # Default: allow if transition exists
        return any(
            t.from_state == self._state and t.to_state == to_state
            for t in self._transitions
        )
    
    def transition(
        self,
        to_state: Enum,
        context: Any = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Transition to new state.
        
        Args:
            to_state: Target state
            context: Optional context
            metadata: Optional metadata
            
        Returns:
            True if transition succeeded
        """
        if not self.can_transition(to_state, context):
            logger.warning(f"Transition not allowed: {self._state} -> {to_state}")
            return False
        
        # Execute transition action
        for transition in self._transitions:
            if transition.from_state == self._state and transition.to_state == to_state:
                if transition.action:
                    try:
                        transition.action(context)
                    except Exception as e:
                        logger.error(f"Transition action failed: {e}")
                        return False
                break
        
        # Update state
        old_state = self._state
        self._state = to_state
        
        # Create state info
        info = StateInfo(
            state=to_state,
            metadata=metadata or {}
        )
        self._history.append(info)
        
        # Call callbacks
        callbacks = self._callbacks.get(to_state, [])
        for callback in callbacks:
            try:
                callback(info)
            except Exception as e:
                logger.error(f"State callback failed: {e}")
        
        logger.info(f"State transition: {old_state} -> {to_state}")
        return True
    
    def on_state(
        self,
        state: Enum,
        callback: Callable[[StateInfo], None]
    ):
        """
        Register callback for state.
        
        Args:
            state: State to watch
            callback: Callback function
        """
        if state not in self._callbacks:
            self._callbacks[state] = []
        self._callbacks[state].append(callback)
    
    def is_in_state(self, state: Enum) -> bool:
        """
        Check if in state.
        
        Args:
            state: State to check
            
        Returns:
            True if in state
        """
        return self._state == state
    
    def reset(self, state: Enum):
        """
        Reset to state.
        
        Args:
            state: State to reset to
        """
        self._state = state
        self._history = [StateInfo(state)]
        logger.info(f"State machine reset to: {state}")


class StateMachineUtils:
    """Unified state machine utilities."""
    
    @staticmethod
    def create_machine(
        initial_state: Enum,
        allowed_transitions: Optional[Dict[Enum, Set[Enum]]] = None
    ) -> StateMachine:
        """
        Create state machine.
        
        Args:
            initial_state: Initial state
            allowed_transitions: Optional allowed transitions
            
        Returns:
            StateMachine
        """
        return StateMachine(initial_state, allowed_transitions)
    
    @staticmethod
    def create_transition(
        from_state: Enum,
        to_state: Enum,
        **kwargs
    ) -> Transition:
        """
        Create transition.
        
        Args:
            from_state: Source state
            to_state: Target state
            **kwargs: Additional options
            
        Returns:
            Transition
        """
        return Transition(from_state=from_state, to_state=to_state, **kwargs)


# Convenience functions
def create_machine(initial_state: Enum, **kwargs) -> StateMachine:
    """Create state machine."""
    return StateMachineUtils.create_machine(initial_state, **kwargs)


def create_transition(from_state: Enum, to_state: Enum, **kwargs) -> Transition:
    """Create transition."""
    return StateMachineUtils.create_transition(from_state, to_state, **kwargs)




