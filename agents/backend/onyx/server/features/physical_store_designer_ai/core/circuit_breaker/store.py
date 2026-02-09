"""
Circuit Breaker State Persistence

State persistence framework for circuit breakers.
"""

from typing import Optional, Dict, Any
from abc import ABC, abstractmethod
import asyncio
import logging

from .breaker import CircuitBreaker
from .config import CircuitBreakerConfig
from .events import CircuitBreakerEvent, CircuitBreakerEventType

logger = logging.getLogger(__name__)


class CircuitBreakerStateStore(ABC):
    """
    Interface for persisting circuit breaker state.
    
    Implementations can save/load state to/from database, Redis, etc.
    """
    
    @abstractmethod
    async def save_state(self, breaker: CircuitBreaker) -> None:
        """Save circuit breaker state"""
        raise NotImplementedError
    
    @abstractmethod
    async def load_state(self, breaker_name: str) -> Optional[Dict[str, Any]]:
        """Load circuit breaker state"""
        raise NotImplementedError


class InMemoryStateStore(CircuitBreakerStateStore):
    """In-memory state store (for testing)"""
    
    def __init__(self):
        self._storage: Dict[str, Dict[str, Any]] = {}
    
    async def save_state(self, breaker: CircuitBreaker):
        """Save state to memory"""
        self._storage[breaker.name] = breaker.get_state()
    
    async def load_state(self, breaker_name: str) -> Optional[Dict[str, Any]]:
        """Load state from memory"""
        return self._storage.get(breaker_name)


def create_circuit_breaker_with_persistence(
    name: str,
    config: Optional[CircuitBreakerConfig] = None,
    state_store: Optional[CircuitBreakerStateStore] = None
) -> CircuitBreaker:
    """
    Create circuit breaker with state persistence.
    
    Args:
        name: Name of circuit breaker
        config: Configuration
        state_store: State store implementation
        
    Returns:
        CircuitBreaker with persistence enabled
    """
    breaker = CircuitBreaker(name=name, config=config)
    
    if state_store:
        # Load persisted state if available
        async def load_persisted_state():
            state = await state_store.load_state(name)
            if state:
                # Restore state (simplified - would need more logic for full restore)
                logger.info(f"Loaded persisted state for circuit breaker {name}")
        
        # Save state on changes
        async def save_state_handler(event: CircuitBreakerEvent):
            if event.event_type in (
                CircuitBreakerEventType.STATE_CHANGED,
                CircuitBreakerEventType.METRICS_UPDATED
            ):
                await state_store.save_state(breaker)
        
        breaker.on_event(save_state_handler)
        asyncio.create_task(load_persisted_state())
    
    return breaker




