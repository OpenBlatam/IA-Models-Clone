"""
Circuit Breaker Events

Defines domain events for circuit breaker.
"""

from typing import Optional, Dict, Any, List, Callable
from datetime import datetime
from dataclasses import dataclass, field
import asyncio
import logging

from .circuit_types import CircuitState, CircuitBreakerEventType

logger = logging.getLogger(__name__)


@dataclass
class CircuitBreakerEvent:
    """Domain event for circuit breaker"""
    event_type: CircuitBreakerEventType
    circuit_name: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    old_state: Optional[CircuitState] = None
    new_state: Optional[CircuitState] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary"""
        return {
            "event_type": self.event_type.value,
            "circuit_name": self.circuit_name,
            "timestamp": self.timestamp.isoformat(),
            "old_state": self.old_state.value if self.old_state else None,
            "new_state": self.new_state.value if self.new_state else None,
            "metadata": self.metadata
        }


class EventEmitter:
    """Event emitter for circuit breaker events"""
    
    def __init__(self, circuit_name: str, max_event_history: int = 100):
        self.circuit_name = circuit_name
        self._event_handlers: List[Callable[[CircuitBreakerEvent], None]] = []
        self._event_history: List[CircuitBreakerEvent] = []
        self._max_event_history = max_event_history
    
    def on_event(self, handler: Callable[[CircuitBreakerEvent], None]):
        """Register event handler"""
        self._event_handlers.append(handler)
    
    def emit(self, event_type: CircuitBreakerEventType, **metadata):
        """Emit circuit breaker domain event"""
        event = CircuitBreakerEvent(
            event_type=event_type,
            circuit_name=self.circuit_name,
            metadata=metadata
        )
        
        # Add to history
        self._event_history.append(event)
        if len(self._event_history) > self._max_event_history:
            self._event_history.pop(0)
        
        # Invoke event handlers
        for handler in self._event_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    asyncio.create_task(handler(event))
                else:
                    handler(event)
            except Exception as e:
                logger.error(f"Error in circuit breaker event handler: {e}")
    
    def get_event_history(self, limit: Optional[int] = None) -> List[CircuitBreakerEvent]:
        """Get event history"""
        events = list(reversed(self._event_history))
        if limit:
            return events[:limit]
        return events
    
    def get_events_by_type(
        self,
        event_type: CircuitBreakerEventType,
        limit: Optional[int] = None
    ) -> List[CircuitBreakerEvent]:
        """Get events filtered by type"""
        events = [e for e in reversed(self._event_history) if e.event_type == event_type]
        if limit:
            return events[:limit]
        return events

