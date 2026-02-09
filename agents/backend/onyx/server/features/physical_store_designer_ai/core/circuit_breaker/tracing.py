"""
OpenTelemetry Integration for Circuit Breaker

Provides distributed tracing support for circuit breakers.
"""

from typing import Dict, Any
import logging

from .breaker import CircuitBreaker
from .events import CircuitBreakerEvent

logger = logging.getLogger(__name__)


def get_trace_context(breaker: CircuitBreaker) -> Dict[str, Any]:
    """
    Get context for distributed tracing (OpenTelemetry compatible).
    
    Args:
        breaker: Circuit breaker instance
        
    Returns:
        Dictionary with trace context information
    """
    return {
        "circuit_breaker.name": breaker.name,
        "circuit_breaker.state": breaker.state.value,
        "circuit_breaker.failure_count": breaker.metrics.current_failure_count,
        "circuit_breaker.success_rate": breaker.metrics.success_rate,
        "circuit_breaker.health_score": breaker.get_health_score(),
    }


def add_tracing_to_circuit_breaker(breaker: CircuitBreaker):
    """
    Add OpenTelemetry tracing to circuit breaker events.
    
    This function sets up event handlers that automatically
    create spans for circuit breaker operations.
    
    Args:
        breaker: Circuit breaker to instrument
    """
    try:
        from opentelemetry import trace
        
        tracer = trace.get_tracer(__name__)
        
        async def trace_event_handler(event: CircuitBreakerEvent):
            """Handle events with OpenTelemetry tracing"""
            with tracer.start_as_current_span(
                f"circuit_breaker.{event.event_type.value}"
            ) as span:
                span.set_attribute("circuit_breaker.name", event.circuit_name)
                span.set_attribute("circuit_breaker.event_type", event.event_type.value)
                span.set_attribute("circuit_breaker.timestamp", event.timestamp.isoformat())
                
                if event.old_state:
                    span.set_attribute("circuit_breaker.old_state", event.old_state.value)
                if event.new_state:
                    span.set_attribute("circuit_breaker.new_state", event.new_state.value)
                
                for key, value in event.metadata.items():
                    span.set_attribute(f"circuit_breaker.{key}", str(value))
        
        breaker.on_event(trace_event_handler)
        logger.info(f"OpenTelemetry tracing enabled for circuit breaker {breaker.name}")
        
    except ImportError:
        logger.debug("OpenTelemetry not available, skipping tracing setup")




