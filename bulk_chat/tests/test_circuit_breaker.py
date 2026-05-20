"""
Tests for Circuit Breaker
==========================
"""

import pytest
import asyncio
from ..core.circuit_breaker import CircuitBreaker, CircuitBreakerConfig, CircuitState


@pytest.fixture
def circuit_breaker():
    """Create circuit breaker for testing."""
    config = CircuitBreakerConfig(
        failure_threshold=5,
        timeout=60.0,
        half_open_max_calls=3
    )
    return CircuitBreaker("test_service", config)


@pytest.mark.asyncio
async def test_circuit_closed_state(circuit_breaker):
    """Test circuit breaker in closed state."""
    assert circuit_breaker.state == CircuitState.CLOSED


@pytest.mark.asyncio
async def test_circuit_breaker_call_success(circuit_breaker):
    """Test successful call through circuit breaker."""
    async def success_func():
        return "success"
    
    result = await circuit_breaker.call(success_func)
    
    assert result == "success"
    assert circuit_breaker.state == CircuitState.CLOSED


@pytest.mark.asyncio
async def test_circuit_breaker_call_failure(circuit_breaker):
    """Test failure handling."""
    async def fail_func():
        raise Exception("Test failure")
    
    with pytest.raises(Exception):
        await circuit_breaker.call(fail_func)
    
    # After multiple failures, should open
    for _ in range(6):
        try:
            await circuit_breaker.call(fail_func)
        except:
            pass
    
    # Circuit should be open after threshold
    assert circuit_breaker.state == CircuitState.OPEN


@pytest.mark.asyncio
async def test_circuit_breaker_open_blocks_calls(circuit_breaker):
    """Test that open circuit blocks calls."""
    # Force open
    circuit_breaker.state = CircuitState.OPEN
    
    async def any_func():
        return "result"
    
    with pytest.raises(Exception):  # Should raise CircuitOpenError
        await circuit_breaker.call(any_func)


@pytest.mark.asyncio
async def test_get_circuit_breaker_status(circuit_breaker):
    """Test getting circuit breaker status."""
    status = circuit_breaker.get_status()
    
    assert status["service"] == "test_service"
    assert status["state"] in [CircuitState.CLOSED, CircuitState.OPEN, CircuitState.HALF_OPEN]
    assert "failure_count" in status or "success_count" in status


