"""
Tests for Circuit Breaker V2
=============================
"""

import pytest
import asyncio
from ..core.circuit_breaker_v2 import CircuitBreakerV2, CircuitState, CircuitStrategy


@pytest.fixture
def circuit_breaker_v2():
    """Create circuit breaker V2 for testing."""
    return CircuitBreakerV2(
        service_name="test_service",
        failure_threshold=5,
        timeout=60.0,
        strategy=CircuitStrategy.FAILURE_COUNT
    )


@pytest.mark.asyncio
async def test_call_success(circuit_breaker_v2):
    """Test successful call through circuit breaker V2."""
    async def success_func():
        return "success"
    
    result = await circuit_breaker_v2.call(success_func)
    
    assert result == "success"
    assert circuit_breaker_v2.state == CircuitState.CLOSED


@pytest.mark.asyncio
async def test_call_failure(circuit_breaker_v2):
    """Test failure handling."""
    async def fail_func():
        raise Exception("Test failure")
    
    # Record failures
    for _ in range(6):
        try:
            await circuit_breaker_v2.call(fail_func)
        except:
            pass
    
    # Circuit should be open after threshold
    assert circuit_breaker_v2.state == CircuitState.OPEN


@pytest.mark.asyncio
async def test_half_open_state(circuit_breaker_v2):
    """Test half-open state transition."""
    # Force open
    circuit_breaker_v2.state = CircuitState.OPEN
    circuit_breaker_v2.last_failure_time = asyncio.get_event_loop().time() - 120  # Past timeout
    
    async def success_func():
        return "success"
    
    # Should transition to half-open
    result = await circuit_breaker_v2.call(success_func)
    
    assert result == "success"
    assert circuit_breaker_v2.state in [CircuitState.HALF_OPEN, CircuitState.CLOSED]


@pytest.mark.asyncio
async def test_get_circuit_breaker_status(circuit_breaker_v2):
    """Test getting circuit breaker status."""
    status = circuit_breaker_v2.get_status()
    
    assert status is not None
    assert status["service"] == "test_service"
    assert status["state"] in [CircuitState.CLOSED, CircuitState.OPEN, CircuitState.HALF_OPEN]


@pytest.mark.asyncio
async def test_get_circuit_breaker_v2_summary(circuit_breaker_v2):
    """Test getting circuit breaker V2 summary."""
    async def func():
        return "result"
    
    await circuit_breaker_v2.call(func)
    
    summary = circuit_breaker_v2.get_circuit_breaker_v2_summary()
    
    assert summary is not None
    assert "service" in summary or "state" in summary


