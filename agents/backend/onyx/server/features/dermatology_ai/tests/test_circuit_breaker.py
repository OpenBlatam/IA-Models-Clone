"""
Tests for Circuit Breaker
Tests for circuit breaker pattern implementation
"""

import pytest
from unittest.mock import Mock, AsyncMock
import asyncio

from core.infrastructure.circuit_breaker import (
    CircuitBreaker,
    CircuitState,
    CircuitBreakerOpenError
)


class TestCircuitBreaker:
    """Tests for CircuitBreaker"""
    
    @pytest.fixture
    def circuit_breaker(self):
        """Create circuit breaker"""
        return CircuitBreaker(
            failure_threshold=3,
            timeout=5.0,
            expected_exception=Exception
        )
    
    def test_initial_state(self, circuit_breaker):
        """Test initial circuit breaker state"""
        assert circuit_breaker.state == CircuitState.CLOSED
    
    @pytest.mark.asyncio
    async def test_successful_call(self, circuit_breaker):
        """Test successful call through circuit breaker"""
        async def success_function():
            return "success"
        
        result = await circuit_breaker.call(success_function)
        
        assert result == "success"
        assert circuit_breaker.state == CircuitState.CLOSED
    
    @pytest.mark.asyncio
    async def test_failure_counting(self, circuit_breaker):
        """Test that failures are counted"""
        async def failing_function():
            raise Exception("Test failure")
        
        # Call failing function multiple times
        for i in range(3):
            with pytest.raises(Exception):
                await circuit_breaker.call(failing_function)
        
        # After threshold, circuit should open
        assert circuit_breaker.state == CircuitState.OPEN
    
    @pytest.mark.asyncio
    async def test_circuit_opens_after_threshold(self, circuit_breaker):
        """Test circuit opens after failure threshold"""
        async def failing_function():
            raise Exception("Test failure")
        
        # Trigger failures up to threshold
        for _ in range(3):
            with pytest.raises(Exception):
                await circuit_breaker.call(failing_function)
        
        # Next call should raise CircuitBreakerOpenError
        with pytest.raises(CircuitBreakerOpenError):
            await circuit_breaker.call(failing_function)
    
    @pytest.mark.asyncio
    async def test_circuit_half_open_state(self, circuit_breaker):
        """Test circuit half-open state after timeout"""
        async def failing_function():
            raise Exception("Test failure")
        
        # Open the circuit
        for _ in range(3):
            with pytest.raises(Exception):
                await circuit_breaker.call(failing_function)
        
        assert circuit_breaker.state == CircuitState.OPEN
        
        # Wait for timeout (in real scenario)
        # For testing, we can manually set to half-open
        circuit_breaker.state = CircuitState.HALF_OPEN
        
        # Successful call should close circuit
        async def success_function():
            return "success"
        
        result = await circuit_breaker.call(success_function)
        
        assert result == "success"
        assert circuit_breaker.state == CircuitState.CLOSED
    
    @pytest.mark.asyncio
    async def test_circuit_resets_on_success(self, circuit_breaker):
        """Test circuit resets failure count on success"""
        call_count = 0
        
        async def sometimes_failing():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise Exception("Failure")
            return "success"
        
        # First call fails
        with pytest.raises(Exception):
            await circuit_breaker.call(sometimes_failing)
        
        # Second call succeeds, should reset counter
        result = await circuit_breaker.call(sometimes_failing)
        assert result == "success"
        assert circuit_breaker.state == CircuitState.CLOSED


class TestCircuitBreakerIntegration:
    """Integration tests for circuit breaker"""
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_with_repository(self):
        """Test circuit breaker with repository calls"""
        circuit_breaker = CircuitBreaker(
            failure_threshold=2,
            timeout=1.0
        )
        
        mock_repository = Mock()
        mock_repository.get_by_id = AsyncMock(side_effect=Exception("DB Error"))
        
        async def repository_call():
            return await mock_repository.get_by_id("test-id")
        
        # First failure
        with pytest.raises(Exception):
            await circuit_breaker.call(repository_call)
        
        # Second failure - circuit should open
        with pytest.raises(Exception):
            await circuit_breaker.call(repository_call)
        
        # Third call should be blocked
        with pytest.raises(CircuitBreakerOpenError):
            await circuit_breaker.call(repository_call)
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_with_external_service(self):
        """Test circuit breaker with external service calls"""
        circuit_breaker = CircuitBreaker(
            failure_threshold=3,
            timeout=2.0
        )
        
        service_available = False
        
        async def external_service_call():
            if not service_available:
                raise ConnectionError("Service unavailable")
            return {"status": "ok"}
        
        # Simulate service failures
        for _ in range(3):
            with pytest.raises(ConnectionError):
                await circuit_breaker.call(external_service_call)
        
        # Circuit should be open
        assert circuit_breaker.state == CircuitState.OPEN
        
        # Service comes back online
        service_available = True
        circuit_breaker.state = CircuitState.HALF_OPEN
        
        # Should allow one call through
        result = await circuit_breaker.call(external_service_call)
        assert result["status"] == "ok"
        assert circuit_breaker.state == CircuitState.CLOSED



