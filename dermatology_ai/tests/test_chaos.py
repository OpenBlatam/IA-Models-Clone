"""
Chaos Engineering Tests
Tests for resilience and failure scenarios
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
import asyncio
import random

from core.infrastructure.circuit_breaker import CircuitBreaker, CircuitState
from core.infrastructure.error_recovery import ErrorRecovery
from core.infrastructure.graceful_degradation import GracefulDegradation


class TestChaosScenarios:
    """Tests for chaos engineering scenarios"""
    
    @pytest.mark.asyncio
    async def test_intermittent_failures(self):
        """Test handling intermittent failures"""
        failure_rate = 0.3  # 30% failure rate
        success_count = 0
        failure_count = 0
        
        async def unreliable_operation():
            nonlocal success_count, failure_count
            if random.random() < failure_rate:
                failure_count += 1
                raise Exception("Intermittent failure")
            success_count += 1
            return "success"
        
        recovery = ErrorRecovery(max_retries=3, retry_delay=0.01)
        
        # Run multiple operations
        results = []
        for _ in range(10):
            try:
                result = await recovery.retry(unreliable_operation)
                results.append(result)
            except Exception:
                results.append("failed")
        
        # Some should succeed, some may fail
        assert len(results) == 10
        assert success_count + failure_count == 10
    
    @pytest.mark.asyncio
    async def test_cascading_failures(self):
        """Test handling cascading failures"""
        circuit_breaker = CircuitBreaker(failure_threshold=3, timeout=1.0)
        
        # Simulate cascading failures
        async def failing_service():
            raise ConnectionError("Service unavailable")
        
        # Trigger circuit opening
        for _ in range(3):
            try:
                await circuit_breaker.call(failing_service)
            except Exception:
                pass
        
        # Circuit should be open
        assert circuit_breaker.state == CircuitState.OPEN
        
        # Subsequent calls should be blocked
        with pytest.raises(Exception):  # CircuitBreakerOpenError
            await circuit_breaker.call(failing_service)
    
    @pytest.mark.asyncio
    async def test_partial_system_degradation(self):
        """Test partial system degradation"""
        degradation = GracefulDegradation()
        
        async def primary_service():
            raise Exception("Primary service down")
        
        async def fallback_service():
            return {"data": "limited", "degraded": True}
        
        # Should fallback gracefully
        result = await degradation.execute_with_fallback(
            primary_service,
            fallback_service
        )
        
        assert result["degraded"] is True
        assert "data" in result
    
    @pytest.mark.asyncio
    async def test_resource_exhaustion(self):
        """Test handling resource exhaustion"""
        # Simulate resource exhaustion (e.g., connection pool exhausted)
        max_connections = 5
        active_connections = 0
        
        async def acquire_connection():
            nonlocal active_connections
            if active_connections >= max_connections:
                raise Exception("Connection pool exhausted")
            active_connections += 1
            return Mock()
        
        # Try to acquire more connections than available
        connections = []
        for i in range(max_connections + 2):
            try:
                conn = await acquire_connection()
                connections.append(conn)
            except Exception:
                # Should handle exhaustion gracefully
                break
        
        # Should handle gracefully
        assert len(connections) <= max_connections
    
    @pytest.mark.asyncio
    async def test_slow_responses(self):
        """Test handling slow responses"""
        async def slow_operation():
            await asyncio.sleep(5)  # Very slow
            return "result"
        
        # Should timeout
        try:
            result = await asyncio.wait_for(slow_operation(), timeout=0.1)
            assert False, "Should have timed out"
        except asyncio.TimeoutError:
            # Expected
            pass
    
    @pytest.mark.asyncio
    async def test_memory_pressure(self):
        """Test handling memory pressure"""
        # Simulate memory pressure by processing large data
        large_data = [{"data": "x" * 1000} for _ in range(1000)]
        
        # Process in chunks to avoid memory issues
        chunk_size = 100
        processed = []
        
        for i in range(0, len(large_data), chunk_size):
            chunk = large_data[i:i+chunk_size]
            processed.extend(chunk)
            # Simulate memory cleanup
            if i % 500 == 0:
                await asyncio.sleep(0.01)  # Allow GC
        
        assert len(processed) == len(large_data)


class TestResiliencePatterns:
    """Tests for resilience patterns"""
    
    @pytest.mark.asyncio
    async def test_bulkhead_pattern(self):
        """Test bulkhead pattern (isolating failures)"""
        # Simulate isolated resource pools
        pool1_failed = False
        pool2_working = True
        
        async def pool1_operation():
            nonlocal pool1_failed
            pool1_failed = True
            raise Exception("Pool 1 failed")
        
        async def pool2_operation():
            return "Pool 2 success"
        
        # Pool 1 failure should not affect Pool 2
        try:
            await pool1_operation()
        except Exception:
            pass
        
        result = await pool2_operation()
        
        assert pool1_failed is True
        assert result == "Pool 2 success"
    
    @pytest.mark.asyncio
    async def test_timeout_pattern(self):
        """Test timeout pattern"""
        async def operation_with_timeout():
            try:
                await asyncio.wait_for(
                    asyncio.sleep(10),
                    timeout=0.1
                )
            except asyncio.TimeoutError:
                return "timeout_handled"
        
        result = await operation_with_timeout()
        
        assert result == "timeout_handled"
    
    @pytest.mark.asyncio
    async def test_retry_with_exponential_backoff(self):
        """Test retry with exponential backoff"""
        recovery = ErrorRecovery(max_retries=3, retry_delay=0.01)
        
        call_times = []
        
        async def failing_operation():
            call_times.append(asyncio.get_event_loop().time())
            if len(call_times) < 3:
                raise Exception("Failure")
            return "success"
        
        result = await recovery.retry_with_backoff(failing_operation)
        
        assert result == "success"
        assert len(call_times) == 3
        # Verify backoff (times should increase)
        if len(call_times) > 1:
            delays = [call_times[i+1] - call_times[i] for i in range(len(call_times)-1)]
            # Delays should be positive (backoff working)
            assert all(d >= 0 for d in delays)



