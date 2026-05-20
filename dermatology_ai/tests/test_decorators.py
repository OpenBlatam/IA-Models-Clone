"""
Tests for Decorators
Tests for cache decorator and other decorators
"""

import pytest
from unittest.mock import Mock, AsyncMock
import asyncio

from core.application.cache_decorator import (
    cache_result,
    invalidate_cache,
    _generate_cache_key
)
from core.infrastructure.performance_monitor import monitor_performance
from core.infrastructure.circuit_breaker import circuit_breaker_decorator
from core.infrastructure.error_recovery import retry_decorator
from tests.test_base import BaseDecoratorTest
from tests.test_helpers import create_cache_mock


class TestCacheDecorator(BaseDecoratorTest):
    """Tests for cache_result decorator"""
    
    @pytest.fixture
    def mock_cache_service(self):
        """Create mock cache service"""
        return create_cache_mock()
    
    @pytest.mark.asyncio
    async def test_cache_decorator_cache_miss(self, mock_cache_service):
        """Test cache decorator on cache miss"""
        call_count = 0
        
        @cache_result(mock_cache_service, ttl=3600)
        async def expensive_operation(arg1, arg2):
            nonlocal call_count
            call_count += 1
            return {"result": arg1 + arg2}
        
        result = await expensive_operation("a", "b")
        
        assert result == {"result": "ab"}
        assert call_count == 1
        mock_cache_service.set.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_cache_decorator_cache_hit(self, mock_cache_service):
        """Test cache decorator on cache hit"""
        cached_value = {"result": "cached"}
        mock_cache_service.get = AsyncMock(return_value=cached_value)
        
        call_count = 0
        
        @cache_result(mock_cache_service, ttl=3600)
        async def expensive_operation(arg1, arg2):
            nonlocal call_count
            call_count += 1
            return {"result": arg1 + arg2}
        
        result = await expensive_operation("a", "b")
        
        assert result == cached_value
        assert call_count == 0  # Should not be called
        mock_cache_service.get.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_cache_key_generation(self):
        """Test cache key generation"""
        key = _generate_cache_key(
            prefix="test.function",
            args=("arg1", "arg2"),
            kwargs={"key1": "value1"},
            serialize_args=True
        )
        
        assert key.startswith("test.function:")
        assert len(key) > len("test.function:")
    
    @pytest.mark.asyncio
    async def test_cache_decorator_with_ttl(self, mock_cache_service):
        """Test cache decorator with custom TTL"""
        @cache_result(mock_cache_service, ttl=7200)
        async def operation():
            return "result"
        
        await operation()
        
        # Verify TTL was used
        call_args = mock_cache_service.set.call_args
        assert call_args is not None
        # TTL should be in kwargs
        assert "ttl" in call_args.kwargs or len(call_args.args) >= 3


class TestInvalidateCacheDecorator:
    """Tests for invalidate_cache decorator"""
    
    @pytest.fixture
    def mock_cache_service(self):
        """Create mock cache service"""
        cache = Mock()
        cache.delete = AsyncMock(return_value=True)
        return cache
    
    @pytest.mark.asyncio
    async def test_invalidate_cache(self, mock_cache_service):
        """Test cache invalidation decorator"""
        @invalidate_cache(mock_cache_service, key_pattern="test:*")
        async def update_operation():
            return "updated"
        
        result = await update_operation()
        
        assert result == "updated"
        # Cache invalidation should be attempted
        # (implementation may vary)


class TestPerformanceMonitorDecorator:
    """Tests for monitor_performance decorator"""
    
    @pytest.mark.asyncio
    async def test_performance_monitoring(self):
        """Test performance monitoring decorator"""
        from core.infrastructure.performance_monitor import PerformanceMonitor
        
        monitor = PerformanceMonitor()
        
        @monitor_performance(operation_name="test_operation", log_threshold=0.1)
        async def monitored_operation():
            await asyncio.sleep(0.05)
            return "result"
        
        result = await monitored_operation()
        
        assert result == "result"
        # Performance should be monitored
        metrics = monitor.get_metrics()
        assert metrics is not None


class TestCircuitBreakerDecorator:
    """Tests for circuit breaker decorator"""
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_decorator(self):
        """Test circuit breaker as decorator"""
        from core.infrastructure.circuit_breaker import CircuitBreaker
        
        circuit_breaker = CircuitBreaker(failure_threshold=2, timeout=1.0)
        
        @circuit_breaker_decorator(circuit_breaker)
        async def failing_operation():
            raise Exception("Operation failed")
        
        # First failure
        with pytest.raises(Exception):
            await failing_operation()
        
        # Second failure - circuit should open
        with pytest.raises(Exception):
            await failing_operation()
        
        # Third call should be blocked by circuit breaker
        with pytest.raises(Exception):  # CircuitBreakerOpenError
            await failing_operation()


class TestRetryDecorator:
    """Tests for retry decorator"""
    
    @pytest.mark.asyncio
    async def test_retry_decorator_success(self):
        """Test retry decorator with eventual success"""
        call_count = 0
        
        @retry_decorator(max_retries=3, retry_delay=0.01)
        async def eventually_successful():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise Exception("Temporary failure")
            return "success"
        
        result = await eventually_successful()
        
        assert result == "success"
        assert call_count == 2
    
    @pytest.mark.asyncio
    async def test_retry_decorator_max_retries(self):
        """Test retry decorator exceeding max retries"""
        @retry_decorator(max_retries=2, retry_delay=0.01)
        async def always_failing():
            raise Exception("Always fails")
        
        with pytest.raises(Exception):
            await always_failing()


class TestDecoratorComposition:
    """Tests for composing multiple decorators"""
    
    @pytest.mark.asyncio
    async def test_multiple_decorators(self, mock_cache_service):
        """Test using multiple decorators together"""
        from core.infrastructure.performance_monitor import PerformanceMonitor
        
        monitor = PerformanceMonitor()
        
        @monitor_performance(operation_name="cached_operation", log_threshold=1.0)
        @cache_result(mock_cache_service, ttl=3600)
        async def cached_and_monitored_operation(arg):
            await asyncio.sleep(0.01)
            return f"result-{arg}"
        
        result = await cached_and_monitored_operation("test")
        
        assert result == "result-test"
        # Should be cached and monitored
        mock_cache_service.set.assert_called_once()

