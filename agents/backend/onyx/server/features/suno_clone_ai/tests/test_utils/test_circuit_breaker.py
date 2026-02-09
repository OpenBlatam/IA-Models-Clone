"""
Comprehensive Unit Tests for Circuit Breaker

Tests cover circuit breaker pattern with diverse test cases
"""

import pytest
import time
from unittest.mock import Mock, patch

from utils.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerStats,
    CircuitState,
    CircuitBreakerOpenError
)


class TestCircuitBreakerConfig:
    """Test cases for CircuitBreakerConfig"""
    
    def test_circuit_breaker_config_defaults(self):
        """Test default configuration"""
        config = CircuitBreakerConfig()
        assert config.failure_threshold == 5
        assert config.success_threshold == 2
        assert config.timeout == 60.0
        assert config.expected_exception == Exception
    
    def test_circuit_breaker_config_custom(self):
        """Test custom configuration"""
        config = CircuitBreakerConfig(
            failure_threshold=10,
            success_threshold=3,
            timeout=30.0
        )
        assert config.failure_threshold == 10
        assert config.success_threshold == 3
        assert config.timeout == 30.0


class TestCircuitBreaker:
    """Test cases for CircuitBreaker class"""
    
    def test_circuit_breaker_init(self):
        """Test initializing circuit breaker"""
        breaker = CircuitBreaker("test_breaker")
        assert breaker.name == "test_breaker"
        assert breaker.stats.state == CircuitState.CLOSED
        assert breaker.stats.failures == 0
    
    def test_circuit_breaker_call_success(self):
        """Test successful call keeps circuit closed"""
        breaker = CircuitBreaker("test")
        
        def success_func():
            return "success"
        
        result = breaker.call(success_func)
        assert result == "success"
        assert breaker.stats.state == CircuitState.CLOSED
        assert breaker.stats.total_successes == 1
    
    def test_circuit_breaker_call_failure(self):
        """Test failure increments failure count"""
        breaker = CircuitBreaker("test", CircuitBreakerConfig(failure_threshold=3))
        
        def fail_func():
            raise ValueError("Test error")
        
        with pytest.raises(ValueError):
            breaker.call(fail_func)
        
        assert breaker.stats.failures == 1
        assert breaker.stats.state == CircuitState.CLOSED
    
    def test_circuit_breaker_opens_after_threshold(self):
        """Test circuit opens after failure threshold"""
        breaker = CircuitBreaker("test", CircuitBreakerConfig(failure_threshold=2))
        
        def fail_func():
            raise ValueError("Error")
        
        # Fail twice to reach threshold
        for _ in range(2):
            with pytest.raises(ValueError):
                breaker.call(fail_func)
        
        assert breaker.stats.state == CircuitState.OPEN
        assert breaker.stats.failures >= 2
    
    def test_circuit_breaker_open_rejects_calls(self):
        """Test open circuit rejects calls"""
        breaker = CircuitBreaker("test", CircuitBreakerConfig(failure_threshold=1, timeout=100))
        breaker.stats.state = CircuitState.OPEN
        breaker.stats.last_failure_time = time.time()
        
        def func():
            return "should not execute"
        
        with pytest.raises(CircuitBreakerOpenError):
            breaker.call(func)
    
    def test_circuit_breaker_half_open_after_timeout(self):
        """Test circuit moves to half-open after timeout"""
        breaker = CircuitBreaker("test", CircuitBreakerConfig(timeout=0.1))
        breaker.stats.state = CircuitState.OPEN
        breaker.stats.last_failure_time = time.time() - 0.2  # Past timeout
        
        def success_func():
            return "success"
        
        result = breaker.call(success_func)
        assert result == "success"
        assert breaker.stats.state == CircuitState.HALF_OPEN
    
    def test_circuit_breaker_half_open_to_closed(self):
        """Test circuit closes after success threshold in half-open"""
        breaker = CircuitBreaker("test", CircuitBreakerConfig(success_threshold=2))
        breaker.stats.state = CircuitState.HALF_OPEN
        
        def success_func():
            return "success"
        
        # Succeed twice
        breaker.call(success_func)
        breaker.call(success_func)
        
        assert breaker.stats.state == CircuitState.CLOSED
        assert breaker.stats.failures == 0
    
    def test_circuit_breaker_half_open_failure_opens_again(self):
        """Test failure in half-open opens circuit again"""
        breaker = CircuitBreaker("test", CircuitBreakerConfig(failure_threshold=1))
        breaker.stats.state = CircuitState.HALF_OPEN
        
        def fail_func():
            raise ValueError("Error")
        
        with pytest.raises(ValueError):
            breaker.call(fail_func)
        
        assert breaker.stats.state == CircuitState.OPEN
    
    def test_circuit_breaker_custom_exception(self):
        """Test circuit breaker with custom exception type"""
        breaker = CircuitBreaker(
            "test",
            CircuitBreakerConfig(expected_exception=ValueError)
        )
        
        def raise_key_error():
            raise KeyError("Different error")
        
        # KeyError should not be caught
        with pytest.raises(KeyError):
            breaker.call(raise_key_error)
        
        # Should not increment failure count
        assert breaker.stats.failures == 0
    
    def test_circuit_breaker_stats_tracking(self):
        """Test statistics tracking"""
        breaker = CircuitBreaker("test")
        
        def success_func():
            return "success"
        
        def fail_func():
            raise ValueError("Error")
        
        breaker.call(success_func)
        breaker.call(success_func)
        
        try:
            breaker.call(fail_func)
        except ValueError:
            pass
        
        assert breaker.stats.total_calls == 3
        assert breaker.stats.total_successes == 2
        assert breaker.stats.total_failures == 1
    
    def test_circuit_breaker_reset_on_success(self):
        """Test failure count resets on success in closed state"""
        breaker = CircuitBreaker("test", CircuitBreakerConfig(failure_threshold=5))
        
        def fail_func():
            raise ValueError("Error")
        
        def success_func():
            return "success"
        
        # Fail once
        try:
            breaker.call(fail_func)
        except ValueError:
            pass
        
        assert breaker.stats.failures == 1
        
        # Success should reset
        breaker.call(success_func)
        assert breaker.stats.failures == 0


class TestCircuitBreakerOpenError:
    """Test cases for CircuitBreakerOpenError"""
    
    def test_circuit_breaker_open_error_message(self):
        """Test error message includes breaker name"""
        error = CircuitBreakerOpenError("Circuit breaker 'test' is OPEN")
        assert "test" in str(error)
        assert "OPEN" in str(error)















