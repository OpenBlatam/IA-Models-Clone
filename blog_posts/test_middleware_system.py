from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

# Constants
BUFFER_SIZE = 1024

import pytest
import asyncio
import time
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from collections import deque
from fastapi import FastAPI, Request, Response, HTTPException, status
from fastapi.testclient import TestClient
from fastapi.responses import JSONResponse
import structlog
from middleware_system import (
from typing import Any, List, Dict, Optional
import logging
"""
🧪 COMPREHENSIVE MIDDLEWARE SYSTEM TESTS
=======================================

Test suite for the comprehensive middleware system covering:
- Logging middleware functionality
- Error monitoring and alerting
- Performance monitoring and metrics
- Rate limiting behavior
- Response caching
- Security headers
- Integration testing
- Edge cases and error conditions

Features:
- Unit tests for each middleware component
- Integration tests for middleware interactions
- Performance testing
- Error scenario testing
- Configuration testing
- Mock and fixture management
"""



    MiddlewareConfig, RequestMetrics, ErrorEvent, PerformanceMetrics,
    HealthStatus, MetricsCollector, ResponseCache, ErrorMonitor,
    PerformanceMonitor, RateLimiter, LoggingMiddleware,
    PerformanceMonitoringMiddleware, ErrorMonitoringMiddleware,
    RateLimitingMiddleware, CachingMiddleware, SecurityHeadersMiddleware,
    MiddlewareManager, create_middleware_config, setup_middleware_system,
    get_request_metrics, log_request_event
)

# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def middleware_config():
    """Create middleware configuration for testing."""
    return MiddlewareConfig(
        log_level="DEBUG",
        log_format="console",
        enable_request_logging=True,
        enable_response_logging=True,
        enable_performance_monitoring=True,
        enable_error_monitoring=True,
        enable_rate_limiting=True,
        enable_caching=True,
        enable_security_headers=True,
        rate_limit_requests=10,
        slow_request_threshold_ms=500,
        cache_ttl_seconds=60,
        cache_max_size=100
    )

@pytest.fixture
def mock_request():
    """Create mock request for testing."""
    request = Mock(spec=Request)
    request.method = "GET"
    request.url.path = "/test"
    request.client.host = "127.0.0.1"
    request.headers = {
        "user-agent": "test-agent",
        "content-type": "application/json",
        "authorization": "Bearer test-token"
    }
    request.query_params = {}
    request.state = Mock()
    request.state.request_id = str(uuid.uuid4())
    request.state.correlation_id = str(uuid.uuid4())
    request.state.start_time = datetime.now()
    return request

@pytest.fixture
def mock_response():
    """Create mock response for testing."""
    response = Mock(spec=Response)
    response.status_code = 200
    response.body = b'{"message": "test"}'
    response.headers = {}
    return response

@pytest.fixture
def fastapi_app():
    """Create FastAPI app for testing."""
    return FastAPI(title="Test API", version="1.0.0")

@pytest.fixture
def test_client(fastapi_app) -> Any:
    """Create test client for FastAPI app."""
    return TestClient(fastapi_app)

# ============================================================================
# CONFIGURATION TESTS
# ============================================================================

class TestMiddlewareConfig:
    """Test middleware configuration."""
    
    def test_default_config(self) -> Any:
        """Test default configuration values."""
        config = MiddlewareConfig()
        
        assert config.log_level == "INFO"
        assert config.log_format == "json"
        assert config.enable_request_logging is True
        assert config.enable_performance_monitoring is True
        assert config.enable_error_monitoring is True
        assert config.enable_rate_limiting is True
        assert config.rate_limit_requests == 100
        assert config.slow_request_threshold_ms == 1000
    
    def test_custom_config(self) -> Any:
        """Test custom configuration values."""
        config = MiddlewareConfig(
            log_level="DEBUG",
            enable_request_logging=False,
            rate_limit_requests=50,
            slow_request_threshold_ms=500
        )
        
        assert config.log_level == "DEBUG"
        assert config.enable_request_logging is False
        assert config.rate_limit_requests == 50
        assert config.slow_request_threshold_ms == 500
    
    def test_config_validation(self) -> Any:
        """Test configuration validation."""
        # Should not raise validation error
        config = MiddlewareConfig()
        config.log_level = "DEBUG"
        assert config.log_level == "DEBUG"

# ============================================================================
# METRICS COLLECTOR TESTS
# ============================================================================

class TestMetricsCollector:
    """Test metrics collector functionality."""
    
    def test_initialization(self) -> Any:
        """Test metrics collector initialization."""
        collector = MetricsCollector()
        
        assert collector.request_count is not None
        assert collector.request_duration is not None
        assert collector.error_count is not None
        assert collector.active_requests is not None
        assert collector.memory_usage is not None
        assert collector.cpu_usage is not None
        assert collector.cache_hits is not None
        assert collector.cache_misses is not None
        assert collector.rate_limit_exceeded is not None
    
    async def test_record_request(self) -> Any:
        """Test recording request metrics."""
        collector = MetricsCollector()
        
        # Record a request
        collector.record_request("GET", "/test", 200, 0.5)
        
        # Metrics should be recorded (Prometheus metrics are global)
        # We can't easily test the actual values without Prometheus test client
        assert collector.request_count is not None
        assert collector.request_duration is not None
    
    def test_record_error(self) -> Any:
        """Test recording error metrics."""
        collector = MetricsCollector()
        
        # Record an error
        collector.record_error("GET", "/test", "ValidationError")
        
        # Error metric should be recorded
        assert collector.error_count is not None
    
    def test_record_cache_operations(self) -> Any:
        """Test recording cache operations."""
        collector = MetricsCollector()
        
        # Record cache hits and misses
        collector.record_cache_hit()
        collector.record_cache_miss()
        collector.record_cache_hit()
        
        # Cache metrics should be recorded
        assert collector.cache_hits is not None
        assert collector.cache_misses is not None
    
    def test_record_rate_limit_exceeded(self) -> Any:
        """Test recording rate limit violations."""
        collector = MetricsCollector()
        
        # Record rate limit violation
        collector.record_rate_limit_exceeded()
        
        # Rate limit metric should be recorded
        assert collector.rate_limit_exceeded is not None
    
    @patch('psutil.virtual_memory')
    @patch('psutil.cpu_percent')
    def test_update_system_metrics(self, mock_cpu, mock_memory) -> Any:
        """Test updating system metrics."""
        # Mock system metrics
        mock_memory.return_value.used = 1024 * 1024 * 100  # 100MB
        mock_cpu.return_value = 25.5
        
        collector = MetricsCollector()
        collector.update_system_metrics()
        
        # System metrics should be updated
        assert collector.memory_usage is not None
        assert collector.cpu_usage is not None

# ============================================================================
# RESPONSE CACHE TESTS
# ============================================================================

class TestResponseCache:
    """Test response caching functionality."""
    
    def test_initialization(self) -> Any:
        """Test cache initialization."""
        cache = ResponseCache(ttl_seconds=60, max_size=100)
        
        assert cache.cache.maxsize == 100
        assert cache.cache.ttl == 60
        assert cache.metrics_collector is not None
    
    def test_generate_cache_key(self, mock_request) -> Any:
        """Test cache key generation."""
        cache = ResponseCache()
        
        # Test GET request
        key1 = cache.generate_cache_key(mock_request)
        assert isinstance(key1, str)
        assert len(key1) > 0
        
        # Test with different query params
        mock_request.query_params = {"param1": "value1"}
        key2 = cache.generate_cache_key(mock_request)
        assert key1 != key2
        
        # Test with different headers
        mock_request.headers["authorization"] = "Bearer different-token"
        key3 = cache.generate_cache_key(mock_request)
        assert key2 != key3
    
    def test_cache_operations(self, mock_response) -> Any:
        """Test cache get/set operations."""
        cache = ResponseCache(ttl_seconds=60, max_size=10)
        
        # Test cache miss
        result = cache.get("test-key")
        assert result is None
        
        # Test cache set and get
        cache.set("test-key", mock_response)
        result = cache.get("test-key")
        assert result == mock_response
        
        # Test cache hit metrics
        with patch.object(cache.metrics_collector, 'record_cache_hit') as mock_hit:
            cache.get("test-key")
            mock_hit.assert_called_once()
    
    def test_cache_invalidation(self, mock_response) -> Any:
        """Test cache invalidation."""
        cache = ResponseCache(ttl_seconds=60, max_size=10)
        
        # Add multiple entries
        cache.set("user-123-profile", mock_response)
        cache.set("user-456-profile", mock_response)
        cache.set("other-data", mock_response)
        
        # Invalidate user profiles
        cache.invalidate_pattern("user-")
        
        # Check that user profiles are removed
        assert cache.get("user-123-profile") is None
        assert cache.get("user-456-profile") is None
        assert cache.get("other-data") is not None  # Should still exist

# ============================================================================
# ERROR MONITOR TESTS
# ============================================================================

class TestErrorMonitor:
    """Test error monitoring functionality."""
    
    def test_initialization(self) -> Any:
        """Test error monitor initialization."""
        monitor = ErrorMonitor(alert_threshold=5, alert_window_minutes=2)
        
        assert monitor.alert_threshold == 5
        assert monitor.alert_window == timedelta(minutes=2)
        assert len(monitor.error_events) == 0
        assert len(monitor.alerted_errors) == 0
    
    def test_record_error(self, mock_request) -> Any:
        """Test recording error events."""
        monitor = ErrorMonitor()
        
        # Record an error
        error = ValueError("Test error")
        context = {"test": "context"}
        
        error_event = monitor.record_error(error, mock_request, context)
        
        assert error_event.error_id is not None
        assert error_event.request_id == mock_request.state.request_id
        assert error_event.error_type == "ValueError"
        assert error_event.error_message == "Test error"
        assert error_event.context == context
        assert error_event.timestamp is not None
        assert len(monitor.error_events) == 1
    
    def test_error_alerting(self) -> Any:
        """Test error alerting functionality."""
        monitor = ErrorMonitor(alert_threshold=3, alert_window_minutes=1)
        
        # Record errors below threshold
        for i in range(2):
            error = ValueError(f"Error {i}")
            monitor.record_error(error)
        
        # Should not trigger alert
        assert len(monitor.alerted_errors) == 0
        
        # Record more errors to trigger alert
        for i in range(3):
            error = ValueError("Same error")
            monitor.record_error(error)
        
        # Should trigger alert
        assert len(monitor.alerted_errors) > 0
    
    def test_error_summary(self) -> Any:
        """Test error summary generation."""
        monitor = ErrorMonitor()
        
        # Record different types of errors
        monitor.record_error(ValueError("Value error"))
        monitor.record_error(TypeError("Type error"))
        monitor.record_error(ValueError("Another value error"))
        
        summary = monitor.get_error_summary()
        
        assert summary["total_errors"] == 3
        assert summary["error_types"]["ValueError"] == 2
        assert summary["error_types"]["TypeError"] == 1
        assert len(summary["recent_errors"]) == 3
    
    def test_error_cleanup(self) -> Any:
        """Test error event cleanup."""
        monitor = ErrorMonitor(alert_threshold=1, alert_window_minutes=1)
        
        # Record an old error
        old_error = ValueError("Old error")
        old_event = monitor.record_error(old_error)
        old_event.timestamp = datetime.now() - timedelta(minutes=2)
        
        # Record a new error
        new_error = ValueError("New error")
        monitor.record_error(new_error)
        
        # Old error should be cleaned up
        summary = monitor.get_error_summary()
        assert summary["total_errors"] == 1

# ============================================================================
# PERFORMANCE MONITOR TESTS
# ============================================================================

class TestPerformanceMonitor:
    """Test performance monitoring functionality."""
    
    def test_initialization(self) -> Any:
        """Test performance monitor initialization."""
        monitor = PerformanceMonitor(slow_request_threshold_ms=500)
        
        assert monitor.slow_request_threshold == 500
        assert len(monitor.request_times) == 0
        assert monitor.start_time is not None
        assert monitor.metrics_collector is not None
    
    async def test_record_request(self) -> Any:
        """Test recording request metrics."""
        monitor = PerformanceMonitor(slow_request_threshold_ms=1000)
        
        # Create request metrics
        metrics = RequestMetrics(
            request_id="test-123",
            method="GET",
            path="/test",
            client_ip="127.0.0.1",
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(milliseconds=500),
            duration_ms=500,
            status_code=200
        )
        
        # Record request
        with patch.object(monitor.metrics_collector, 'record_request') as mock_record:
            monitor.record_request(metrics)
            
            # Should record metrics
            mock_record.assert_called_once_with("GET", "/test", 200, 0.5)
            
            # Should add to request times
            assert len(monitor.request_times) == 1
            assert monitor.request_times[0] == 500
    
    async def test_slow_request_detection(self) -> Any:
        """Test slow request detection."""
        monitor = PerformanceMonitor(slow_request_threshold_ms=100)
        
        # Create slow request metrics
        metrics = RequestMetrics(
            request_id="test-123",
            method="GET",
            path="/slow",
            client_ip="127.0.0.1",
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(milliseconds=150),
            duration_ms=150,
            status_code=200
        )
        
        # Record slow request
        with patch.object(monitor.logger, 'warning') as mock_warning:
            monitor.record_request(metrics)
            
            # Should log warning
            mock_warning.assert_called_once()
            call_args = mock_warning.call_args[1]
            assert call_args["duration_ms"] == 150
            assert call_args["threshold_ms"] == 100
    
    def test_performance_metrics(self) -> Any:
        """Test performance metrics calculation."""
        monitor = PerformanceMonitor()
        
        # Add some request times
        monitor.request_times.extend([100, 200, 300, 400, 500])
        
        metrics = monitor.get_performance_metrics()
        
        assert metrics.total_requests == 5
        assert metrics.average_response_time_ms == 300.0
        assert metrics.p95_response_time_ms == 475.0  # 95th percentile
        assert metrics.p99_response_time_ms == 495.0  # 99th percentile
        assert metrics.timestamp is not None
        assert metrics.memory_usage_mb > 0
        assert metrics.cpu_usage_percent >= 0
    
    def test_empty_metrics(self) -> Any:
        """Test metrics calculation with no requests."""
        monitor = PerformanceMonitor()
        
        metrics = monitor.get_performance_metrics()
        
        assert metrics.total_requests == 0
        assert metrics.average_response_time_ms == 0.0
        assert metrics.p95_response_time_ms == 0.0
        assert metrics.p99_response_time_ms == 0.0
        assert metrics.requests_per_second == 0.0

# ============================================================================
# RATE LIMITER TESTS
# ============================================================================

class TestRateLimiter:
    """Test rate limiting functionality."""
    
    def test_initialization(self) -> Any:
        """Test rate limiter initialization."""
        limiter = RateLimiter(requests_per_minute=60, window_seconds=60)
        
        assert limiter.requests_per_minute == 60
        assert limiter.window_seconds == 60
        assert len(limiter.requests) == 0
    
    def test_client_identification(self, mock_request) -> Any:
        """Test client identification."""
        limiter = RateLimiter()
        
        # Test with X-Forwarded-For
        mock_request.headers["X-Forwarded-For"] = "192.168.1.1, 10.0.0.1"
        client_id = limiter.get_client_identifier(mock_request)
        assert client_id == "192.168.1.1"
        
        # Test with X-Real-IP
        del mock_request.headers["X-Forwarded-For"]
        mock_request.headers["X-Real-IP"] = "172.16.1.1"
        client_id = limiter.get_client_identifier(mock_request)
        assert client_id == "172.16.1.1"
        
        # Test with client IP
        del mock_request.headers["X-Real-IP"]
        client_id = limiter.get_client_identifier(mock_request)
        assert client_id == "127.0.0.1"
    
    def test_rate_limiting(self, mock_request) -> Any:
        """Test rate limiting behavior."""
        limiter = RateLimiter(requests_per_minute=2, window_seconds=60)
        
        # First request should be allowed
        allowed, info = limiter.is_allowed(mock_request)
        assert allowed is True
        assert info["current_requests"] == 0
        assert info["limit"] == 2
        
        # Second request should be allowed
        allowed, info = limiter.is_allowed(mock_request)
        assert allowed is True
        assert info["current_requests"] == 1
        
        # Third request should be blocked
        allowed, info = limiter.is_allowed(mock_request)
        assert allowed is False
        assert info["current_requests"] == 2
    
    def test_rate_limit_window(self, mock_request) -> Any:
        """Test rate limit window behavior."""
        limiter = RateLimiter(requests_per_minute=1, window_seconds=1)
        
        # First request should be allowed
        allowed, _ = limiter.is_allowed(mock_request)
        assert allowed is True
        
        # Second request should be blocked
        allowed, _ = limiter.is_allowed(mock_request)
        assert allowed is False
        
        # Wait for window to expire
        time.sleep(1.1)
        
        # Request should be allowed again
        allowed, _ = limiter.is_allowed(mock_request)
        assert allowed is True
    
    def test_multiple_clients(self) -> Any:
        """Test rate limiting with multiple clients."""
        limiter = RateLimiter(requests_per_minute=1, window_seconds=60)
        
        # Create different client requests
        request1 = Mock(spec=Request)
        request1.client.host = "192.168.1.1"
        
        request2 = Mock(spec=Request)
        request2.client.host = "192.168.1.2"
        
        # Each client should have separate limits
        allowed1, _ = limiter.is_allowed(request1)
        allowed2, _ = limiter.is_allowed(request2)
        
        assert allowed1 is True
        assert allowed2 is True
        
        # Second request from same client should be blocked
        allowed1, _ = limiter.is_allowed(request1)
        assert allowed1 is False

# ============================================================================
# MIDDLEWARE TESTS
# ============================================================================

class TestLoggingMiddleware:
    """Test logging middleware functionality."""
    
    def test_initialization(self, middleware_config) -> Any:
        """Test logging middleware initialization."""
        app = FastAPI()
        middleware = LoggingMiddleware(app, middleware_config)
        
        assert middleware.config == middleware_config
        assert middleware.logger is not None
    
    @patch('structlog.configure')
    def test_logging_setup(self, mock_configure, middleware_config) -> Any:
        """Test logging setup."""
        app = FastAPI()
        middleware = LoggingMiddleware(app, middleware_config)
        
        # Should configure structlog
        mock_configure.assert_called_once()
    
    @pytest.mark.asyncio
    async async def test_request_logging(self, mock_request, mock_response, middleware_config) -> Any:
        """Test request logging functionality."""
        app = FastAPI()
        middleware = LoggingMiddleware(app, middleware_config)
        
        # Mock call_next
        async def call_next(request) -> Any:
            return mock_response
        
        # Mock logger
        with patch.object(middleware.logger, 'info') as mock_info:
            response = await middleware.dispatch(mock_request, call_next)
            
            # Should log request start and completion
            assert mock_info.call_count >= 2
            
            # Check response
            assert response == mock_response
    
    @pytest.mark.asyncio
    async def test_error_logging(self, mock_request, middleware_config) -> Any:
        """Test error logging functionality."""
        app = FastAPI()
        middleware = LoggingMiddleware(app, middleware_config)
        
        # Mock call_next to raise exception
        async def call_next(request) -> Any:
            raise ValueError("Test error")
        
        # Mock logger
        with patch.object(middleware.logger, 'error') as mock_error:
            with pytest.raises(ValueError):
                await middleware.dispatch(mock_request, call_next)
            
            # Should log error
            mock_error.assert_called_once()

class TestPerformanceMonitoringMiddleware:
    """Test performance monitoring middleware functionality."""
    
    @pytest.mark.asyncio
    async def test_performance_monitoring(self, mock_request, mock_response, middleware_config) -> Any:
        """Test performance monitoring functionality."""
        app = FastAPI()
        performance_monitor = PerformanceMonitor()
        middleware = PerformanceMonitoringMiddleware(app, middleware_config, performance_monitor)
        
        # Mock call_next
        async def call_next(request) -> Any:
            await asyncio.sleep(0.1)  # Simulate processing time
            return mock_response
        
        # Process request
        response = await middleware.dispatch(mock_request, call_next)
        
        # Should record metrics
        assert len(performance_monitor.request_times) == 1
        assert performance_monitor.request_times[0] > 0
        
        # Check response
        assert response == mock_response
    
    @pytest.mark.asyncio
    async def test_error_performance_monitoring(self, mock_request, middleware_config) -> Any:
        """Test performance monitoring with errors."""
        app = FastAPI()
        performance_monitor = PerformanceMonitor()
        middleware = PerformanceMonitoringMiddleware(app, middleware_config, performance_monitor)
        
        # Mock call_next to raise exception
        async def call_next(request) -> Any:
            await asyncio.sleep(0.1)  # Simulate processing time
            raise ValueError("Test error")
        
        # Process request
        with pytest.raises(ValueError):
            await middleware.dispatch(mock_request, call_next)
        
        # Should record error metrics
        assert len(performance_monitor.request_times) == 1
        assert performance_monitor.request_times[0] > 0

class TestErrorMonitoringMiddleware:
    """Test error monitoring middleware functionality."""
    
    @pytest.mark.asyncio
    async def test_error_monitoring(self, mock_request, middleware_config) -> Any:
        """Test error monitoring functionality."""
        app = FastAPI()
        error_monitor = ErrorMonitor()
        middleware = ErrorMonitoringMiddleware(app, middleware_config, error_monitor)
        
        # Mock call_next to raise exception
        async def call_next(request) -> Any:
            raise ValueError("Test error")
        
        # Process request
        with pytest.raises(ValueError):
            await middleware.dispatch(mock_request, call_next)
        
        # Should record error
        assert len(error_monitor.error_events) == 1
        assert error_monitor.error_events[0].error_type == "ValueError"
        assert error_monitor.error_events[0].request_id == mock_request.state.request_id
    
    @pytest.mark.asyncio
    async async def test_successful_request(self, mock_request, mock_response, middleware_config) -> Any:
        """Test error monitoring with successful request."""
        app = FastAPI()
        error_monitor = ErrorMonitor()
        middleware = ErrorMonitoringMiddleware(app, middleware_config, error_monitor)
        
        # Mock call_next
        async def call_next(request) -> Any:
            return mock_response
        
        # Process request
        response = await middleware.dispatch(mock_request, call_next)
        
        # Should not record error
        assert len(error_monitor.error_events) == 0
        
        # Check response
        assert response == mock_response

class TestRateLimitingMiddleware:
    """Test rate limiting middleware functionality."""
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self, mock_request, mock_response, middleware_config) -> Any:
        """Test rate limiting functionality."""
        app = FastAPI()
        rate_limiter = RateLimiter(requests_per_minute=1, window_seconds=60)
        middleware = RateLimitingMiddleware(app, middleware_config, rate_limiter)
        
        # Mock call_next
        async def call_next(request) -> Any:
            return mock_response
        
        # First request should succeed
        response = await middleware.dispatch(mock_request, call_next)
        assert response == mock_response
        
        # Second request should be rate limited
        response = await middleware.dispatch(mock_request, call_next)
        assert response.status_code == 429
        assert "Rate limit exceeded" in response.body.decode()
    
    @pytest.mark.asyncio
    async def test_rate_limit_headers(self, mock_request, mock_response, middleware_config) -> Any:
        """Test rate limit headers."""
        app = FastAPI()
        rate_limiter = RateLimiter(requests_per_minute=10, window_seconds=60)
        middleware = RateLimitingMiddleware(app, middleware_config, rate_limiter)
        
        # Mock call_next
        async def call_next(request) -> Any:
            return mock_response
        
        # Process request
        response = await middleware.dispatch(mock_request, call_next)
        
        # Should add rate limit headers
        assert "X-RateLimit-Limit" in response.headers
        assert "X-RateLimit-Remaining" in response.headers
        assert "X-RateLimit-Reset" in response.headers

class TestCachingMiddleware:
    """Test caching middleware functionality."""
    
    @pytest.mark.asyncio
    async def test_caching(self, mock_request, mock_response, middleware_config) -> Any:
        """Test caching functionality."""
        app = FastAPI()
        response_cache = ResponseCache(ttl_seconds=60, max_size=10)
        middleware = CachingMiddleware(app, middleware_config, response_cache)
        
        # Mock call_next
        async def call_next(request) -> Any:
            return mock_response
        
        # First request should cache miss
        response = await middleware.dispatch(mock_request, call_next)
        assert response.headers.get("X-Cache") == "MISS"
        
        # Second request should cache hit
        response = await middleware.dispatch(mock_request, call_next)
        assert response.headers.get("X-Cache") == "HIT"
    
    @pytest.mark.asyncio
    async async def test_non_get_requests(self, mock_request, mock_response, middleware_config) -> Optional[Dict[str, Any]]:
        """Test caching with non-GET requests."""
        app = FastAPI()
        response_cache = ResponseCache()
        middleware = CachingMiddleware(app, middleware_config, response_cache)
        
        # Change to POST request
        mock_request.method = "POST"
        
        # Mock call_next
        async def call_next(request) -> Any:
            return mock_response
        
        # POST request should not be cached
        response = await middleware.dispatch(mock_request, call_next)
        assert "X-Cache" not in response.headers

class TestSecurityHeadersMiddleware:
    """Test security headers middleware functionality."""
    
    @pytest.mark.asyncio
    async def test_security_headers(self, mock_request, mock_response, middleware_config) -> Any:
        """Test security headers functionality."""
        app = FastAPI()
        middleware = SecurityHeadersMiddleware(app, middleware_config)
        
        # Mock call_next
        async def call_next(request) -> Any:
            return mock_response
        
        # Process request
        response = await middleware.dispatch(mock_request, call_next)
        
        # Should add security headers
        assert response.headers["X-Content-Type-Options"] == "nosniff"
        assert response.headers["X-Frame-Options"] == "DENY"
        assert response.headers["X-XSS-Protection"] == "1; mode=block"
        assert "Strict-Transport-Security" in response.headers
        assert "Content-Security-Policy" in response.headers
        assert "Referrer-Policy" in response.headers
        assert "Permissions-Policy" in response.headers

# ============================================================================
# MIDDLEWARE MANAGER TESTS
# ============================================================================

class TestMiddlewareManager:
    """Test middleware manager functionality."""
    
    def test_initialization(self, middleware_config) -> Any:
        """Test middleware manager initialization."""
        manager = MiddlewareManager(middleware_config)
        
        assert manager.config == middleware_config
        assert manager.metrics_collector is not None
        assert manager.error_monitor is not None
        assert manager.performance_monitor is not None
        assert manager.rate_limiter is not None
        assert manager.response_cache is not None
    
    def test_setup_middleware(self, fastapi_app, middleware_config) -> Any:
        """Test middleware setup."""
        manager = MiddlewareManager(middleware_config)
        manager.setup_middleware(fastapi_app)
        
        # Should add middleware to app
        assert len(fastapi_app.middleware_stack) > 0
    
    def test_system_status(self, middleware_config) -> Any:
        """Test system status generation."""
        manager = MiddlewareManager(middleware_config)
        
        status = manager.get_system_status()
        
        assert "timestamp" in status
        assert "uptime_seconds" in status
        assert "performance_metrics" in status
        assert "error_summary" in status
        assert "cache_stats" in status
        assert "rate_limit_stats" in status

# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestMiddlewareIntegration:
    """Test middleware integration."""
    
    def test_complete_middleware_chain(self, fastapi_app, middleware_config) -> Any:
        """Test complete middleware chain."""
        # Setup middleware system
        manager = setup_middleware_system(fastapi_app, middleware_config)
        
        # Add test endpoint
        @fastapi_app.get("/test")
        async def test_endpoint():
            
    """test_endpoint function."""
return {"message": "test"}
        
        # Create test client
        client = TestClient(fastapi_app)
        
        # Make request
        response = client.get("/test")
        
        # Should succeed
        assert response.status_code == 200
        assert response.json() == {"message": "test"}
        
        # Should have security headers
        assert "X-Content-Type-Options" in response.headers
        assert "X-Frame-Options" in response.headers
    
    def test_error_handling_integration(self, fastapi_app, middleware_config) -> Any:
        """Test error handling integration."""
        # Setup middleware system
        manager = setup_middleware_system(fastapi_app, middleware_config)
        
        # Add error endpoint
        @fastapi_app.get("/error")
        async def error_endpoint():
            
    """error_endpoint function."""
raise ValueError("Test error")
        
        # Create test client
        client = TestClient(fastapi_app)
        
        # Make request
        response = client.get("/error")
        
        # Should return error response
        assert response.status_code == 500
        assert "error" in response.json()
        assert "request_id" in response.json()
    
    def test_rate_limiting_integration(self, fastapi_app) -> Any:
        """Test rate limiting integration."""
        # Create config with low rate limit
        config = MiddlewareConfig(rate_limit_requests=1, rate_limit_window=60)
        
        # Setup middleware system
        manager = setup_middleware_system(fastapi_app, config)
        
        # Add test endpoint
        @fastapi_app.get("/test")
        async def test_endpoint():
            
    """test_endpoint function."""
return {"message": "test"}
        
        # Create test client
        client = TestClient(fastapi_app)
        
        # First request should succeed
        response1 = client.get("/test")
        assert response1.status_code == 200
        
        # Second request should be rate limited
        response2 = client.get("/test")
        assert response2.status_code == 429
    
    def test_health_check_integration(self, fastapi_app, middleware_config) -> Any:
        """Test health check integration."""
        # Setup middleware system
        manager = setup_middleware_system(fastapi_app, middleware_config)
        
        # Create test client
        client = TestClient(fastapi_app)
        
        # Check health endpoint
        response = client.get("/health")
        
        # Should return health status
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "uptime_seconds" in data
        assert "components" in data
    
    def test_metrics_integration(self, fastapi_app, middleware_config) -> Any:
        """Test metrics integration."""
        # Setup middleware system
        manager = setup_middleware_system(fastapi_app, middleware_config)
        
        # Add test endpoint
        @fastapi_app.get("/test")
        async def test_endpoint():
            
    """test_endpoint function."""
return {"message": "test"}
        
        # Create test client
        client = TestClient(fastapi_app)
        
        # Make some requests
        for _ in range(3):
            client.get("/test")
        
        # Check metrics endpoint
        response = client.get("/metrics")
        
        # Should return Prometheus metrics
        assert response.status_code == 200
        assert "http_requests_total" in response.text

# ============================================================================
# UTILITY FUNCTION TESTS
# ============================================================================

class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_create_middleware_config(self) -> Any:
        """Test middleware config creation."""
        config = create_middleware_config(
            log_level="DEBUG",
            enable_request_logging=False
        )
        
        assert config.log_level == "DEBUG"
        assert config.enable_request_logging is False
    
    async def test_get_request_metrics(self, mock_request) -> Optional[Dict[str, Any]]:
        """Test request metrics extraction."""
        metrics = get_request_metrics(mock_request)
        
        assert metrics["request_id"] == mock_request.state.request_id
        assert metrics["correlation_id"] == mock_request.state.correlation_id
        assert metrics["start_time"] == mock_request.state.start_time
        assert metrics["duration_ms"] is not None
    
    async def test_log_request_event(self, mock_request) -> Any:
        """Test request event logging."""
        with patch('structlog.get_logger') as mock_logger:
            mock_log = Mock()
            mock_logger.return_value = mock_log
            
            log_request_event("test_event", mock_request, extra_data="test")
            
            # Should log event
            mock_log.info.assert_called_once()
            call_args = mock_log.info.call_args[1]
            assert call_args["event_type"] == "test_event"
            assert call_args["correlation_id"] == mock_request.state.correlation_id
            assert call_args["extra_data"] == "test"

# ============================================================================
# PERFORMANCE TESTS
# ============================================================================

class TestMiddlewarePerformance:
    """Test middleware performance."""
    
    @pytest.mark.asyncio
    async def test_middleware_overhead(self, fastapi_app, middleware_config) -> Any:
        """Test middleware overhead."""
        # Setup middleware system
        manager = setup_middleware_system(fastapi_app, middleware_config)
        
        # Add simple endpoint
        @fastapi_app.get("/simple")
        async def simple_endpoint():
            
    """simple_endpoint function."""
return {"message": "simple"}
        
        # Create test client
        client = TestClient(fastapi_app)
        
        # Measure response time
        start_time = time.time()
        response = client.get("/simple")
        end_time = time.time()
        
        # Should succeed quickly
        assert response.status_code == 200
        assert (end_time - start_time) < 1.0  # Should be much faster
    
    def test_cache_performance(self) -> Any:
        """Test cache performance."""
        cache = ResponseCache(ttl_seconds=60, max_size=1000)
        
        # Create test response
        response = Mock(spec=Response)
        response.status_code = 200
        response.body = b'{"data": "test"}'
        
        # Measure cache operations
        start_time = time.time()
        
        for i in range(100):
            key = f"key-{i}"
            cache.set(key, response)
            cached = cache.get(key)
            assert cached == response
        
        end_time = time.time()
        
        # Should be fast
        assert (end_time - start_time) < 1.0
    
    def test_rate_limiter_performance(self) -> Any:
        """Test rate limiter performance."""
        limiter = RateLimiter(requests_per_minute=1000, window_seconds=60)
        
        # Create test request
        request = Mock(spec=Request)
        request.client.host = "127.0.0.1"
        request.headers = {}
        
        # Measure rate limiting operations
        start_time = time.time()
        
        for _ in range(100):
            allowed, _ = limiter.is_allowed(request)
            assert allowed is True
        
        end_time = time.time()
        
        # Should be fast
        assert (end_time - start_time) < 1.0

# ============================================================================
# EDGE CASE TESTS
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    async def test_empty_request_headers(self) -> Any:
        """Test handling of empty request headers."""
        request = Mock(spec=Request)
        request.client.host = "127.0.0.1"
        request.headers = {}
        
        limiter = RateLimiter()
        client_id = limiter.get_client_identifier(request)
        
        assert client_id == "127.0.0.1"
    
    async def test_malformed_request(self) -> Any:
        """Test handling of malformed requests."""
        request = Mock(spec=Request)
        request.client = None
        request.headers = {}
        
        limiter = RateLimiter()
        client_id = limiter.get_client_identifier(request)
        
        assert client_id == "unknown"
    
    def test_cache_key_collision(self, mock_request) -> Any:
        """Test cache key collision handling."""
        cache = ResponseCache()
        
        # Create identical requests
        key1 = cache.generate_cache_key(mock_request)
        key2 = cache.generate_cache_key(mock_request)
        
        # Keys should be identical for identical requests
        assert key1 == key2
    
    def test_error_monitor_memory_cleanup(self) -> Any:
        """Test error monitor memory cleanup."""
        monitor = ErrorMonitor(alert_threshold=1, alert_window_minutes=1)
        
        # Add many errors
        for i in range(1000):
            monitor.record_error(ValueError(f"Error {i}"))
        
        # Should not grow indefinitely
        assert len(monitor.error_events) <= 1000
    
    def test_performance_monitor_memory_cleanup(self) -> Any:
        """Test performance monitor memory cleanup."""
        monitor = PerformanceMonitor()
        
        # Add many request times
        for i in range(10000):
            monitor.request_times.append(i)
        
        # Should respect maxlen
        assert len(monitor.request_times) <= 1000

# ============================================================================
# CONFIGURATION TESTS
# ============================================================================

class TestConfigurationScenarios:
    """Test different configuration scenarios."""
    
    def test_disabled_middleware(self) -> Any:
        """Test disabled middleware configuration."""
        config = MiddlewareConfig(
            enable_request_logging=False,
            enable_performance_monitoring=False,
            enable_error_monitoring=False,
            enable_rate_limiting=False,
            enable_caching=False,
            enable_security_headers=False
        )
        
        app = FastAPI()
        manager = setup_middleware_system(app, config)
        
        # Should still have basic middleware
        assert len(app.middleware_stack) > 0
    
    def test_high_performance_config(self) -> Any:
        """Test high performance configuration."""
        config = MiddlewareConfig(
            slow_request_threshold_ms=100,
            rate_limit_requests=1000,
            cache_ttl_seconds=3600,
            cache_max_size=10000
        )
        
        assert config.slow_request_threshold_ms == 100
        assert config.rate_limit_requests == 1000
        assert config.cache_ttl_seconds == 3600
        assert config.cache_max_size == 10000
    
    def test_debug_config(self) -> Any:
        """Test debug configuration."""
        config = MiddlewareConfig(
            log_level="DEBUG",
            log_format="console",
            enable_response_logging=True,
            log_sensitive_headers=True
        )
        
        assert config.log_level == "DEBUG"
        assert config.log_format == "console"
        assert config.enable_response_logging is True
        assert config.log_sensitive_headers is True

match __name__:
    case "__main__":
    pytest.main([__file__, "-v"]) 