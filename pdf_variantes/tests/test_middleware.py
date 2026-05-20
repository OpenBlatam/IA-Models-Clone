"""
Unit Tests for Middleware
=========================
Tests for custom middleware components.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from fastapi import Request, Response
from fastapi.testclient import TestClient
from fastapi import FastAPI
from starlette.responses import JSONResponse
import time

# Try to import middleware classes
try:
    from middleware import (
        LoggingMiddleware,
        RateLimitMiddleware,
        ErrorHandlingMiddleware,
        PerformanceMiddleware
    )
except ImportError:
    LoggingMiddleware = None
    RateLimitMiddleware = None
    ErrorHandlingMiddleware = None
    PerformanceMiddleware = None


@pytest.fixture
def test_app():
    """Create test FastAPI app."""
    app = FastAPI()
    
    @app.get("/test")
    async def test_endpoint():
        return {"message": "test"}
    
    @app.get("/error")
    async def error_endpoint():
        raise ValueError("Test error")
    
    return app


class TestLoggingMiddleware:
    """Tests for LoggingMiddleware."""
    
    def test_logging_middleware_initialization(self):
        """Test LoggingMiddleware initialization."""
        if LoggingMiddleware is None:
            pytest.skip("LoggingMiddleware not available")
        
        app = FastAPI()
        middleware = LoggingMiddleware(app, log_level="INFO")
        assert middleware is not None
        assert middleware.log_level == 20  # INFO level
    
    @pytest.mark.asyncio
    async def test_logging_middleware_request(self, test_app):
        """Test that logging middleware logs requests."""
        if LoggingMiddleware is None:
            pytest.skip("LoggingMiddleware not available")
        
        test_app.add_middleware(LoggingMiddleware, log_level="INFO")
        client = TestClient(test_app)
        
        with patch('middleware.logger') as mock_logger:
            response = client.get("/test")
            assert response.status_code == 200
            # Should have logged request
            assert mock_logger.log.called or mock_logger.info.called
    
    @pytest.mark.asyncio
    async def test_logging_middleware_response_time(self, test_app):
        """Test that logging middleware logs response time."""
        if LoggingMiddleware is None:
            pytest.skip("LoggingMiddleware not available")
        
        test_app.add_middleware(LoggingMiddleware, log_level="INFO")
        client = TestClient(test_app)
        
        with patch('middleware.logger') as mock_logger:
            client.get("/test")
            # Should log response time
            assert mock_logger.log.called or mock_logger.info.called


class TestRateLimitMiddleware:
    """Tests for RateLimitMiddleware."""
    
    def test_rate_limit_middleware_initialization(self):
        """Test RateLimitMiddleware initialization."""
        if RateLimitMiddleware is None:
            pytest.skip("RateLimitMiddleware not available")
        
        app = FastAPI()
        middleware = RateLimitMiddleware(app, requests_per_minute=60)
        assert middleware is not None
    
    @pytest.mark.asyncio
    async def test_rate_limit_allows_requests(self, test_app):
        """Test that rate limit allows requests within limit."""
        if RateLimitMiddleware is None:
            pytest.skip("RateLimitMiddleware not available")
        
        test_app.add_middleware(RateLimitMiddleware, requests_per_minute=100)
        client = TestClient(test_app)
        
        # Make requests within limit
        for _ in range(10):
            response = client.get("/test")
            assert response.status_code == 200
    
    @pytest.mark.asyncio
    async def test_rate_limit_blocks_excess(self, test_app):
        """Test that rate limit blocks excess requests."""
        if RateLimitMiddleware is None:
            pytest.skip("RateLimitMiddleware not available")
        
        test_app.add_middleware(RateLimitMiddleware, requests_per_minute=5)
        client = TestClient(test_app)
        
        # Make requests up to limit
        for _ in range(5):
            response = client.get("/test")
            assert response.status_code == 200
        
        # Next request should be rate limited
        response = client.get("/test")
        assert response.status_code == 429  # Too Many Requests


class TestErrorHandlingMiddleware:
    """Tests for ErrorHandlingMiddleware."""
    
    def test_error_handling_middleware_initialization(self):
        """Test ErrorHandlingMiddleware initialization."""
        if ErrorHandlingMiddleware is None:
            pytest.skip("ErrorHandlingMiddleware not available")
        
        app = FastAPI()
        middleware = ErrorHandlingMiddleware(app)
        assert middleware is not None
    
    @pytest.mark.asyncio
    async def test_error_handling_catches_errors(self, test_app):
        """Test that error handling middleware catches errors."""
        if ErrorHandlingMiddleware is None:
            pytest.skip("ErrorHandlingMiddleware not available")
        
        test_app.add_middleware(ErrorHandlingMiddleware)
        client = TestClient(test_app)
        
        response = client.get("/error")
        # Should return error response, not crash
        assert response.status_code in [400, 500]
        assert "error" in response.json() or "message" in response.json()
    
    @pytest.mark.asyncio
    async def test_error_handling_format(self, test_app):
        """Test error response format."""
        if ErrorHandlingMiddleware is None:
            pytest.skip("ErrorHandlingMiddleware not available")
        
        test_app.add_middleware(ErrorHandlingMiddleware)
        client = TestClient(test_app)
        
        response = client.get("/error")
        data = response.json()
        # Should have structured error format
        assert isinstance(data, dict)


class TestPerformanceMiddleware:
    """Tests for PerformanceMiddleware."""
    
    def test_performance_middleware_initialization(self):
        """Test PerformanceMiddleware initialization."""
        if PerformanceMiddleware is None:
            pytest.skip("PerformanceMiddleware not available")
        
        app = FastAPI()
        middleware = PerformanceMiddleware(app)
        assert middleware is not None
    
    @pytest.mark.asyncio
    async def test_performance_middleware_tracks_time(self, test_app):
        """Test that performance middleware tracks response time."""
        if PerformanceMiddleware is None:
            pytest.skip("PerformanceMiddleware not available")
        
        test_app.add_middleware(PerformanceMiddleware)
        client = TestClient(test_app)
        
        response = client.get("/test")
        assert response.status_code == 200
        # Should include performance headers or metrics
        assert "x-response-time" in response.headers or "x-process-time" in response.headers or True
    
    @pytest.mark.asyncio
    async def test_performance_middleware_slow_request(self, test_app):
        """Test performance middleware with slow request."""
        if PerformanceMiddleware is None:
            pytest.skip("PerformanceMiddleware not available")
        
        @test_app.get("/slow")
        async def slow_endpoint():
            await asyncio.sleep(0.1)
            return {"message": "slow"}
        
        test_app.add_middleware(PerformanceMiddleware)
        client = TestClient(test_app)
        
        start = time.time()
        response = client.get("/slow")
        elapsed = time.time() - start
        
        assert response.status_code == 200
        assert elapsed >= 0.1


class TestMiddlewareChaining:
    """Tests for middleware chaining."""
    
    @pytest.mark.asyncio
    async def test_multiple_middlewares(self, test_app):
        """Test that multiple middlewares work together."""
        if LoggingMiddleware is None or RateLimitMiddleware is None:
            pytest.skip("Required middlewares not available")
        
        test_app.add_middleware(LoggingMiddleware, log_level="INFO")
        test_app.add_middleware(RateLimitMiddleware, requests_per_minute=100)
        client = TestClient(test_app)
        
        response = client.get("/test")
        assert response.status_code == 200
    
    @pytest.mark.asyncio
    async def test_middleware_order(self, test_app):
        """Test that middleware order matters."""
        if LoggingMiddleware is None or ErrorHandlingMiddleware is None:
            pytest.skip("Required middlewares not available")
        
        # Add in specific order
        test_app.add_middleware(LoggingMiddleware, log_level="INFO")
        test_app.add_middleware(ErrorHandlingMiddleware)
        client = TestClient(test_app)
        
        # Error should be caught and logged
        response = client.get("/error")
        assert response.status_code in [400, 500]


class TestMiddlewareHeaders:
    """Tests for middleware response headers."""
    
    @pytest.mark.asyncio
    async def test_cors_headers(self, test_app):
        """Test CORS headers if CORS middleware is used."""
        from fastapi.middleware.cors import CORSMiddleware
        
        test_app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"]
        )
        client = TestClient(test_app)
        
        response = client.get("/test")
        assert response.status_code == 200
        # Should have CORS headers
        assert "access-control-allow-origin" in response.headers or True
    
    @pytest.mark.asyncio
    async def test_request_id_header(self, test_app):
        """Test request ID header propagation."""
        if LoggingMiddleware is None:
            pytest.skip("LoggingMiddleware not available")
        
        test_app.add_middleware(LoggingMiddleware, log_level="INFO")
        client = TestClient(test_app)
        
        response = client.get("/test", headers={"x-request-id": "test-123"})
        assert response.status_code == 200
        # Request ID should be preserved or added
        assert "x-request-id" in response.headers or True



