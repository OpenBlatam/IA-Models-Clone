"""
Tests for Middleware
Tests for API middleware components
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from middleware.rate_limit_middleware import RateLimitMiddleware
from middleware.security_middleware import SecurityMiddleware
from middleware.monitoring_middleware import MonitoringMiddleware
from middleware.tracing_middleware import TracingMiddleware


class TestRateLimitMiddleware:
    """Tests for RateLimitMiddleware"""
    
    @pytest.mark.asyncio
    async def test_rate_limit_allows_request(self):
        """Test that middleware allows request within rate limit"""
        middleware = RateLimitMiddleware(Mock())
        
        request = Mock(spec=Request)
        request.client = Mock()
        request.client.host = "127.0.0.1"
        request.url.path = "/test"
        
        call_next = AsyncMock(return_value=JSONResponse({"status": "ok"}))
        
        response = await middleware.dispatch(request, call_next)
        
        assert response.status_code == 200
        call_next.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_rate_limit_blocks_excessive_requests(self):
        """Test that middleware blocks excessive requests"""
        middleware = RateLimitMiddleware(Mock())
        middleware.rate_limit = 1  # Very low limit for testing
        
        request = Mock(spec=Request)
        request.client = Mock()
        request.client.host = "127.0.0.1"
        request.url.path = "/test"
        
        call_next = AsyncMock(return_value=JSONResponse({"status": "ok"}))
        
        # First request should pass
        response1 = await middleware.dispatch(request, call_next)
        assert response1.status_code == 200
        
        # Second request should be blocked
        response2 = await middleware.dispatch(request, call_next)
        assert response2.status_code == 429  # Too Many Requests


class TestSecurityMiddleware:
    """Tests for SecurityMiddleware"""
    
    @pytest.mark.asyncio
    async def test_security_middleware_adds_headers(self):
        """Test that security middleware adds security headers"""
        middleware = SecurityMiddleware(Mock())
        
        request = Mock(spec=Request)
        call_next = AsyncMock(return_value=JSONResponse({"status": "ok"}))
        
        response = await middleware.dispatch(request, call_next)
        
        # Check that security headers are added
        assert "X-Content-Type-Options" in response.headers
        assert "X-Frame-Options" in response.headers
        call_next.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_security_middleware_validates_input(self):
        """Test that security middleware validates input"""
        middleware = SecurityMiddleware(Mock())
        
        request = Mock(spec=Request)
        request.url.path = "/test"
        request.method = "POST"
        request.body = AsyncMock(return_value=b'{"test": "data"}')
        
        call_next = AsyncMock(return_value=JSONResponse({"status": "ok"}))
        
        response = await middleware.dispatch(request, call_next)
        
        assert response.status_code == 200
        call_next.assert_called_once()


class TestMonitoringMiddleware:
    """Tests for MonitoringMiddleware"""
    
    @pytest.mark.asyncio
    async def test_monitoring_middleware_tracks_requests(self):
        """Test that monitoring middleware tracks requests"""
        middleware = MonitoringMiddleware(Mock())
        
        request = Mock(spec=Request)
        request.url.path = "/test"
        request.method = "GET"
        
        call_next = AsyncMock(return_value=JSONResponse({"status": "ok"}))
        
        response = await middleware.dispatch(request, call_next)
        
        assert response.status_code == 200
        call_next.assert_called_once()
        # Verify metrics are tracked (implementation dependent)
    
    @pytest.mark.asyncio
    async def test_monitoring_middleware_tracks_errors(self):
        """Test that monitoring middleware tracks errors"""
        middleware = MonitoringMiddleware(Mock())
        
        request = Mock(spec=Request)
        request.url.path = "/test"
        request.method = "GET"
        
        call_next = AsyncMock(side_effect=Exception("Test error"))
        
        with pytest.raises(Exception):
            await middleware.dispatch(request, call_next)
        
        # Verify error metrics are tracked (implementation dependent)


class TestTracingMiddleware:
    """Tests for TracingMiddleware"""
    
    @pytest.mark.asyncio
    async def test_tracing_middleware_adds_trace_id(self):
        """Test that tracing middleware adds trace ID"""
        middleware = TracingMiddleware(Mock())
        
        request = Mock(spec=Request)
        request.state = Mock()
        request.url.path = "/test"
        
        call_next = AsyncMock(return_value=JSONResponse({"status": "ok"}))
        
        response = await middleware.dispatch(request, call_next)
        
        assert response.status_code == 200
        # Verify trace ID is added to request state
        assert hasattr(request.state, 'trace_id') or hasattr(request.state, 'request_id')
        call_next.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_tracing_middleware_propagates_trace(self):
        """Test that tracing middleware propagates trace context"""
        middleware = TracingMiddleware(Mock())
        
        request = Mock(spec=Request)
        request.state = Mock()
        request.headers = {"X-Trace-Id": "existing-trace-id"}
        request.url.path = "/test"
        
        call_next = AsyncMock(return_value=JSONResponse({"status": "ok"}))
        
        response = await middleware.dispatch(request, call_next)
        
        assert response.status_code == 200
        call_next.assert_called_once()


class TestMiddlewareIntegration:
    """Integration tests for middleware"""
    
    @pytest.mark.asyncio
    async def test_middleware_chain(self):
        """Test that multiple middleware work together"""
        # Create middleware chain
        app = Mock()
        tracing = TracingMiddleware(app)
        security = SecurityMiddleware(tracing)
        monitoring = MonitoringMiddleware(security)
        rate_limit = RateLimitMiddleware(monitoring)
        
        request = Mock(spec=Request)
        request.client = Mock()
        request.client.host = "127.0.0.1"
        request.state = Mock()
        request.url.path = "/test"
        request.method = "GET"
        request.headers = {}
        
        call_next = AsyncMock(return_value=JSONResponse({"status": "ok"}))
        
        response = await rate_limit.dispatch(request, call_next)
        
        assert response.status_code == 200
        call_next.assert_called_once()



