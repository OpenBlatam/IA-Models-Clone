"""
Comprehensive Unit Tests for Rate Limiter Middleware

Tests cover rate limiting middleware with diverse test cases
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from fastapi import Request, HTTPException, status
from starlette.responses import Response

from middleware.rate_limiter import RateLimiterMiddleware


class TestRateLimiterMiddleware:
    """Test cases for RateLimiterMiddleware class"""
    
    def test_rate_limiter_middleware_init_defaults(self):
        """Test initializing middleware with defaults"""
        app = Mock()
        middleware = RateLimiterMiddleware(app)
        
        assert middleware.requests_per_minute is not None
        assert middleware.window_seconds is not None
    
    def test_rate_limiter_middleware_init_custom(self):
        """Test initializing middleware with custom values"""
        app = Mock()
        middleware = RateLimiterMiddleware(
            app,
            requests_per_minute=100,
            window_seconds=60
        )
        
        assert middleware.requests_per_minute == 100
        assert middleware.window_seconds == 60
    
    @pytest.mark.asyncio
    @patch('middleware.rate_limiter.check_rate_limit')
    @patch('middleware.rate_limiter.get_rate_limit_info')
    async def test_dispatch_allowed_request(self, mock_get_info, mock_check):
        """Test dispatching allowed request"""
        app = Mock()
        middleware = RateLimiterMiddleware(app, requests_per_minute=10, window_seconds=60)
        
        mock_check.return_value = (True, None)
        mock_get_info.return_value = {
            "limit": 10,
            "remaining": 9,
            "reset_in": 60
        }
        
        request = Mock(spec=Request)
        request.headers = {}
        request.query_params = {}
        
        response = Mock(spec=Response)
        response.headers = {}
        call_next = AsyncMock(return_value=response)
        
        result = await middleware.dispatch(request, call_next)
        
        assert result == response
        call_next.assert_called_once()
        assert "X-RateLimit-Limit" in response.headers
    
    @pytest.mark.asyncio
    @patch('middleware.rate_limiter.check_rate_limit')
    async def test_dispatch_rate_limit_exceeded(self, mock_check):
        """Test dispatching request when rate limit exceeded"""
        app = Mock()
        middleware = RateLimiterMiddleware(app, requests_per_minute=10, window_seconds=60)
        
        mock_check.return_value = (False, 30)  # Not allowed, retry after 30s
        
        request = Mock(spec=Request)
        request.headers = {}
        request.query_params = {}
        
        call_next = AsyncMock()
        
        with pytest.raises(HTTPException) as exc_info:
            await middleware.dispatch(request, call_next)
        
        assert exc_info.value.status_code == status.HTTP_429_TOO_MANY_REQUESTS
        assert "Retry-After" in exc_info.value.headers
    
    @pytest.mark.asyncio
    @patch('middleware.rate_limiter.get_client_ip')
    @patch('middleware.rate_limiter.check_rate_limit')
    async def test_get_client_id_from_ip(self, mock_check, mock_get_ip):
        """Test getting client ID from IP"""
        app = Mock()
        middleware = RateLimiterMiddleware(app)
        
        mock_check.return_value = (True, None)
        mock_get_ip.return_value = "192.168.1.1"
        
        request = Mock(spec=Request)
        request.headers = {}
        request.query_params = {}
        
        response = Mock(spec=Response)
        response.headers = {}
        call_next = AsyncMock(return_value=response)
        
        await middleware.dispatch(request, call_next)
        
        # Verify IP was used
        call_args = mock_check.call_args[0]
        assert call_args[0].startswith("ip:")
    
    @pytest.mark.asyncio
    @patch('middleware.rate_limiter.check_rate_limit')
    async def test_get_client_id_from_user_id_header(self, mock_check):
        """Test getting client ID from user ID header"""
        app = Mock()
        middleware = RateLimiterMiddleware(app)
        
        mock_check.return_value = (True, None)
        
        request = Mock(spec=Request)
        request.headers = {"X-User-ID": "user123"}
        request.query_params = {}
        
        response = Mock(spec=Response)
        response.headers = {}
        call_next = AsyncMock(return_value=response)
        
        await middleware.dispatch(request, call_next)
        
        # Verify user ID was used
        call_args = mock_check.call_args[0]
        assert call_args[0] == "user:user123"
    
    @pytest.mark.asyncio
    @patch('middleware.rate_limiter.check_rate_limit')
    async def test_get_client_id_from_query_param(self, mock_check):
        """Test getting client ID from query parameter"""
        app = Mock()
        middleware = RateLimiterMiddleware(app)
        
        mock_check.return_value = (True, None)
        
        request = Mock(spec=Request)
        request.headers = {}
        request.query_params = {"user_id": "user456"}
        
        response = Mock(spec=Response)
        response.headers = {}
        call_next = AsyncMock(return_value=response)
        
        await middleware.dispatch(request, call_next)
        
        # Verify user ID from query was used
        call_args = mock_check.call_args[0]
        assert call_args[0] == "user:user456"
    
    @pytest.mark.asyncio
    @patch('middleware.rate_limiter.check_rate_limit')
    @patch('middleware.rate_limiter.get_rate_limit_info')
    async def test_response_headers_added(self, mock_get_info, mock_check):
        """Test rate limit headers are added to response"""
        app = Mock()
        middleware = RateLimiterMiddleware(app, requests_per_minute=10, window_seconds=60)
        
        mock_check.return_value = (True, None)
        mock_get_info.return_value = {
            "limit": 10,
            "remaining": 5,
            "reset_in": 30
        }
        
        request = Mock(spec=Request)
        request.headers = {}
        request.query_params = {}
        
        response = Mock(spec=Response)
        response.headers = {}
        call_next = AsyncMock(return_value=response)
        
        await middleware.dispatch(request, call_next)
        
        assert response.headers["X-RateLimit-Limit"] == "10"
        assert response.headers["X-RateLimit-Remaining"] == "5"
        assert response.headers["X-RateLimit-Reset"] == "30"
    
    @pytest.mark.asyncio
    @patch('middleware.rate_limiter.check_rate_limit')
    async def test_retry_after_header(self, mock_check):
        """Test Retry-After header in error response"""
        app = Mock()
        middleware = RateLimiterMiddleware(app)
        
        mock_check.return_value = (False, 45)  # Retry after 45 seconds
        
        request = Mock(spec=Request)
        request.headers = {}
        request.query_params = {}
        
        call_next = AsyncMock()
        
        with pytest.raises(HTTPException) as exc_info:
            await middleware.dispatch(request, call_next)
        
        assert exc_info.value.headers["Retry-After"] == "45"
    
    @pytest.mark.asyncio
    @patch('middleware.rate_limiter.check_rate_limit')
    async def test_retry_after_default(self, mock_check):
        """Test default Retry-After when not provided"""
        app = Mock()
        middleware = RateLimiterMiddleware(app, window_seconds=60)
        
        mock_check.return_value = (False, None)  # No retry_after provided
        
        request = Mock(spec=Request)
        request.headers = {}
        request.query_params = {}
        
        call_next = AsyncMock()
        
        with pytest.raises(HTTPException) as exc_info:
            await middleware.dispatch(request, call_next)
        
        assert exc_info.value.headers["Retry-After"] == "60"  # Default to window_seconds















