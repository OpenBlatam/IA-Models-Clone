"""
Tests para security headers middleware
"""

import pytest
from unittest.mock import Mock, AsyncMock
from fastapi import Request
from starlette.responses import Response

from middleware.security_headers_middleware import SecurityHeadersMiddleware


@pytest.fixture
def mock_app():
    """Mock de aplicación FastAPI"""
    app = Mock()
    return app


@pytest.fixture
def security_middleware(mock_app):
    """Fixture para SecurityHeadersMiddleware"""
    return SecurityHeadersMiddleware(mock_app)


@pytest.mark.unit
@pytest.mark.middleware
class TestSecurityHeadersMiddleware:
    """Tests para SecurityHeadersMiddleware"""
    
    @pytest.mark.asyncio
    async def test_security_headers_middleware_adds_headers(self, security_middleware):
        """Test de agregar security headers"""
        request = Mock(spec=Request)
        request.headers = {}
        
        response = Response(content=b"test")
        call_next = AsyncMock(return_value=response)
        
        result = await security_middleware.dispatch(request, call_next)
        
        assert "Strict-Transport-Security" in result.headers
        assert "Content-Security-Policy" in result.headers
        assert "X-Content-Type-Options" in result.headers
        assert "X-Frame-Options" in result.headers
        assert "X-XSS-Protection" in result.headers
        assert "Referrer-Policy" in result.headers
        assert "Permissions-Policy" in result.headers
    
    @pytest.mark.asyncio
    async def test_security_headers_middleware_removes_server_header(self, security_middleware):
        """Test de remover server header"""
        request = Mock(spec=Request)
        request.headers = {}
        
        response = Response(content=b"test")
        response.headers["server"] = "nginx"
        call_next = AsyncMock(return_value=response)
        
        result = await security_middleware.dispatch(request, call_next)
        
        assert "server" not in result.headers
    
    @pytest.mark.asyncio
    async def test_security_headers_middleware_adds_request_id(self, security_middleware):
        """Test de agregar X-Request-ID"""
        request = Mock(spec=Request)
        request.headers = {}
        request_id = "test-request-id"
        
        response = Response(content=b"test")
        call_next = AsyncMock(return_value=response)
        
        result = await security_middleware.dispatch(request, call_next)
        
        assert "X-Request-ID" in result.headers
    
    @pytest.mark.asyncio
    async def test_security_headers_middleware_custom_headers(self, mock_app):
        """Test con headers personalizados"""
        middleware = SecurityHeadersMiddleware(
            mock_app,
            strict_transport_security="max-age=600",
            x_frame_options="SAMEORIGIN"
        )
        
        request = Mock(spec=Request)
        request.headers = {}
        
        response = Response(content=b"test")
        call_next = AsyncMock(return_value=response)
        
        result = await middleware.dispatch(request, call_next)
        
        assert result.headers["Strict-Transport-Security"] == "max-age=600"
        assert result.headers["X-Frame-Options"] == "SAMEORIGIN"
