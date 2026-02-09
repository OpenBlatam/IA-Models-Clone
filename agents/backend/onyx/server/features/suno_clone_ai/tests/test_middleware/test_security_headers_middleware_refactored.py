"""
Tests refactorizados para security headers middleware
Usando clases base y helpers para eliminar duplicación
"""

import pytest
from unittest.mock import Mock, AsyncMock
from fastapi import Request
from starlette.responses import Response

from middleware.security_headers_middleware import SecurityHeadersMiddleware
from test_helpers import BaseServiceTestCase, StandardTestMixin


class TestSecurityHeadersMiddlewareRefactored(BaseServiceTestCase, StandardTestMixin):
    """Tests refactorizados para SecurityHeadersMiddleware"""
    
    @pytest.fixture
    def mock_app(self):
        """Mock de aplicación FastAPI"""
        app = Mock()
        return app
    
    @pytest.fixture
    def security_middleware(self, mock_app):
        """Fixture para SecurityHeadersMiddleware"""
        return SecurityHeadersMiddleware(mock_app)
    
    @pytest.mark.asyncio
    async def test_security_headers_middleware_adds_all_headers(self, security_middleware):
        """Test de agregar todos los security headers"""
        request = Mock(spec=Request)
        request.headers = {}
        
        response = Response(content=b"test")
        call_next = AsyncMock(return_value=response)
        
        result = await security_middleware.dispatch(request, call_next)
        
        required_headers = [
            "Strict-Transport-Security",
            "Content-Security-Policy",
            "X-Content-Type-Options",
            "X-Frame-Options",
            "X-XSS-Protection",
            "Referrer-Policy",
            "Permissions-Policy",
            "X-Request-ID"
        ]
        
        for header in required_headers:
            assert header in result.headers, f"Missing header: {header}"
    
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
    @pytest.mark.parametrize("custom_hsts,custom_csp", [
        ("max-age=600", None),
        (None, "default-src 'self'"),
        ("max-age=600", "default-src 'self'")
    ])
    async def test_security_headers_middleware_custom_headers(
        self,
        mock_app,
        custom_hsts,
        custom_csp
    ):
        """Test con headers personalizados"""
        kwargs = {}
        if custom_hsts:
            kwargs["strict_transport_security"] = custom_hsts
        if custom_csp:
            kwargs["content_security_policy"] = custom_csp
        
        middleware = SecurityHeadersMiddleware(mock_app, **kwargs)
        
        request = Mock(spec=Request)
        request.headers = {}
        
        response = Response(content=b"test")
        call_next = AsyncMock(return_value=response)
        
        result = await middleware.dispatch(request, call_next)
        
        if custom_hsts:
            assert result.headers["Strict-Transport-Security"] == custom_hsts
        if custom_csp:
            assert result.headers["Content-Security-Policy"] == custom_csp



