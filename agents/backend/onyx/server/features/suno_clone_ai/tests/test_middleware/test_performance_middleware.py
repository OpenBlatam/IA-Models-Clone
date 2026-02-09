"""
Tests para performance middleware
"""

import pytest
import time
from unittest.mock import Mock, AsyncMock
from fastapi import Request
from starlette.responses import Response

from middleware.performance_middleware import PerformanceMiddleware


@pytest.fixture
def mock_app():
    """Mock de aplicación FastAPI"""
    app = Mock()
    return app


@pytest.fixture
def performance_middleware(mock_app):
    """Fixture para PerformanceMiddleware"""
    return PerformanceMiddleware(mock_app, enable_caching=True, cache_ttl=60)


@pytest.mark.unit
@pytest.mark.middleware
class TestPerformanceMiddleware:
    """Tests para PerformanceMiddleware"""
    
    @pytest.mark.asyncio
    async def test_performance_middleware_adds_headers(self, performance_middleware):
        """Test de agregar headers de performance"""
        request = Mock(spec=Request)
        request.method = "GET"
        request.url.path = "/test"
        request.url.query = ""
        request.headers = {}
        
        response = Response(content=b"test")
        call_next = AsyncMock(return_value=response)
        
        result = await performance_middleware.dispatch(request, call_next)
        
        assert result.headers.get("X-Content-Type-Options") == "nosniff"
        assert result.headers.get("Connection") == "keep-alive"
    
    @pytest.mark.asyncio
    async def test_performance_middleware_caches_get(self, performance_middleware):
        """Test de cachear GET requests"""
        request = Mock(spec=Request)
        request.method = "GET"
        request.url.path = "/test"
        request.url.query = ""
        request.headers = {}
        
        response = Response(content=b"test", status_code=200)
        call_next = AsyncMock(return_value=response)
        
        # Primera llamada
        result1 = await performance_middleware.dispatch(request, call_next)
        
        # Segunda llamada (debería usar cache)
        result2 = await performance_middleware.dispatch(request, call_next)
        
        assert call_next.call_count == 1
    
    @pytest.mark.asyncio
    async def test_performance_middleware_no_cache_post(self, performance_middleware):
        """Test de no cachear POST requests"""
        request = Mock(spec=Request)
        request.method = "POST"
        request.url.path = "/test"
        request.url.query = ""
        request.headers = {}
        
        response = Response(content=b"test")
        call_next = AsyncMock(return_value=response)
        
        result = await performance_middleware.dispatch(request, call_next)
        
        assert result == response
        assert call_next.call_count == 1
    
    @pytest.mark.asyncio
    async def test_performance_middleware_cache_disabled(self, mock_app):
        """Test con cache deshabilitado"""
        middleware = PerformanceMiddleware(mock_app, enable_caching=False)
        
        request = Mock(spec=Request)
        request.method = "GET"
        request.url.path = "/test"
        request.url.query = ""
        request.headers = {}
        
        response = Response(content=b"test", status_code=200)
        call_next = AsyncMock(return_value=response)
        
        # Múltiples llamadas
        await middleware.dispatch(request, call_next)
        await middleware.dispatch(request, call_next)
        
        assert call_next.call_count == 2  # No debería usar cache
    
    @pytest.mark.asyncio
    async def test_performance_middleware_cache_expires(self, mock_app):
        """Test de expiración de cache"""
        middleware = PerformanceMiddleware(mock_app, enable_caching=True, cache_ttl=1)
        
        request = Mock(spec=Request)
        request.method = "GET"
        request.url.path = "/test"
        request.url.query = ""
        request.headers = {}
        
        response = Response(content=b"test", status_code=200)
        call_next = AsyncMock(return_value=response)
        
        # Primera llamada
        await middleware.dispatch(request, call_next)
        
        # Esperar a que expire
        time.sleep(1.1)
        
        # Segunda llamada
        await middleware.dispatch(request, call_next)
        
        assert call_next.call_count == 2



