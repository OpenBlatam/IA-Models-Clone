"""
Tests refactorizados para response cache middleware
Usando clases base y helpers para eliminar duplicación
"""

import pytest
import time
from unittest.mock import Mock, AsyncMock
from fastapi import Request
from starlette.responses import Response

from middleware.response_cache_middleware import ResponseCacheMiddleware
from test_helpers import BaseServiceTestCase, StandardTestMixin


class TestResponseCacheMiddlewareRefactored(BaseServiceTestCase, StandardTestMixin):
    """Tests refactorizados para ResponseCacheMiddleware"""
    
    @pytest.fixture
    def mock_app(self):
        """Mock de aplicación FastAPI"""
        app = Mock()
        return app
    
    @pytest.fixture
    def cache_middleware(self, mock_app):
        """Fixture para ResponseCacheMiddleware"""
        return ResponseCacheMiddleware(mock_app, ttl=60, max_size=100)
    
    def create_request(self, method="GET", path="/test", query=""):
        """Helper para crear requests"""
        request = Mock(spec=Request)
        request.method = method
        request.url.path = path
        request.url.query = query
        request.headers = {}
        return request
    
    @pytest.mark.asyncio
    @pytest.mark.parametrize("method,should_cache", [
        ("GET", True),
        ("POST", False),
        ("PUT", False),
        ("DELETE", False)
    ])
    async def test_cache_middleware_methods(self, cache_middleware, method, should_cache):
        """Test de cache según método HTTP"""
        request = self.create_request(method=method)
        
        response = Response(content=b"test response", status_code=200)
        call_next = AsyncMock(return_value=response)
        
        # Primera llamada
        result1 = await cache_middleware.dispatch(request, call_next)
        
        # Segunda llamada
        result2 = await cache_middleware.dispatch(request, call_next)
        
        if should_cache:
            assert call_next.call_count == 1  # Solo se llamó una vez
        else:
            assert call_next.call_count == 2  # Se llamó dos veces
    
    @pytest.mark.asyncio
    async def test_cache_middleware_cache_expires(self, mock_app):
        """Test de expiración de cache"""
        middleware = ResponseCacheMiddleware(mock_app, ttl=1, max_size=100)
        
        request = self.create_request()
        
        response = Response(content=b"test response", status_code=200)
        call_next = AsyncMock(return_value=response)
        
        # Primera llamada
        await middleware.dispatch(request, call_next)
        
        # Esperar a que expire
        time.sleep(1.1)
        
        # Segunda llamada (debería llamar de nuevo)
        await middleware.dispatch(request, call_next)
        
        assert call_next.call_count == 2
    
    @pytest.mark.asyncio
    async def test_cache_middleware_different_paths(self, cache_middleware):
        """Test de diferentes paths no comparten cache"""
        response1 = Response(content=b"response1", status_code=200)
        response2 = Response(content=b"response2", status_code=200)
        call_next = AsyncMock(side_effect=[response1, response2])
        
        request1 = self.create_request(path="/path1")
        request2 = self.create_request(path="/path2")
        
        result1 = await cache_middleware.dispatch(request1, call_next)
        result2 = await cache_middleware.dispatch(request2, call_next)
        
        assert call_next.call_count == 2
        assert result1 != result2



