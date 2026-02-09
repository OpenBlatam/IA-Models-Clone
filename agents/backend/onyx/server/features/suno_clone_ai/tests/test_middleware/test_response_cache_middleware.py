"""
Tests para response cache middleware
"""

import pytest
import time
from unittest.mock import Mock, AsyncMock
from fastapi import Request
from starlette.responses import Response

from middleware.response_cache_middleware import ResponseCacheMiddleware


@pytest.fixture
def mock_app():
    """Mock de aplicación FastAPI"""
    app = Mock()
    return app


@pytest.fixture
def cache_middleware(mock_app):
    """Fixture para ResponseCacheMiddleware"""
    return ResponseCacheMiddleware(mock_app, ttl=60, max_size=100)


@pytest.mark.unit
@pytest.mark.middleware
class TestResponseCacheMiddleware:
    """Tests para ResponseCacheMiddleware"""
    
    @pytest.mark.asyncio
    async def test_cache_middleware_caches_get_requests(self, cache_middleware):
        """Test de cachear GET requests"""
        request = Mock(spec=Request)
        request.method = "GET"
        request.url.path = "/test"
        request.url.query = ""
        request.headers = {}
        
        response = Response(content=b"test response", status_code=200)
        call_next = AsyncMock(return_value=response)
        
        # Primera llamada
        result1 = await cache_middleware.dispatch(request, call_next)
        
        # Segunda llamada (debería usar cache)
        result2 = await cache_middleware.dispatch(request, call_next)
        
        assert call_next.call_count == 1  # Solo se llamó una vez
        assert result1 == result2
    
    @pytest.mark.asyncio
    async def test_cache_middleware_no_cache_post(self, cache_middleware):
        """Test de no cachear POST requests"""
        request = Mock(spec=Request)
        request.method = "POST"
        request.url.path = "/test"
        request.url.query = ""
        request.headers = {}
        
        response = Response(content=b"test response")
        call_next = AsyncMock(return_value=response)
        
        result = await cache_middleware.dispatch(request, call_next)
        
        assert result == response
        assert call_next.call_count == 1
    
    @pytest.mark.asyncio
    async def test_cache_middleware_cache_expires(self, mock_app):
        """Test de expiración de cache"""
        middleware = ResponseCacheMiddleware(mock_app, ttl=1, max_size=100)
        
        request = Mock(spec=Request)
        request.method = "GET"
        request.url.path = "/test"
        request.url.query = ""
        request.headers = {}
        
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
        
        request1 = Mock(spec=Request)
        request1.method = "GET"
        request1.url.path = "/path1"
        request1.url.query = ""
        request1.headers = {}
        
        request2 = Mock(spec=Request)
        request2.method = "GET"
        request2.url.path = "/path2"
        request2.url.query = ""
        request2.headers = {}
        
        result1 = await cache_middleware.dispatch(request1, call_next)
        result2 = await cache_middleware.dispatch(request2, call_next)
        
        assert call_next.call_count == 2
        assert result1 != result2



