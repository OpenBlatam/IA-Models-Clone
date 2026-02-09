"""
Tests para compression middleware
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from starlette.responses import Response

from middleware.compression_middleware import CompressionMiddleware


@pytest.fixture
def mock_app():
    """Mock de aplicación FastAPI"""
    app = Mock()
    app.dispatch = AsyncMock()
    return app


@pytest.fixture
def compression_middleware(mock_app):
    """Fixture para CompressionMiddleware"""
    return CompressionMiddleware(mock_app, minimum_size=500, compress_level=6)


@pytest.mark.unit
@pytest.mark.middleware
class TestCompressionMiddleware:
    """Tests para CompressionMiddleware"""
    
    @pytest.mark.asyncio
    async def test_compression_middleware_no_accept_encoding(self, compression_middleware):
        """Test sin Accept-Encoding header"""
        request = Mock(spec=Request)
        request.headers = {}
        
        response = Response(content=b"test data")
        call_next = AsyncMock(return_value=response)
        
        result = await compression_middleware.dispatch(request, call_next)
        
        assert result == response
        assert "content-encoding" not in result.headers
    
    @pytest.mark.asyncio
    async def test_compression_middleware_small_response(self, compression_middleware):
        """Test con respuesta pequeña (menor que minimum_size)"""
        request = Mock(spec=Request)
        request.headers = {"accept-encoding": "gzip"}
        
        response = Response(content=b"x" * 100)  # Menor que 500
        call_next = AsyncMock(return_value=response)
        
        result = await compression_middleware.dispatch(request, call_next)
        
        # No debería comprimir si es muy pequeño
        assert result == response
    
    @pytest.mark.asyncio
    @patch('middleware.compression_middleware.brotli')
    async def test_compression_middleware_brotli(self, mock_brotli, compression_middleware):
        """Test de compresión Brotli"""
        request = Mock(spec=Request)
        request.headers = {"accept-encoding": "br, gzip"}
        
        large_data = b"x" * 10000
        compressed_data = b"compressed"
        mock_brotli.compress.return_value = compressed_data
        
        response = Response(content=large_data)
        response.body = large_data
        call_next = AsyncMock(return_value=response)
        
        result = await compression_middleware.dispatch(request, call_next)
        
        assert result.headers.get("content-encoding") == "br"
        assert result.headers.get("vary") == "Accept-Encoding"
    
    @pytest.mark.asyncio
    @patch('middleware.compression_middleware.gzip')
    async def test_compression_middleware_gzip(self, mock_gzip, compression_middleware):
        """Test de compresión Gzip"""
        request = Mock(spec=Request)
        request.headers = {"accept-encoding": "gzip"}
        
        large_data = b"x" * 10000
        compressed_data = b"compressed"
        mock_gzip.compress.return_value = compressed_data
        
        response = Response(content=large_data)
        response.body = large_data
        call_next = AsyncMock(return_value=response)
        
        result = await compression_middleware.dispatch(request, call_next)
        
        assert result.headers.get("content-encoding") == "gzip"
        assert result.headers.get("vary") == "Accept-Encoding"
    
    @pytest.mark.asyncio
    async def test_compression_middleware_custom_minimum_size(self, mock_app):
        """Test con minimum_size personalizado"""
        middleware = CompressionMiddleware(mock_app, minimum_size=1000)
        
        request = Mock(spec=Request)
        request.headers = {"accept-encoding": "gzip"}
        
        response = Response(content=b"x" * 500)  # Menor que 1000
        call_next = AsyncMock(return_value=response)
        
        result = await middleware.dispatch(request, call_next)
        
        # No debería comprimir
        assert result == response
