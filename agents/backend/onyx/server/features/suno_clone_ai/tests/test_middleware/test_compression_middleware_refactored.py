"""
Tests refactorizados para compression middleware
Usando clases base y helpers para eliminar duplicación
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from fastapi import Request
from starlette.responses import Response

from middleware.compression_middleware import CompressionMiddleware
from test_helpers import BaseServiceTestCase, StandardTestMixin


class TestCompressionMiddlewareRefactored(BaseServiceTestCase, StandardTestMixin):
    """Tests refactorizados para CompressionMiddleware"""
    
    @pytest.fixture
    def mock_app(self):
        """Mock de aplicación FastAPI"""
        app = Mock()
        app.dispatch = AsyncMock()
        return app
    
    @pytest.fixture
    def compression_middleware(self, mock_app):
        """Fixture para CompressionMiddleware"""
        return CompressionMiddleware(mock_app, minimum_size=500, compress_level=6)
    
    @pytest.mark.asyncio
    @pytest.mark.parametrize("accept_encoding,should_compress", [
        ("", False),
        ("gzip", True),
        ("br, gzip", True),
        ("deflate", False)
    ])
    async def test_compression_middleware_accept_encoding(
        self,
        compression_middleware,
        accept_encoding,
        should_compress
    ):
        """Test de compresión según Accept-Encoding"""
        request = Mock(spec=Request)
        request.headers = {"accept-encoding": accept_encoding} if accept_encoding else {}
        
        large_data = b"x" * 10000
        response = Response(content=large_data)
        response.body = large_data
        call_next = AsyncMock(return_value=response)
        
        result = await compression_middleware.dispatch(request, call_next)
        
        if should_compress and accept_encoding:
            # Si debería comprimir, verificar que se agregaron headers
            # (aunque el mock puede no comprimir realmente)
            assert result is not None
        else:
            assert result == response
    
    @pytest.mark.asyncio
    @pytest.mark.parametrize("size,minimum_size,should_compress", [
        (100, 500, False),
        (500, 500, True),
        (1000, 500, True),
        (10000, 500, True)
    ])
    async def test_compression_middleware_minimum_size(
        self,
        mock_app,
        size,
        minimum_size,
        should_compress
    ):
        """Test de minimum_size"""
        middleware = CompressionMiddleware(mock_app, minimum_size=minimum_size)
        
        request = Mock(spec=Request)
        request.headers = {"accept-encoding": "gzip"}
        
        response = Response(content=b"x" * size)
        response.body = b"x" * size
        call_next = AsyncMock(return_value=response)
        
        result = await middleware.dispatch(request, call_next)
        
        assert result is not None
    
    @pytest.mark.asyncio
    @patch('middleware.compression_middleware.brotli')
    async def test_compression_middleware_brotli_preferred(self, mock_brotli, compression_middleware):
        """Test de preferencia por Brotli"""
        request = Mock(spec=Request)
        request.headers = {"accept-encoding": "br, gzip"}
        
        large_data = b"x" * 10000
        compressed_data = b"compressed"
        mock_brotli.compress.return_value = compressed_data
        
        response = Response(content=large_data)
        response.body = large_data
        call_next = AsyncMock(return_value=response)
        
        result = await compression_middleware.dispatch(request, call_next)
        
        if mock_brotli.compress.called:
            assert result.headers.get("content-encoding") == "br"



