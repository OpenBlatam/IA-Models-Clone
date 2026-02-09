"""
Tests para logging middleware
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from fastapi import Request
from starlette.responses import Response

from middleware.logging_middleware import LoggingMiddleware


@pytest.fixture
def mock_app():
    """Mock de aplicación FastAPI"""
    app = Mock()
    return app


@pytest.fixture
def logging_middleware(mock_app):
    """Fixture para LoggingMiddleware"""
    return LoggingMiddleware(mock_app)


@pytest.mark.unit
@pytest.mark.middleware
class TestLoggingMiddleware:
    """Tests para LoggingMiddleware"""
    
    @pytest.mark.asyncio
    @patch('middleware.logging_middleware.logger')
    async def test_logging_middleware_logs_request(self, mock_logger, logging_middleware):
        """Test de logging de request"""
        request = Mock(spec=Request)
        request.method = "GET"
        request.url.path = "/test"
        
        response = Response(content=b"test")
        call_next = AsyncMock(return_value=response)
        
        result = await logging_middleware.dispatch(request, call_next)
        
        assert result == response
        mock_logger.info.assert_called()
    
    @pytest.mark.asyncio
    @patch('middleware.logging_middleware.logger')
    async def test_logging_middleware_logs_response(self, mock_logger, logging_middleware):
        """Test de logging de response"""
        request = Mock(spec=Request)
        request.method = "GET"
        request.url.path = "/test"
        
        response = Response(content=b"test", status_code=200)
        call_next = AsyncMock(return_value=response)
        
        result = await logging_middleware.dispatch(request, call_next)
        
        assert result == response
        # Verificar que se llamó logger.info múltiples veces (request + response)
        assert mock_logger.info.call_count >= 2
    
    @pytest.mark.asyncio
    @patch('middleware.logging_middleware.logger')
    async def test_logging_middleware_logs_duration(self, mock_logger, logging_middleware):
        """Test de logging de duración"""
        import asyncio
        
        request = Mock(spec=Request)
        request.method = "GET"
        request.url.path = "/test"
        
        async def slow_call_next(request):
            await asyncio.sleep(0.01)
            return Response(content=b"test")
        
        result = await logging_middleware.dispatch(request, slow_call_next)
        
        assert result is not None
        # Verificar que se logueó la duración
        call_args = mock_logger.info.call_args_list
        assert any("Duration" in str(call) for call in call_args)
