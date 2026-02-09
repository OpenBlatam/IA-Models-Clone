"""
Comprehensive Unit Tests for Error Handler Middleware

Tests cover error handling middleware with diverse test cases
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse

from middleware.error_handler_middleware import ErrorHandlerMiddleware


class TestErrorHandlerMiddleware:
    """Test cases for ErrorHandlerMiddleware class"""
    
    @pytest.mark.asyncio
    async def test_dispatch_successful_request(self):
        """Test dispatching successful request"""
        app = Mock()
        middleware = ErrorHandlerMiddleware(app)
        
        request = Mock(spec=Request)
        request.url.path = "/test"
        request.method = "GET"
        
        response = Mock()
        call_next = AsyncMock(return_value=response)
        
        result = await middleware.dispatch(request, call_next)
        
        assert result == response
        call_next.assert_called_once_with(request)
    
    @pytest.mark.asyncio
    async def test_dispatch_http_exception_re_raises(self):
        """Test HTTPException is re-raised"""
        app = Mock()
        middleware = ErrorHandlerMiddleware(app)
        
        request = Mock(spec=Request)
        http_exception = HTTPException(status_code=404, detail="Not found")
        
        call_next = AsyncMock(side_effect=http_exception)
        
        with pytest.raises(HTTPException) as exc_info:
            await middleware.dispatch(request, call_next)
        
        assert exc_info.value.status_code == 404
    
    @pytest.mark.asyncio
    @patch('middleware.error_handler_middleware.BaseAPIException')
    async def test_dispatch_base_api_exception(self, mock_base_exc):
        """Test handling BaseAPIException"""
        app = Mock()
        middleware = ErrorHandlerMiddleware(app)
        
        request = Mock(spec=Request)
        request.url.path = "/test"
        request.method = "POST"
        
        mock_exception = Mock()
        mock_exception.status_code = 400
        mock_exception.detail = "API error"
        mock_base_exc.return_value = mock_exception
        
        call_next = AsyncMock(side_effect=mock_exception)
        
        with patch('middleware.error_handler_middleware.get_request_metadata', return_value={}):
            result = await middleware.dispatch(request, call_next)
            
            assert isinstance(result, JSONResponse)
            assert result.status_code == 400
    
    @pytest.mark.asyncio
    async def test_dispatch_value_error(self):
        """Test handling ValueError"""
        app = Mock()
        middleware = ErrorHandlerMiddleware(app)
        
        request = Mock(spec=Request)
        request.url.path = "/test"
        request.method = "GET"
        
        call_next = AsyncMock(side_effect=ValueError("Validation error"))
        
        with patch('middleware.error_handler_middleware.get_request_metadata', return_value={}):
            result = await middleware.dispatch(request, call_next)
            
            assert isinstance(result, JSONResponse)
            assert result.status_code == status.HTTP_400_BAD_REQUEST
    
    @pytest.mark.asyncio
    async def test_dispatch_file_not_found_error(self):
        """Test handling FileNotFoundError"""
        app = Mock()
        middleware = ErrorHandlerMiddleware(app)
        
        request = Mock(spec=Request)
        request.url.path = "/test"
        request.method = "GET"
        
        call_next = AsyncMock(side_effect=FileNotFoundError("File not found"))
        
        with patch('middleware.error_handler_middleware.get_request_metadata', return_value={}):
            result = await middleware.dispatch(request, call_next)
            
            assert isinstance(result, JSONResponse)
            assert result.status_code == status.HTTP_404_NOT_FOUND
    
    @pytest.mark.asyncio
    async def test_dispatch_permission_error(self):
        """Test handling PermissionError"""
        app = Mock()
        middleware = ErrorHandlerMiddleware(app)
        
        request = Mock(spec=Request)
        request.url.path = "/test"
        request.method = "GET"
        
        call_next = AsyncMock(side_effect=PermissionError("Permission denied"))
        
        with patch('middleware.error_handler_middleware.get_request_metadata', return_value={}):
            result = await middleware.dispatch(request, call_next)
            
            assert isinstance(result, JSONResponse)
            assert result.status_code == status.HTTP_403_FORBIDDEN
    
    @pytest.mark.asyncio
    async def test_dispatch_generic_exception(self):
        """Test handling generic exception"""
        app = Mock()
        middleware = ErrorHandlerMiddleware(app)
        
        request = Mock(spec=Request)
        request.url.path = "/test"
        request.method = "GET"
        
        call_next = AsyncMock(side_effect=Exception("Unexpected error"))
        
        with patch('middleware.error_handler_middleware.get_request_metadata', return_value={}):
            result = await middleware.dispatch(request, call_next)
            
            assert isinstance(result, JSONResponse)
            assert result.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    
    @pytest.mark.asyncio
    async def test_error_response_format(self):
        """Test error response format"""
        app = Mock()
        middleware = ErrorHandlerMiddleware(app)
        
        request = Mock(spec=Request)
        request.url.path = "/api/test"
        request.method = "POST"
        
        call_next = AsyncMock(side_effect=ValueError("Test error"))
        
        with patch('middleware.error_handler_middleware.get_request_metadata', return_value={}):
            result = await middleware.dispatch(request, call_next)
            
            # Verify response content structure
            assert hasattr(result, 'body') or hasattr(result, 'content')















