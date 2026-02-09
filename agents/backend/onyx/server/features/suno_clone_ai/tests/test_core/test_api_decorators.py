"""
Tests para API decorators
"""

import pytest
from unittest.mock import Mock, patch

from core.api.api_decorators import (
    api_endpoint,
    require_auth,
    rate_limited
)
from test_helpers import BaseServiceTestCase, StandardTestMixin


class TestApiEndpoint(BaseServiceTestCase, StandardTestMixin):
    """Tests para api_endpoint decorator"""
    
    def test_api_endpoint_default_methods(self):
        """Test de decorador con métodos por defecto"""
        @api_endpoint("/test")
        def test_handler():
            return {"result": "success"}
        
        assert hasattr(test_handler, 'path')
        assert test_handler.path == "/test"
        assert hasattr(test_handler, 'methods')
        assert 'GET' in test_handler.methods
        assert 'POST' in test_handler.methods
    
    @pytest.mark.parametrize("methods", [
        ['GET'],
        ['POST'],
        ['PUT'],
        ['DELETE'],
        ['GET', 'POST', 'PUT']
    ])
    def test_api_endpoint_custom_methods(self, methods):
        """Test de decorador con métodos personalizados"""
        @api_endpoint("/test", methods=methods)
        def test_handler():
            return {"result": "success"}
        
        assert test_handler.methods == methods
    
    def test_api_endpoint_execution(self):
        """Test de ejecución del handler decorado"""
        @api_endpoint("/test")
        def test_handler():
            return {"result": "success"}
        
        result = test_handler()
        
        assert result == {"result": "success"}


class TestRequireAuth(BaseServiceTestCase, StandardTestMixin):
    """Tests para require_auth decorator"""
    
    def test_require_auth_with_token(self):
        """Test de autenticación con token"""
        @require_auth()
        def test_handler(request):
            return {"result": "success"}
        
        request = {
            'headers': {'token': 'valid_token'}
        }
        
        result = test_handler(request)
        
        assert result == {"result": "success"}
    
    def test_require_auth_without_token(self):
        """Test de autenticación sin token"""
        @require_auth()
        def test_handler(request):
            return {"result": "success"}
        
        request = {
            'headers': {}
        }
        
        result = test_handler(request)
        
        assert result['success'] is False
        assert 'Unauthorized' in result['error']
    
    def test_require_auth_with_custom_func(self):
        """Test de autenticación con función personalizada"""
        auth_func = Mock(return_value=True)
        
        @require_auth(auth_func=auth_func)
        def test_handler(request):
            return {"result": "success"}
        
        request = {}
        result = test_handler(request)
        
        assert result == {"result": "success"}
        auth_func.assert_called_once_with(request)
    
    def test_require_auth_with_custom_func_failed(self):
        """Test de autenticación con función personalizada que falla"""
        auth_func = Mock(return_value=False)
        
        @require_auth(auth_func=auth_func)
        def test_handler(request):
            return {"result": "success"}
        
        request = {}
        result = test_handler(request)
        
        assert result['success'] is False
        assert 'Unauthorized' in result['error']


class TestRateLimited(BaseServiceTestCase, StandardTestMixin):
    """Tests para rate_limited decorator"""
    
    @patch('core.api.api_decorators.RateLimiter')
    def test_rate_limited_allowed(self, mock_rate_limiter_class):
        """Test cuando el rate limit permite la request"""
        mock_limiter = Mock()
        mock_limiter.is_allowed.return_value = True
        mock_rate_limiter_class.return_value = mock_limiter
        
        @rate_limited(max_requests=100, time_window=60.0)
        def test_handler(request):
            return {"result": "success"}
        
        request = {'user_id': 'user-123'}
        result = test_handler(request)
        
        assert result == {"result": "success"}
    
    @patch('core.api.api_decorators.RateLimiter')
    def test_rate_limited_exceeded(self, mock_rate_limiter_class):
        """Test cuando se excede el rate limit"""
        mock_limiter = Mock()
        mock_limiter.is_allowed.return_value = False
        mock_rate_limiter_class.return_value = mock_limiter
        
        @rate_limited(max_requests=100, time_window=60.0)
        def test_handler(request):
            return {"result": "success"}
        
        request = {'user_id': 'user-123'}
        result = test_handler(request)
        
        assert result['success'] is False
        assert 'Rate limit exceeded' in result['error']
    
    @patch('core.api.api_decorators.RateLimiter')
    def test_rate_limited_with_ip(self, mock_rate_limiter_class):
        """Test de rate limit usando IP cuando no hay user_id"""
        mock_limiter = Mock()
        mock_limiter.is_allowed.return_value = True
        mock_rate_limiter_class.return_value = mock_limiter
        
        @rate_limited(max_requests=100, time_window=60.0)
        def test_handler(request):
            return {"result": "success"}
        
        request = {'ip': '192.168.1.1'}
        result = test_handler(request)
        
        assert result == {"result": "success"}



