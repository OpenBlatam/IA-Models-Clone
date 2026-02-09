"""
Tests refactorizados para API decorators
Usando clases base y helpers para eliminar duplicación
"""

import pytest
from unittest.mock import Mock, patch

from core.api.api_decorators import (
    api_endpoint,
    require_auth,
    rate_limited
)
from test_helpers import BaseServiceTestCase, StandardTestMixin


class TestApiEndpointRefactored(BaseServiceTestCase, StandardTestMixin):
    """Tests refactorizados para api_endpoint decorator"""
    
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
        """Test de decorador con diferentes métodos personalizados"""
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


class TestRequireAuthRefactored(BaseServiceTestCase, StandardTestMixin):
    """Tests refactorizados para require_auth decorator"""
    
    @pytest.mark.parametrize("has_token,auth_func_result,should_succeed", [
        (True, None, True),
        (False, None, False),
        (False, True, True),
        (False, False, False)
    ])
    def test_require_auth(self, has_token, auth_func_result, should_succeed):
        """Test de autenticación con diferentes configuraciones"""
        auth_func = Mock(return_value=auth_func_result) if auth_func_result is not None else None
        
        @require_auth(auth_func=auth_func)
        def test_handler(request):
            return {"result": "success"}
        
        request = {
            'headers': {'token': 'valid_token'} if has_token else {}
        }
        
        result = test_handler(request)
        
        if should_succeed:
            assert result == {"result": "success"}
        else:
            assert result['success'] is False
            assert 'Unauthorized' in result['error']


class TestRateLimitedRefactored(BaseServiceTestCase, StandardTestMixin):
    """Tests refactorizados para rate_limited decorator"""
    
    @pytest.mark.parametrize("is_allowed,identifier_type", [
        (True, 'user_id'),
        (False, 'user_id'),
        (True, 'ip'),
        (False, 'ip')
    ])
    @patch('core.api.api_decorators.RateLimiter')
    def test_rate_limited(self, mock_rate_limiter_class, is_allowed, identifier_type):
        """Test de rate limit con diferentes configuraciones"""
        mock_limiter = Mock()
        mock_limiter.is_allowed.return_value = is_allowed
        mock_rate_limiter_class.return_value = mock_limiter
        
        @rate_limited(max_requests=100, time_window=60.0)
        def test_handler(request):
            return {"result": "success"}
        
        request = {identifier_type: 'test-123'} if identifier_type == 'user_id' else {'ip': '192.168.1.1'}
        result = test_handler(request)
        
        if is_allowed:
            assert result == {"result": "success"}
        else:
            assert result['success'] is False
            assert 'Rate limit exceeded' in result['error']



