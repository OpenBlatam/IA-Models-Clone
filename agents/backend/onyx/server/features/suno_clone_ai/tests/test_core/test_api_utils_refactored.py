"""
Tests refactorizados para API utils
Usando clases base y helpers para eliminar duplicación
"""

import pytest
from unittest.mock import Mock

from core.api.api_utils import (
    APIHandler,
    create_api_handler,
    validate_request
)
from test_helpers import BaseServiceTestCase, StandardTestMixin


class TestAPIHandlerRefactored(BaseServiceTestCase, StandardTestMixin):
    """Tests refactorizados para APIHandler"""
    
    @pytest.fixture
    def api_handler(self):
        """Fixture para APIHandler"""
        return APIHandler()
    
    def test_api_handler_init(self, api_handler):
        """Test de inicialización"""
        assert api_handler.middlewares == []
    
    def test_add_middleware(self, api_handler):
        """Test de agregar middleware"""
        middleware = Mock()
        
        result = api_handler.add_middleware(middleware)
        
        assert result is api_handler  # Chaining
        assert middleware in api_handler.middlewares
    
    @pytest.mark.parametrize("num_middlewares", [1, 2, 3, 5])
    def test_add_multiple_middlewares(self, api_handler, num_middlewares):
        """Test de agregar múltiples middlewares"""
        middlewares = [Mock() for _ in range(num_middlewares)]
        
        for middleware in middlewares:
            api_handler.add_middleware(middleware)
        
        assert len(api_handler.middlewares) == num_middlewares
    
    def test_handle_success(self, api_handler):
        """Test de manejo exitoso de request"""
        handler = Mock(return_value={"data": "test"})
        request = {}
        
        result = api_handler.handle(request, handler)
        
        assert result['success'] is True
        assert result['data'] == {"data": "test"}
        handler.assert_called_once_with(request)
    
    def test_handle_with_middleware(self, api_handler):
        """Test de manejo con middleware"""
        middleware = Mock(return_value={'modified': True})
        handler = Mock(return_value={"data": "test"})
        request = {}
        
        api_handler.add_middleware(middleware)
        result = api_handler.handle(request, handler)
        
        assert result['success'] is True
        middleware.assert_called_once_with(request)
        handler.assert_called_once_with({'modified': True})
    
    def test_handle_error(self, api_handler):
        """Test de manejo de error"""
        handler = Mock(side_effect=Exception("Test error"))
        request = {}
        
        result = api_handler.handle(request, handler)
        
        assert result['success'] is False
        assert 'error' in result
        assert 'Test error' in result['error']


class TestCreateApiHandlerRefactored(BaseServiceTestCase, StandardTestMixin):
    """Tests refactorizados para create_api_handler"""
    
    def test_create_api_handler(self):
        """Test de creación de API handler"""
        handler = create_api_handler()
        
        assert isinstance(handler, APIHandler)


class TestValidateRequestRefactored(BaseServiceTestCase, StandardTestMixin):
    """Tests refactorizados para validate_request"""
    
    @pytest.mark.parametrize("required_fields", [
        ['field1'],
        ['field1', 'field2'],
        ['field1', 'field2', 'field3']
    ])
    def test_validate_request_success(self, required_fields):
        """Test de validación exitosa con diferentes campos"""
        request = {field: f'value_{field}' for field in required_fields}
        
        is_valid, error = validate_request(request, required_fields)
        
        assert is_valid is True
        assert error is None
    
    @pytest.mark.parametrize("missing_field", ['field1', 'field2', 'field3'])
    def test_validate_request_missing_field(self, missing_field):
        """Test de validación con diferentes campos faltantes"""
        required_fields = ['field1', 'field2', 'field3']
        request = {field: f'value_{field}' for field in required_fields if field != missing_field}
        
        is_valid, error = validate_request(request, required_fields)
        
        assert is_valid is False
        assert f'Missing required field: {missing_field}' in error



