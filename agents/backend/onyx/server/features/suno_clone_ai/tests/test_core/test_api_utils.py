"""
Tests para API utils
"""

import pytest
from unittest.mock import Mock

from core.api.api_utils import (
    APIHandler,
    create_api_handler,
    validate_request
)
from test_helpers import BaseServiceTestCase, StandardTestMixin


class TestAPIHandler(BaseServiceTestCase, StandardTestMixin):
    """Tests para APIHandler"""
    
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
    
    def test_add_multiple_middlewares(self, api_handler):
        """Test de agregar múltiples middlewares"""
        middleware1 = Mock()
        middleware2 = Mock()
        
        api_handler.add_middleware(middleware1)
        api_handler.add_middleware(middleware2)
        
        assert len(api_handler.middlewares) == 2
    
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


class TestCreateApiHandler(BaseServiceTestCase, StandardTestMixin):
    """Tests para create_api_handler"""
    
    def test_create_api_handler(self):
        """Test de creación de API handler"""
        handler = create_api_handler()
        
        assert isinstance(handler, APIHandler)


class TestValidateRequest(BaseServiceTestCase, StandardTestMixin):
    """Tests para validate_request"""
    
    def test_validate_request_success(self):
        """Test de validación exitosa"""
        request = {
            'field1': 'value1',
            'field2': 'value2'
        }
        required_fields = ['field1', 'field2']
        
        is_valid, error = validate_request(request, required_fields)
        
        assert is_valid is True
        assert error is None
    
    def test_validate_request_missing_field(self):
        """Test de validación con campo faltante"""
        request = {
            'field1': 'value1'
        }
        required_fields = ['field1', 'field2']
        
        is_valid, error = validate_request(request, required_fields)
        
        assert is_valid is False
        assert 'Missing required field: field2' in error
    
    @pytest.mark.parametrize("required_fields", [
        ['field1'],
        ['field1', 'field2'],
        ['field1', 'field2', 'field3']
    ])
    def test_validate_request_different_fields(self, required_fields):
        """Test de validación con diferentes campos requeridos"""
        request = {field: f'value_{field}' for field in required_fields}
        
        is_valid, error = validate_request(request, required_fields)
        
        assert is_valid is True
        assert error is None



