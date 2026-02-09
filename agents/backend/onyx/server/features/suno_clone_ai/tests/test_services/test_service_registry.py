"""
Tests para service registry
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Type

from services.service_registry import (
    ServiceRegistry,
    get_service_registry,
    register_service
)


@pytest.fixture
def mock_container():
    """Mock del contenedor de dependencias"""
    container = MagicMock()
    container.register = Mock()
    container.get = Mock(return_value=None)
    container.resolve = Mock(return_value=None)
    container.has = Mock(return_value=False)
    return container


@pytest.fixture
def service_registry(mock_container):
    """Fixture para ServiceRegistry con mock container"""
    with patch('services.service_registry.get_container', return_value=mock_container):
        return ServiceRegistry()


@pytest.mark.unit
@pytest.mark.services
class TestServiceRegistry:
    """Tests para ServiceRegistry"""
    
    def test_service_registry_init(self, service_registry):
        """Test de inicialización"""
        assert service_registry._service_types == {}
    
    def test_register_service(self, service_registry, mock_container):
        """Test de registro de servicio"""
        service = Mock()
        
        service_registry.register("test_service", service)
        
        mock_container.register.assert_called_once_with("test_service", service, singleton=True)
    
    def test_register_service_with_type(self, service_registry, mock_container):
        """Test de registro con tipo"""
        service = Mock()
        service_type = type(service)
        
        service_registry.register("test_service", service, service_type=service_type)
        
        assert "test_service" in service_registry._service_types
        assert service_registry._service_types["test_service"] == service_type
    
    def test_register_service_not_singleton(self, service_registry, mock_container):
        """Test de registro sin singleton"""
        service = Mock()
        
        service_registry.register("test_service", service, singleton=False)
        
        mock_container.register.assert_called_once_with("test_service", service, singleton=False)
    
    def test_get_service(self, service_registry, mock_container):
        """Test de obtención de servicio"""
        mock_service = Mock()
        mock_container.get.return_value = mock_service
        
        result = service_registry.get("test_service")
        
        assert result == mock_service
        mock_container.get.assert_called_once_with("test_service")
    
    def test_resolve_service(self, service_registry, mock_container):
        """Test de resolución por tipo"""
        class TestService:
            pass
        
        mock_service = TestService()
        mock_container.resolve.return_value = mock_service
        
        result = service_registry.resolve(TestService)
        
        assert result == mock_service
        mock_container.resolve.assert_called_once_with(TestService)
    
    def test_has_service(self, service_registry, mock_container):
        """Test de verificación de existencia"""
        mock_container.has.return_value = True
        
        result = service_registry.has("test_service")
        
        assert result is True
        mock_container.has.assert_called_once_with("test_service")
    
    def test_list_services(self, service_registry):
        """Test de listado de servicios"""
        service_registry._service_types = {
            "service1": Mock,
            "service2": Mock
        }
        
        services = service_registry.list_services()
        
        assert "service1" in services
        assert "service2" in services


@pytest.mark.unit
@pytest.mark.services
class TestGetServiceRegistry:
    """Tests para get_service_registry"""
    
    def test_get_service_registry_singleton(self):
        """Test de que retorna singleton"""
        with patch('services.service_registry._service_registry', None):
            instance1 = get_service_registry()
            instance2 = get_service_registry()
            
            assert instance1 is instance2
            assert isinstance(instance1, ServiceRegistry)


@pytest.mark.unit
@pytest.mark.services
class TestRegisterService:
    """Tests para register_service helper"""
    
    def test_register_service_helper(self, mock_container):
        """Test de helper register_service"""
        with patch('services.service_registry.get_service_registry') as mock_get_registry:
            registry = Mock()
            mock_get_registry.return_value = registry
            
            service = Mock()
            register_service("test_service", service)
            
            registry.register.assert_called_once_with("test_service", service, service_type=None, singleton=True)



