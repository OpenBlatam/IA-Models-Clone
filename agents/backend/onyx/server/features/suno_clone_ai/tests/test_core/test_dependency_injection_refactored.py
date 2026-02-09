"""
Tests refactorizados para dependency injection
Usando clases base y helpers para eliminar duplicación
"""

import pytest
from unittest.mock import Mock, patch
from typing import Type

from core.dependency_injection import (
    DependencyContainer,
    get_container
)
from test_helpers import BaseServiceTestCase, StandardTestMixin


class TestDependencyContainerRefactored(BaseServiceTestCase, StandardTestMixin):
    """Tests refactorizados para DependencyContainer"""
    
    @pytest.fixture
    def container(self):
        """Fixture para DependencyContainer"""
        return DependencyContainer()
    
    def test_container_init(self, container):
        """Test de inicialización"""
        assert container._services == {}
        assert container._factories == {}
        assert container._singletons == {}
    
    @pytest.mark.parametrize("singleton", [True, False])
    def test_register_service(self, container, singleton):
        """Test de registro de servicio con diferentes configuraciones"""
        service = Mock()
        
        container.register("test_service", service, singleton=singleton)
        
        if singleton:
            assert "test_service" in container._singletons
            assert container._singletons["test_service"] == service
        else:
            assert "test_service" in container._services
            assert container._services["test_service"] == service
    
    def test_register_factory(self, container):
        """Test de registro de factory"""
        factory = Mock(return_value=Mock())
        
        container.register_factory("test_service", factory)
        
        assert "test_service" in container._factories
        assert container._factories["test_service"] == factory
    
    def test_get_service_singleton(self, container):
        """Test de obtención de servicio singleton"""
        service = Mock()
        container.register("test_service", service, singleton=True)
        
        result = container.get("test_service")
        
        assert result == service
    
    def test_get_service_factory(self, container):
        """Test de obtención de servicio desde factory"""
        service = Mock()
        factory = Mock(return_value=service)
        
        container.register_factory("test_service", factory)
        
        result = container.get("test_service")
        
        assert result == service
        factory.assert_called_once()
    
    def test_get_service_not_found(self, container):
        """Test de servicio no encontrado"""
        result = container.get("nonexistent")
        
        assert result is None
    
    def test_resolve_by_type(self, container):
        """Test de resolución por tipo"""
        class TestService:
            pass
        
        service = TestService()
        container.register("TestService", service)
        
        result = container.resolve(TestService)
        
        assert result == service
    
    @pytest.mark.parametrize("has_service", [True, False])
    def test_has_service(self, container, has_service):
        """Test de verificación de existencia"""
        if has_service:
            container.register("test_service", Mock())
            assert container.has("test_service") is True
        else:
            assert container.has("nonexistent") is False
    
    def test_clear(self, container):
        """Test de limpieza"""
        container.register("service1", Mock())
        container.register("service2", Mock(), singleton=True)
        container.register_factory("service3", Mock())
        
        container.clear()
        
        assert len(container._services) == 0
        assert len(container._singletons) == 0
        assert len(container._factories) == 0


class TestGetContainerRefactored(BaseServiceTestCase, StandardTestMixin):
    """Tests refactorizados para get_container"""
    
    def test_get_container_singleton(self):
        """Test de que retorna singleton"""
        with patch('core.dependency_injection._container', None):
            instance1 = get_container()
            instance2 = get_container()
            
            assert instance1 is instance2
            assert isinstance(instance1, DependencyContainer)



