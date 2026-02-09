"""
Tests refactorizados para Factories
Usando clases base y helpers para eliminar duplicación
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from core.factories import ServiceFactory, MusicGeneratorFactory, CacheFactory
from test_helpers import BaseServiceTestCase, StandardTestMixin


class TestServiceFactoryRefactored(BaseServiceTestCase, StandardTestMixin):
    """Tests refactorizados para ServiceFactory class"""
    
    def test_register_service(self):
        """Test de registro de servicio"""
        class TestService:
            pass
        
        ServiceFactory.register("test_service", TestService)
        
        assert "test_service" in ServiceFactory._registry
        assert ServiceFactory._registry["test_service"] == TestService
    
    def test_create_service_singleton(self):
        """Test de creación de servicio retorna singleton"""
        class TestService:
            def __init__(self):
                self.value = "test"
        
        ServiceFactory.register("test_service", TestService)
        
        instance1 = ServiceFactory.create("test_service")
        instance2 = ServiceFactory.create("test_service")
        
        assert instance1 is instance2
        assert isinstance(instance1, TestService)
    
    @pytest.mark.parametrize("param1,param2", [
        ("value1", "value2"),
        ("value1", None),
        (123, 456)
    ])
    def test_create_service_with_kwargs(self, param1, param2):
        """Test de creación de servicio con diferentes kwargs"""
        class TestService:
            def __init__(self, param1, param2=None):
                self.param1 = param1
                self.param2 = param2
        
        ServiceFactory.register("test_service", TestService)
        
        instance = ServiceFactory.create("test_service", param1=param1, param2=param2)
        
        assert instance.param1 == param1
        assert instance.param2 == param2
    
    def test_create_service_not_registered(self):
        """Test de creación de servicio no registrado"""
        with pytest.raises(ValueError, match="not registered"):
            ServiceFactory.create("nonexistent")
    
    def test_get_service_exists(self):
        """Test de obtención de servicio existente"""
        class TestService:
            pass
        
        ServiceFactory.register("test_service", TestService)
        ServiceFactory.create("test_service")
        
        instance = ServiceFactory.get("test_service")
        assert instance is not None
        assert isinstance(instance, TestService)
    
    def test_get_service_not_exists(self):
        """Test de obtención de servicio no existente"""
        instance = ServiceFactory.get("nonexistent")
        assert instance is None
    
    def test_reset_factory(self):
        """Test de reset de factory"""
        class TestService:
            pass
        
        ServiceFactory.register("test_service", TestService)
        instance1 = ServiceFactory.create("test_service")
        
        ServiceFactory.reset()
        
        # Debería crear nueva instancia después del reset
        instance2 = ServiceFactory.create("test_service")
        assert instance1 is not instance2


class TestMusicGeneratorFactoryRefactored(BaseServiceTestCase, StandardTestMixin):
    """Tests refactorizados para MusicGeneratorFactory class"""
    
    @pytest.mark.parametrize("generator_type", ["default", "advanced", "ultra"])
    @patch('core.factories.MusicGenerator')
    def test_create_generator(self, mock_generator_class, generator_type):
        """Test de creación de generador con diferentes tipos"""
        mock_instance = Mock()
        mock_generator_class.return_value = mock_instance
        
        generator = MusicGeneratorFactory.create_generator(generator_type=generator_type)
        
        assert generator is not None
        mock_generator_class.assert_called()


class TestCacheFactoryRefactored(BaseServiceTestCase, StandardTestMixin):
    """Tests refactorizados para CacheFactory class"""
    
    @pytest.mark.parametrize("cache_type", ["memory", "redis", "memcached"])
    @patch('core.factories.CacheManager')
    def test_create_cache(self, mock_cache_class, cache_type):
        """Test de creación de cache con diferentes tipos"""
        mock_instance = Mock()
        mock_cache_class.return_value = mock_instance
        
        cache = CacheFactory.create_cache(cache_type=cache_type)
        
        assert cache is not None
        mock_cache_class.assert_called()



