"""
Tests para system initialization
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from core.initialization import SystemInitializer
from test_helpers import BaseServiceTestCase, StandardTestMixin


class TestSystemInitializer(BaseServiceTestCase, StandardTestMixin):
    """Tests para SystemInitializer"""
    
    @pytest.fixture
    def initializer(self):
        """Fixture para SystemInitializer"""
        return SystemInitializer()
    
    def test_initializer_init(self, initializer):
        """Test de inicialización"""
        assert initializer.initialized is False
        assert initializer.initialization_order == []
        assert initializer.initialization_times == {}
    
    @pytest.mark.asyncio
    @patch('core.initialization.get_container')
    @patch('core.initialization.StorageFactory')
    @patch('core.initialization.CacheFactory')
    @patch('core.initialization.MusicGeneratorFactory')
    @patch('core.initialization.get_event_bus')
    @patch('core.initialization.get_plugin_manager')
    @patch('core.initialization.get_module_registry')
    async def test_initialize_all_success(
        self, mock_get_registry, mock_get_plugins, mock_get_events,
        mock_music_factory, mock_cache_factory, mock_storage_factory,
        mock_get_container, initializer
    ):
        """Test de inicialización exitosa de todos los componentes"""
        # Setup mocks
        mock_container = Mock()
        mock_get_container.return_value = mock_container
        
        mock_storage = Mock()
        mock_storage_factory.create_storage.return_value = mock_storage
        
        mock_cache = Mock()
        mock_cache_factory.create_cache.return_value = mock_cache
        
        mock_generator = Mock()
        mock_music_factory.create_generator.return_value = mock_generator
        
        mock_event_bus = Mock()
        mock_get_events.return_value = mock_event_bus
        
        mock_plugin_manager = Mock()
        mock_get_plugins.return_value = mock_plugin_manager
        
        mock_registry = Mock()
        mock_get_registry.return_value = mock_registry
        
        results = await initializer.initialize_all()
        
        assert initializer.initialized is True
        assert len(results) > 0
        assert "dependency_container" in results
        assert "storage" in results
        assert "cache" in results
        assert "music_generator" in results
    
    @pytest.mark.asyncio
    async def test_initialize_all_already_initialized(self, initializer):
        """Test de inicialización cuando ya está inicializado"""
        initializer.initialized = True
        
        results = await initializer.initialize_all()
        
        assert results == {}
    
    @pytest.mark.asyncio
    @patch('core.initialization.get_container')
    @patch('core.initialization.StorageFactory')
    @patch('core.initialization.CacheFactory')
    @patch('core.initialization.MusicGeneratorFactory')
    @patch('core.initialization.get_event_bus')
    @patch('core.initialization.get_plugin_manager')
    @patch('core.initialization.get_module_registry')
    async def test_initialize_all_with_config(
        self, mock_get_registry, mock_get_plugins, mock_get_events,
        mock_music_factory, mock_cache_factory, mock_storage_factory,
        mock_get_container, initializer
    ):
        """Test de inicialización con configuración personalizada"""
        mock_container = Mock()
        mock_get_container.return_value = mock_container
        
        mock_storage = Mock()
        mock_storage_factory.create_storage.return_value = mock_storage
        
        mock_cache = Mock()
        mock_cache_factory.create_cache.return_value = mock_cache
        
        mock_generator = Mock()
        mock_music_factory.create_generator.return_value = mock_generator
        
        mock_event_bus = Mock()
        mock_get_events.return_value = mock_event_bus
        
        mock_plugin_manager = Mock()
        mock_get_plugins.return_value = mock_plugin_manager
        
        mock_registry = Mock()
        mock_get_registry.return_value = mock_registry
        
        config = {
            "storage_type": "s3",
            "cache_type": "redis",
            "generator_type": "advanced"
        }
        
        results = await initializer.initialize_all(config=config)
        
        assert initializer.initialized is True
        mock_storage_factory.create_storage.assert_called_with(
            storage_type="s3",
            base_path="storage"
        )
        mock_cache_factory.create_cache.assert_called_with(cache_type="redis")
        mock_music_factory.create_generator.assert_called_with(generator_type="advanced")
    
    def test_get_initialization_order(self, initializer):
        """Test de obtención del orden de inicialización"""
        initializer.initialization_order = ["service1", "service2", "service3"]
        
        order = initializer.get_initialization_order()
        
        assert order == ["service1", "service2", "service3"]
    
    def test_get_initialization_times(self, initializer):
        """Test de obtención de tiempos de inicialización"""
        initializer.initialization_times = {
            "service1": 0.1,
            "service2": 0.2
        }
        
        times = initializer.get_initialization_times()
        
        assert times["service1"] == 0.1
        assert times["service2"] == 0.2



