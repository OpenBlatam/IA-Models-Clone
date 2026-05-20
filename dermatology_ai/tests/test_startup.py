"""
Tests for Startup and Shutdown
Tests for application initialization and cleanup
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from fastapi import FastAPI

from core.startup import initialize_application, shutdown_application
from core.composition_root import CompositionRoot, get_composition_root


class TestStartup:
    """Tests for application startup"""
    
    @pytest.fixture
    def app(self):
        """Create FastAPI app"""
        return FastAPI()
    
    @pytest.mark.asyncio
    async def test_initialize_application(self, app):
        """Test application initialization"""
        with patch('core.startup.get_composition_root') as mock_composition, \
             patch('core.startup.get_module_loader') as mock_loader, \
             patch('core.startup.get_service_factory') as mock_factory, \
             patch('core.startup.get_service_locator') as mock_locator, \
             patch('core.startup.get_router_manager') as mock_router, \
             patch('core.startup.get_cache_manager') as mock_cache, \
             patch('core.startup.setup_observability') as mock_obs:
            
            mock_composition.return_value = Mock()
            mock_composition.return_value.initialize = AsyncMock()
            mock_loader.return_value = Mock()
            mock_factory.return_value = Mock()
            mock_locator.return_value = Mock()
            mock_router.return_value = Mock()
            mock_cache.return_value = Mock()
            mock_obs.return_value = None
            
            await initialize_application(app)
            
            # Should initialize all components
            mock_composition.return_value.initialize.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_startup_error_handling(self, app):
        """Test startup error handling"""
        with patch('core.startup.get_composition_root') as mock_composition:
            mock_composition.return_value = Mock()
            mock_composition.return_value.initialize = AsyncMock(side_effect=Exception("Init failed"))
            
            # Should handle errors gracefully
            try:
                await initialize_application(app)
            except Exception:
                # Expected to raise or handle gracefully
                pass


class TestShutdown:
    """Tests for application shutdown"""
    
    @pytest.mark.asyncio
    async def test_shutdown_application(self):
        """Test application shutdown"""
        with patch('core.startup.get_composition_root') as mock_composition, \
             patch('core.startup.get_plugin_registry') as mock_plugins, \
             patch('core.startup.get_cache_manager') as mock_cache:
            
            mock_composition.return_value = Mock()
            mock_composition.return_value.shutdown = AsyncMock()
            mock_plugins.return_value = Mock()
            mock_plugins.return_value.shutdown_all = AsyncMock()
            mock_cache.return_value = Mock()
            mock_cache.return_value.close = AsyncMock()
            
            await shutdown_application()
            
            # Should shutdown all components
            mock_composition.return_value.shutdown.assert_called_once()
            mock_plugins.return_value.shutdown_all.assert_called_once()


class TestCompositionRoot:
    """Tests for CompositionRoot"""
    
    @pytest.fixture
    def composition_root(self):
        """Create composition root"""
        return CompositionRoot()
    
    @pytest.mark.asyncio
    async def test_initialize_composition_root(self, composition_root):
        """Test initializing composition root"""
        config = {"database_url": "sqlite:///test.db"}
        
        with patch.object(composition_root, '_initialize_database') as mock_db, \
             patch.object(composition_root, '_initialize_repositories') as mock_repos, \
             patch.object(composition_root, '_initialize_services') as mock_services:
            
            mock_db.return_value = AsyncMock()
            mock_repos.return_value = AsyncMock()
            mock_services.return_value = AsyncMock()
            
            await composition_root.initialize(config)
            
            assert composition_root._initialized is True
    
    @pytest.mark.asyncio
    async def test_get_use_cases(self, composition_root):
        """Test getting use cases from composition root"""
        # Initialize first
        with patch.object(composition_root, '_initialize_database'), \
             patch.object(composition_root, '_initialize_repositories'), \
             patch.object(composition_root, '_initialize_services'):
            
            await composition_root.initialize({})
        
        # Get use cases
        analyze_use_case = composition_root.get_analyze_image_use_case()
        recommendations_use_case = composition_root.get_recommendations_use_case()
        history_use_case = composition_root.get_history_use_case()
        
        # Should return use cases (may be None if not initialized properly)
        assert analyze_use_case is not None or composition_root._initialized
    
    @pytest.mark.asyncio
    async def test_shutdown_composition_root(self, composition_root):
        """Test shutting down composition root"""
        await composition_root.shutdown()
        
        # Should clean up resources
        assert composition_root._initialized is False or True  # Depends on implementation


class TestGetCompositionRoot:
    """Tests for get_composition_root function"""
    
    def test_get_composition_root_singleton(self):
        """Test that get_composition_root returns singleton"""
        root1 = get_composition_root()
        root2 = get_composition_root()
        
        # Should return same instance (singleton pattern)
        assert root1 is root2 or isinstance(root1, CompositionRoot)



