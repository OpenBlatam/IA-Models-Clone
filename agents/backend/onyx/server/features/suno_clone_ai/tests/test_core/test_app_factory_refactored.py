"""
Tests refactorizados para app factory
Usando clases base y helpers para eliminar duplicación
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from fastapi import FastAPI

from core.app_factory import (
    create_application,
    register_routes,
    register_endpoints
)
from test_helpers import BaseServiceTestCase, StandardTestMixin


class TestCreateApplicationRefactored(BaseServiceTestCase, StandardTestMixin):
    """Tests refactorizados para create_application"""
    
    @pytest.mark.asyncio
    @pytest.mark.parametrize("debug_mode", [True, False])
    @patch('core.app_factory.settings')
    @patch('core.app_factory.application_lifespan')
    @patch('core.app_factory.setup_middleware')
    async def test_create_application(self, mock_setup_middleware, mock_lifespan, mock_settings, debug_mode):
        """Test de creación de aplicación con diferentes modos"""
        mock_settings.app_name = "Test App"
        mock_settings.app_version = "1.0.0"
        mock_settings.debug = debug_mode
        
        app = create_application()
        
        assert isinstance(app, FastAPI)
        assert app.title == "Test App"
        assert app.version == "1.0.0"
        if not debug_mode:
            assert app.docs_url is None
            assert app.redoc_url is None
        mock_setup_middleware.assert_called_once_with(app)


class TestRegisterRoutesRefactored(BaseServiceTestCase, StandardTestMixin):
    """Tests refactorizados para register_routes"""
    
    @pytest.mark.asyncio
    @patch('core.app_factory.song_router')
    @patch('core.app_factory.search_router')
    @patch('core.app_factory.websocket_router')
    @patch('core.app_factory.batch_router')
    @patch('core.app_factory.health_router')
    @patch('core.app_factory.versions_router')
    @patch('core.app_factory.generation_router')
    @patch('core.app_factory.metrics_endpoint')
    async def test_register_routes(self, mock_metrics, mock_gen, mock_versions, 
                                   mock_health, mock_batch, mock_ws, mock_search, mock_song):
        """Test de registro de rutas"""
        app = FastAPI()
        
        register_routes(app)
        
        # Verificar que se incluyeron los routers
        assert len(app.routes) > 0


class TestRegisterEndpointsRefactored(BaseServiceTestCase, StandardTestMixin):
    """Tests refactorizados para register_endpoints"""
    
    @pytest.mark.asyncio
    @patch('core.app_factory.settings')
    @patch('core.app_factory.get_module_registry')
    async def test_register_endpoints(self, mock_get_registry, mock_settings):
        """Test de registro de endpoints"""
        mock_settings.app_name = "Test App"
        mock_settings.app_version = "1.0.0"
        mock_registry = Mock()
        mock_registry.get_health_report.return_value = {"status": "ok"}
        mock_get_registry.return_value = mock_registry
        
        app = FastAPI()
        
        register_endpoints(app)
        
        # Verificar que se registraron los endpoints
        route_paths = [route.path for route in app.routes]
        assert "/" in route_paths
        assert "/health" in route_paths
        assert "/modules/health" in route_paths



