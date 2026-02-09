"""
Tests refactorizados para base service
Usando clases base y helpers para eliminar duplicación
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from services.base_service import BaseService
from test_helpers import BaseServiceTestCase, StandardTestMixin


class ConcreteService(BaseService):
    """Servicio concreto para tests"""
    
    async def _on_initialize(self, config):
        """Inicialización personalizada"""
        return True
    
    async def _on_shutdown(self):
        """Cierre personalizado"""
        pass


class TestBaseServiceRefactored(BaseServiceTestCase, StandardTestMixin):
    """Tests refactorizados para BaseService"""
    
    def test_base_service_init(self):
        """Test de inicialización"""
        service = ConcreteService("test_service")
        
        assert service.service_name == "test_service"
        assert service.initialized is False
        assert service.initialized_at is None
        assert service.logger is not None
    
    @pytest.mark.asyncio
    @pytest.mark.parametrize("config", [
        None,
        {},
        {"key": "value"},
        {"setting1": "value1", "setting2": 123}
    ])
    async def test_initialize(self, config):
        """Test de inicialización con diferentes configuraciones"""
        service = ConcreteService("test_service")
        
        result = await service.initialize(config=config)
        
        assert result is True
        assert service.initialized is True
        assert service.initialized_at is not None
    
    @pytest.mark.asyncio
    async def test_initialize_already_initialized(self):
        """Test de inicialización cuando ya está inicializado"""
        service = ConcreteService("test_service")
        await service.initialize()
        
        result = await service.initialize()
        
        assert result is True
    
    @pytest.mark.asyncio
    async def test_shutdown(self):
        """Test de cierre"""
        service = ConcreteService("test_service")
        await service.initialize()
        
        await service.shutdown()
        
        assert service.initialized is False
    
    @pytest.mark.asyncio
    async def test_shutdown_not_initialized(self):
        """Test de cierre cuando no está inicializado"""
        service = ConcreteService("test_service")
        
        # No debería lanzar error
        await service.shutdown()
    
    @pytest.mark.asyncio
    @patch('services.base_service.get_event_bus')
    async def test_publish_event(self, mock_get_event_bus):
        """Test de publicación de evento"""
        mock_event_bus = Mock()
        mock_event_bus.publish = AsyncMock()
        mock_get_event_bus.return_value = mock_event_bus
        
        service = ConcreteService("test_service")
        await service.initialize()
        
        from core.events import EventType
        service._publish_event(EventType.MUSIC_GENERATED, {"key": "value"})
        
        # Dar tiempo para que se ejecute la tarea async
        import asyncio
        await asyncio.sleep(0.01)
        
        # Verificar que se llamó publish
        assert mock_event_bus.publish.called
    
    def test_get_stats(self):
        """Test de obtención de estadísticas"""
        service = ConcreteService("test_service")
        
        stats = service.get_stats()
        
        assert stats["service_name"] == "test_service"
        assert stats["initialized"] is False
        assert "initialized_at" in stats



