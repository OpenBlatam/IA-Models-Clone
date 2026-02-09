"""
Tests para el sistema de degradación elegante
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock

from core.graceful_degradation import (
    GracefulDegradation,
    DegradationLevel,
    ServiceStatus,
    get_graceful_degradation
)


@pytest.fixture
def degradation():
    """Fixture para GracefulDegradation"""
    return GracefulDegradation()


@pytest.fixture
def mock_health_check():
    """Mock de health check"""
    return Mock(return_value=True)


@pytest.fixture
def mock_fallback():
    """Mock de fallback"""
    return Mock(return_value="fallback_result")


@pytest.mark.unit
@pytest.mark.core
class TestGracefulDegradation:
    """Tests para GracefulDegradation"""
    
    def test_init(self, degradation):
        """Test de inicialización"""
        assert degradation._services == {}
        assert degradation._fallbacks == {}
        assert degradation._degradation_strategies == {}
    
    def test_register_service(self, degradation, mock_health_check, mock_fallback):
        """Test de registro de servicio"""
        degradation.register_service(
            "test_service",
            mock_health_check,
            fallback=mock_fallback
        )
        
        assert "test_service" in degradation._services
        assert "test_service" in degradation._fallbacks
        assert degradation._services["test_service"].name == "test_service"
        assert degradation._services["test_service"].available is True
    
    def test_register_service_no_fallback(self, degradation, mock_health_check):
        """Test de registro sin fallback"""
        degradation.register_service("test_service", mock_health_check)
        
        assert "test_service" in degradation._services
        assert "test_service" not in degradation._fallbacks
    
    def test_set_degradation_strategy(self, degradation):
        """Test de configuración de estrategia de degradación"""
        strategy = Mock()
        
        degradation.set_degradation_strategy(
            "test_service",
            DegradationLevel.MINOR,
            strategy
        )
        
        assert "test_service" in degradation._degradation_strategies
        assert DegradationLevel.MINOR in degradation._degradation_strategies["test_service"]
    
    @pytest.mark.asyncio
    async def test_check_service_success(self, degradation, mock_health_check):
        """Test de verificación de servicio exitosa"""
        degradation.register_service("test_service", mock_health_check)
        
        status = await degradation.check_service("test_service")
        
        assert status.available is True
        assert status.degradation_level == DegradationLevel.NONE
    
    @pytest.mark.asyncio
    async def test_check_service_not_registered(self, degradation):
        """Test de verificación de servicio no registrado"""
        with pytest.raises(ValueError, match="not registered"):
            await degradation.check_service("nonexistent")
    
    @pytest.mark.asyncio
    async def test_execute_with_fallback_success(self, degradation, mock_health_check):
        """Test de ejecución exitosa sin fallback"""
        degradation.register_service("test_service", mock_health_check)
        
        async def operation():
            return "success"
        
        result = await degradation.execute_with_fallback(
            "test_service",
            operation
        )
        
        assert result == "success"
    
    @pytest.mark.asyncio
    async def test_execute_with_fallback_uses_fallback(self, degradation, mock_health_check, mock_fallback):
        """Test de ejecución usando fallback"""
        degradation.register_service("test_service", mock_health_check)
        degradation._services["test_service"].available = False
        
        async def operation():
            raise RuntimeError("Service unavailable")
        
        result = await degradation.execute_with_fallback(
            "test_service",
            operation
        )
        
        assert result == "fallback_result"
        mock_fallback.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_execute_with_fallback_sync_operation(self, degradation, mock_health_check):
        """Test de ejecución con operación síncrona"""
        degradation.register_service("test_service", mock_health_check)
        
        def sync_operation():
            return "sync_success"
        
        result = await degradation.execute_with_fallback(
            "test_service",
            sync_operation
        )
        
        assert result == "sync_success"
    
    @pytest.mark.asyncio
    async def test_execute_with_fallback_sync_fallback(self, degradation, mock_health_check):
        """Test de ejecución con fallback síncrono"""
        sync_fallback = Mock(return_value="sync_fallback")
        degradation.register_service("test_service", mock_health_check, fallback=sync_fallback)
        degradation._services["test_service"].available = False
        
        async def operation():
            raise RuntimeError("Error")
        
        result = await degradation.execute_with_fallback(
            "test_service",
            operation
        )
        
        assert result == "sync_fallback"
    
    @pytest.mark.asyncio
    async def test_execute_with_fallback_no_fallback_available(self, degradation, mock_health_check):
        """Test de ejecución sin fallback disponible"""
        degradation.register_service("test_service", mock_health_check)
        degradation._services["test_service"].available = False
        
        async def operation():
            raise RuntimeError("Error")
        
        with pytest.raises(RuntimeError, match="unavailable and no fallback"):
            await degradation.execute_with_fallback(
                "test_service",
                operation
            )
    
    @pytest.mark.asyncio
    async def test_execute_with_fallback_operation_fails_then_fallback(self, degradation, mock_health_check, mock_fallback):
        """Test de ejecución donde operación falla y se usa fallback"""
        degradation.register_service("test_service", mock_health_check, fallback=mock_fallback)
        
        async def operation():
            raise RuntimeError("Operation failed")
        
        result = await degradation.execute_with_fallback(
            "test_service",
            operation
        )
        
        assert result == "fallback_result"
        assert degradation._services["test_service"].available is False
    
    def test_get_service_status(self, degradation, mock_health_check):
        """Test de obtención de estado de servicio"""
        degradation.register_service("test_service", mock_health_check)
        
        status = degradation.get_service_status("test_service")
        
        assert status is not None
        assert status.name == "test_service"
    
    def test_get_service_status_not_found(self, degradation):
        """Test de obtención de estado de servicio no encontrado"""
        status = degradation.get_service_status("nonexistent")
        
        assert status is None
    
    def test_get_all_status(self, degradation, mock_health_check):
        """Test de obtención de todos los estados"""
        degradation.register_service("service1", mock_health_check)
        degradation.register_service("service2", mock_health_check)
        
        all_status = degradation.get_all_status()
        
        assert len(all_status) == 2
        assert "service1" in all_status
        assert "service2" in all_status


@pytest.mark.unit
@pytest.mark.core
class TestDegradationLevel:
    """Tests para DegradationLevel enum"""
    
    def test_degradation_level_values(self):
        """Test de valores del enum"""
        assert DegradationLevel.NONE.value == "none"
        assert DegradationLevel.MINOR.value == "minor"
        assert DegradationLevel.MAJOR.value == "major"
        assert DegradationLevel.CRITICAL.value == "critical"


@pytest.mark.unit
@pytest.mark.core
class TestServiceStatus:
    """Tests para ServiceStatus"""
    
    def test_service_status_creation(self):
        """Test de creación de ServiceStatus"""
        status = ServiceStatus(
            name="test",
            available=True,
            response_time=0.5,
            error_rate=0.1
        )
        
        assert status.name == "test"
        assert status.available is True
        assert status.response_time == 0.5
        assert status.error_rate == 0.1
        assert status.degradation_level == DegradationLevel.NONE


@pytest.mark.unit
@pytest.mark.core
class TestGetGracefulDegradation:
    """Tests para get_graceful_degradation"""
    
    def test_get_graceful_degradation_singleton(self):
        """Test de que retorna singleton"""
        instance1 = get_graceful_degradation()
        instance2 = get_graceful_degradation()
        
        assert instance1 is instance2
        assert isinstance(instance1, GracefulDegradation)
