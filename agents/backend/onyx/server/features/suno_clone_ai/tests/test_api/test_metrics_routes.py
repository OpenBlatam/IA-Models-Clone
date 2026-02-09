"""
Tests para las rutas de métricas
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from fastapi import status
from fastapi.testclient import TestClient
from datetime import datetime

from api.routes.metrics import router
from services.metrics_service import MetricsService


@pytest.fixture
def mock_metrics_service():
    """Mock del servicio de métricas"""
    service = Mock(spec=MetricsService)
    service.get_stats = Mock(return_value={
        "total_songs": 100,
        "total_users": 50,
        "avg_generation_time": 5.2
    })
    return service


@pytest.fixture
def client(mock_metrics_service):
    """Cliente de prueba con mocks"""
    from fastapi import FastAPI
    from api.dependencies import MetricsServiceDep
    
    app = FastAPI()
    app.include_router(router)
    
    def get_metrics_service():
        return mock_metrics_service
    
    app.dependency_overrides[MetricsServiceDep] = get_metrics_service
    
    # Mock de módulos de monitoreo
    with patch('api.routes.metrics.GenerationMetrics') as mock_gen_metrics:
        with patch('api.routes.metrics.SystemMonitor') as mock_sys_monitor:
            with patch('api.routes.metrics.PerformanceTracker') as mock_perf:
                with patch('api.routes.metrics.get_alert_manager') as mock_alert:
                    mock_gen_metrics.return_value.get_stats.return_value = {}
                    mock_sys_monitor.return_value.get_system_info.return_value = {}
                    mock_perf.return_value.get_performance_trend.return_value = []
                    mock_alert.return_value.check_metrics.return_value = None
                    
                    yield TestClient(app)
    
    app.dependency_overrides.clear()


@pytest.mark.unit
@pytest.mark.api
class TestGetStats:
    """Tests para obtener estadísticas"""
    
    def test_get_stats_success(self, client, mock_metrics_service):
        """Test de obtención exitosa de estadísticas"""
        response = client.get("/metrics/stats")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "real_time" in data or "service_stats" in data
    
    def test_get_stats_with_days(self, client, mock_metrics_service):
        """Test con número de días personalizado"""
        response = client.get("/metrics/stats?days=30")
        
        assert response.status_code == status.HTTP_200_OK
        mock_metrics_service.get_stats.assert_called_with(days=30)


@pytest.mark.unit
@pytest.mark.api
class TestGetGenerationMetrics:
    """Tests para obtener métricas de generación"""
    
    def test_get_generation_metrics_success(self, client):
        """Test de obtención exitosa de métricas de generación"""
        with patch('api.routes.metrics._generation_metrics') as mock_metrics:
            mock_metrics.get_stats.return_value = {
                "total_generations": 100,
                "success_rate": 0.95
            }
            
            response = client.get("/metrics/generation")
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert isinstance(data, dict)


@pytest.mark.unit
@pytest.mark.api
class TestGetSystemInfo:
    """Tests para obtener información del sistema"""
    
    def test_get_system_info_success(self, client):
        """Test de obtención exitosa de información del sistema"""
        with patch('api.routes.metrics._system_monitor') as mock_monitor:
            mock_monitor.get_system_info.return_value = {
                "cpu_usage": 50.0,
                "memory_usage": 60.0
            }
            
            response = client.get("/metrics/system")
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert isinstance(data, dict)


@pytest.mark.integration
@pytest.mark.api
class TestMetricsIntegration:
    """Tests de integración para métricas"""
    
    def test_full_metrics_workflow(self, client, mock_metrics_service):
        """Test del flujo completo de métricas"""
        # 1. Stats generales
        stats_response = client.get("/metrics/stats?days=7")
        assert stats_response.status_code == status.HTTP_200_OK
        
        # 2. Métricas de generación
        gen_response = client.get("/metrics/generation")
        assert gen_response.status_code == status.HTTP_200_OK
        
        # 3. Información del sistema
        sys_response = client.get("/metrics/system")
        assert sys_response.status_code == status.HTTP_200_OK



