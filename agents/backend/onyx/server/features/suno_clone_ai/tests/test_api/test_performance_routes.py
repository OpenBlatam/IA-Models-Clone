"""
Tests para las rutas de rendimiento
"""

import pytest
from unittest.mock import Mock, patch
from fastapi import status
from fastapi.testclient import TestClient

from api.routes.performance import router


@pytest.fixture
def client():
    """Cliente de prueba con mocks"""
    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(router)
    
    with patch('api.routes.performance.get_performance_stats', return_value={"avg_time": 0.5}):
        with patch('api.routes.performance.get_cache_stats', return_value={"hits": 100}):
            with patch('api.routes.performance.clear_performance_stats'):
                with patch('api.routes.performance.clear_response_cache'):
                    yield TestClient(app)


@pytest.mark.unit
@pytest.mark.api
class TestPerformanceStats:
    """Tests para estadísticas de rendimiento"""
    
    def test_get_performance_stats_success(self, client):
        """Test de obtención exitosa de estadísticas"""
        response = client.get("/performance/stats")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "performance_stats" in data
        assert "cache_stats" in data
    
    def test_get_performance_stats_with_operation(self, client):
        """Test con operación específica"""
        response = client.get("/performance/stats?operation=generate")
        
        assert response.status_code == status.HTTP_200_OK


@pytest.mark.unit
@pytest.mark.api
class TestClearPerformanceStats:
    """Tests para limpiar estadísticas"""
    
    def test_clear_performance_stats_success(self, client):
        """Test de limpieza exitosa"""
        response = client.post("/performance/stats/clear")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "message" in data
        assert "cleared" in data["message"].lower()


@pytest.mark.integration
@pytest.mark.api
class TestPerformanceIntegration:
    """Tests de integración para rendimiento"""
    
    def test_full_performance_workflow(self, client):
        """Test del flujo completo de rendimiento"""
        # 1. Obtener estadísticas
        stats_response = client.get("/performance/stats")
        assert stats_response.status_code == status.HTTP_200_OK
        
        # 2. Limpiar estadísticas
        clear_response = client.post("/performance/stats/clear")
        assert clear_response.status_code == status.HTTP_200_OK
        
        # 3. Obtener estadísticas después de limpiar
        stats_after_response = client.get("/performance/stats")
        assert stats_after_response.status_code == status.HTTP_200_OK



