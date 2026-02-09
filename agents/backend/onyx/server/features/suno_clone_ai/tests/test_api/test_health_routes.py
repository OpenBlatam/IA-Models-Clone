"""
Tests para las rutas de health checks
"""

import pytest
from unittest.mock import Mock, patch
from fastapi import status
from fastapi.testclient import TestClient

from api.routes.health import router


@pytest.fixture
def client():
    """Cliente de prueba"""
    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


@pytest.mark.unit
@pytest.mark.api
class TestHealthCheck:
    """Tests para health check"""
    
    def test_health_check_success(self, client):
        """Test de health check exitoso"""
        with patch('api.routes.health.get_music_generator') as mock_gen:
            with patch('api.routes.health.get_cache_manager') as mock_cache:
                with patch('api.routes.health.SongService') as mock_song:
                    mock_gen.return_value = Mock()
                    mock_cache.return_value = Mock()
                    mock_song.return_value = Mock()
                    
                    response = client.get("/health")
                    
                    assert response.status_code == status.HTTP_200_OK
                    data = response.json()
                    assert "status" in data
                    assert "timestamp" in data
                    assert "services" in data
                    assert data["status"] in ["healthy", "degraded", "unhealthy"]
    
    def test_health_check_degraded(self, client):
        """Test cuando algunos servicios fallan"""
        with patch('api.routes.health.get_music_generator', side_effect=Exception("Error")):
            with patch('api.routes.health.get_cache_manager') as mock_cache:
                with patch('api.routes.health.SongService') as mock_song:
                    mock_cache.return_value = Mock()
                    mock_song.return_value = Mock()
                    
                    response = client.get("/health")
                    
                    assert response.status_code == status.HTTP_200_OK
                    data = response.json()
                    assert data["status"] in ["degraded", "unhealthy"]
    
    def test_health_check_unhealthy(self, client):
        """Test cuando el servicio principal falla"""
        with patch('api.routes.health.get_music_generator') as mock_gen:
            with patch('api.routes.health.get_cache_manager') as mock_cache:
                with patch('api.routes.health.SongService', side_effect=Exception("Critical error")):
                    mock_gen.return_value = Mock()
                    mock_cache.return_value = Mock()
                    
                    response = client.get("/health")
                    
                    assert response.status_code == status.HTTP_200_OK
                    data = response.json()
                    assert data["status"] == "unhealthy"


@pytest.mark.unit
@pytest.mark.api
class TestReadinessCheck:
    """Tests para readiness check"""
    
    def test_readiness_check_success(self, client):
        """Test de readiness check exitoso"""
        with patch('api.routes.health.SongService') as mock_song:
            mock_song.return_value = Mock()
            
            response = client.get("/health/ready")
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["ready"] is True
            assert "timestamp" in data
    
    def test_readiness_check_failure(self, client):
        """Test cuando el servicio no está listo"""
        with patch('api.routes.health.SongService', side_effect=Exception("Not ready")):
            response = client.get("/health/ready")
            
            assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
            assert "not ready" in response.json()["detail"].lower()


@pytest.mark.unit
@pytest.mark.api
class TestLivenessCheck:
    """Tests para liveness check"""
    
    def test_liveness_check_success(self, client):
        """Test de liveness check exitoso"""
        response = client.get("/health/live")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["alive"] is True
        assert "timestamp" in data
    
    def test_liveness_check_always_returns(self, client):
        """Test que liveness siempre retorna"""
        # Liveness debería ser muy simple y siempre funcionar
        response = client.get("/health/live")
        assert response.status_code == status.HTTP_200_OK


@pytest.mark.integration
@pytest.mark.api
class TestHealthIntegration:
    """Tests de integración para health checks"""
    
    def test_all_health_endpoints(self, client):
        """Test de todos los endpoints de health"""
        # Health check
        health_response = client.get("/health")
        assert health_response.status_code == status.HTTP_200_OK
        
        # Readiness check
        ready_response = client.get("/health/ready")
        assert ready_response.status_code in [status.HTTP_200_OK, status.HTTP_503_SERVICE_UNAVAILABLE]
        
        # Liveness check
        live_response = client.get("/health/live")
        assert live_response.status_code == status.HTTP_200_OK



