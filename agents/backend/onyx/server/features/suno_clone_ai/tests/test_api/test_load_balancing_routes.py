"""
Tests para las rutas de load balancing
"""

import pytest
from unittest.mock import Mock, patch
from fastapi import status
from fastapi.testclient import TestClient

from api.routes.load_balancing import router
from services.load_balancer import LoadBalancer, LoadBalancingStrategy, Backend


@pytest.fixture
def mock_load_balancer():
    """Mock del load balancer"""
    balancer = Mock(spec=LoadBalancer)
    
    # Mock de backend
    backend = Mock(spec=Backend)
    backend.id = "backend-123"
    backend.url = "http://backend.example.com"
    backend.weight = 1
    backend.active_connections = 5
    backend.healthy = True
    
    balancer.add_backend = Mock(return_value=True)
    balancer.get_backend = Mock(return_value=backend)
    balancer.get_stats = Mock(return_value={
        "total_backends": 3,
        "healthy_backends": 2,
        "total_requests": 1000
    })
    return balancer


@pytest.fixture
def client(mock_load_balancer):
    """Cliente de prueba con mocks"""
    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(router)
    
    with patch('api.routes.load_balancing.get_load_balancer', return_value=mock_load_balancer):
        with patch('api.routes.load_balancing.require_role', return_value=lambda: None):
            yield TestClient(app)


@pytest.mark.unit
@pytest.mark.api
class TestAddBackend:
    """Tests para agregar backend"""
    
    def test_add_backend_success(self, client, mock_load_balancer):
        """Test de agregado exitoso"""
        response = client.post(
            "/load-balancer/backends",
            json={
                "backend_id": "backend-123",
                "url": "http://backend.example.com",
                "weight": 1
            }
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "message" in data
        assert data["backend_id"] == "backend-123"
    
    def test_add_backend_with_health_check(self, client, mock_load_balancer):
        """Test con health check URL"""
        response = client.post(
            "/load-balancer/backends",
            json={
                "backend_id": "backend-123",
                "url": "http://backend.example.com",
                "weight": 1,
                "health_check_url": "http://backend.example.com/health"
            }
        )
        
        assert response.status_code == status.HTTP_200_OK


@pytest.mark.unit
@pytest.mark.api
class TestGetBackend:
    """Tests para obtener backend"""
    
    def test_get_backend_success(self, client, mock_load_balancer):
        """Test de obtención exitosa"""
        response = client.get("/load-balancer/backend")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "backend_id" in data
        assert "url" in data
    
    def test_get_backend_no_available(self, client, mock_load_balancer):
        """Test cuando no hay backends disponibles"""
        mock_load_balancer.get_backend.return_value = None
        
        response = client.get("/load-balancer/backend")
        
        assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
        assert "No backends available" in response.json()["detail"]


@pytest.mark.unit
@pytest.mark.api
class TestGetStats:
    """Tests para obtener estadísticas"""
    
    def test_get_stats_success(self, client, mock_load_balancer):
        """Test de obtención exitosa"""
        response = client.get("/load-balancer/stats")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "stats" in data or isinstance(data, dict)


@pytest.mark.integration
@pytest.mark.api
class TestLoadBalancingIntegration:
    """Tests de integración para load balancing"""
    
    def test_full_load_balancing_workflow(self, client, mock_load_balancer):
        """Test del flujo completo"""
        # 1. Agregar backend
        add_response = client.post(
            "/load-balancer/backends",
            json={
                "backend_id": "backend-123",
                "url": "http://backend.example.com"
            }
        )
        assert add_response.status_code == status.HTTP_200_OK
        
        # 2. Obtener backend
        get_response = client.get("/load-balancer/backend")
        assert get_response.status_code == status.HTTP_200_OK
        
        # 3. Obtener estadísticas
        stats_response = client.get("/load-balancer/stats")
        assert stats_response.status_code == status.HTTP_200_OK



