"""
Tests para las rutas de inferencia distribuida
"""

import pytest
from unittest.mock import Mock, patch
from fastapi import status
from fastapi.testclient import TestClient

from api.routes.distributed import router
from services.distributed_inference import DistributedInference


@pytest.fixture
def mock_distributed_inference():
    """Mock del servicio de inferencia distribuida"""
    inference = Mock(spec=DistributedInference)
    
    # Mock de worker
    worker = Mock()
    worker.id = "worker-123"
    worker.url = "http://worker.example.com"
    worker.capacity = 10
    worker.active_tasks = 5
    worker.healthy = True
    
    inference.register_worker = Mock(return_value=True)
    inference.get_available_worker = Mock(return_value=worker)
    inference.get_stats = Mock(return_value={
        "total_workers": 5,
        "active_workers": 4,
        "total_capacity": 50
    })
    return inference


@pytest.fixture
def client(mock_distributed_inference):
    """Cliente de prueba con mocks"""
    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(router)
    
    with patch('api.routes.distributed.get_distributed_inference', return_value=mock_distributed_inference):
        with patch('api.routes.distributed.require_role', return_value=lambda: None):
            yield TestClient(app)


@pytest.mark.unit
@pytest.mark.api
class TestRegisterWorker:
    """Tests para registrar worker"""
    
    def test_register_worker_success(self, client, mock_distributed_inference):
        """Test de registro exitoso"""
        response = client.post(
            "/distributed/workers",
            json={
                "worker_id": "worker-123",
                "url": "http://worker.example.com",
                "capacity": 10
            }
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "message" in data
        assert data["worker_id"] == "worker-123"
        assert data["url"] == "http://worker.example.com"
        assert data["capacity"] == 10


@pytest.mark.unit
@pytest.mark.api
class TestGetAvailableWorker:
    """Tests para obtener worker disponible"""
    
    def test_get_available_worker_success(self, client, mock_distributed_inference):
        """Test de obtención exitosa"""
        response = client.get("/distributed/worker")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "worker_id" in data
        assert "url" in data
        assert "capacity" in data
        assert "available_capacity" in data
    
    def test_get_available_worker_none_available(self, client, mock_distributed_inference):
        """Test cuando no hay workers disponibles"""
        mock_distributed_inference.get_available_worker.return_value = None
        
        response = client.get("/distributed/worker")
        
        assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
        assert "No workers available" in response.json()["detail"]


@pytest.mark.unit
@pytest.mark.api
class TestGetStats:
    """Tests para obtener estadísticas"""
    
    def test_get_stats_success(self, client, mock_distributed_inference):
        """Test de obtención exitosa"""
        response = client.get("/distributed/stats")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "total_workers" in data or "stats" in data


@pytest.mark.integration
@pytest.mark.api
class TestDistributedIntegration:
    """Tests de integración para inferencia distribuida"""
    
    def test_full_distributed_workflow(self, client, mock_distributed_inference):
        """Test del flujo completo de inferencia distribuida"""
        # 1. Registrar worker
        register_response = client.post(
            "/distributed/workers",
            json={
                "worker_id": "worker-123",
                "url": "http://worker.example.com",
                "capacity": 10
            }
        )
        assert register_response.status_code == status.HTTP_200_OK
        
        # 2. Obtener worker disponible
        worker_response = client.get("/distributed/worker")
        assert worker_response.status_code == status.HTTP_200_OK
        
        # 3. Obtener estadísticas
        stats_response = client.get("/distributed/stats")
        assert stats_response.status_code == status.HTTP_200_OK



