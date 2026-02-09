"""
Tests para las rutas de procesamiento por lotes
"""

import pytest
from unittest.mock import Mock, patch
from fastapi import status
from fastapi.testclient import TestClient

from api.routes.batch_processing import router
from services.batch_processor import BatchProcessor, BatchPriority


@pytest.fixture
def mock_batch_processor():
    """Mock del procesador por lotes"""
    processor = Mock(spec=BatchProcessor)
    
    # Mock de batch
    batch = Mock()
    batch.batch_id = "batch-123"
    batch.status = "processing"
    batch.items_count = 10
    batch.completed_count = 5
    batch.failed_count = 0
    batch.priority = BatchPriority.NORMAL
    
    processor.create_batch = Mock(return_value="batch-123")
    processor.get_batch = Mock(return_value=batch)
    processor.process_batch = Mock(return_value=True)
    processor.cancel_batch = Mock(return_value=True)
    
    return processor


@pytest.fixture
def client(mock_batch_processor):
    """Cliente de prueba con mocks"""
    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(router)
    
    with patch('api.routes.batch_processing.get_batch_processor', return_value=mock_batch_processor):
        with patch('api.routes.batch_processing.get_current_user', return_value={"user_id": "test_user"}):
            yield TestClient(app)


@pytest.mark.unit
@pytest.mark.api
class TestCreateBatch:
    """Tests para crear batch"""
    
    def test_create_batch_success(self, client, mock_batch_processor):
        """Test de creación exitosa de batch"""
        response = client.post(
            "/batch/create",
            json={
                "items": [{"id": 1}, {"id": 2}, {"id": 3}],
                "priority": "normal"
            }
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "batch_id" in data
        assert data["message"] == "Batch created successfully"
        assert data["items_count"] == 3
        assert data["priority"] == "normal"
    
    def test_create_batch_different_priorities(self, client, mock_batch_processor):
        """Test con diferentes prioridades"""
        priorities = ["low", "normal", "high", "urgent"]
        
        for priority in priorities:
            response = client.post(
                "/batch/create",
                json={
                    "items": [{"id": 1}],
                    "priority": priority
                }
            )
            assert response.status_code == status.HTTP_200_OK
            assert response.json()["priority"] == priority
    
    def test_create_batch_invalid_priority(self, client):
        """Test con prioridad inválida"""
        response = client.post(
            "/batch/create",
            json={
                "items": [{"id": 1}],
                "priority": "invalid"
            }
        )
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "Invalid priority" in response.json()["detail"]
    
    def test_create_batch_empty_items(self, client):
        """Test con items vacíos"""
        response = client.post(
            "/batch/create",
            json={
                "items": [],
                "priority": "normal"
            }
        )
        
        # Puede ser válido o inválido dependiendo de la validación
        assert response.status_code in [
            status.HTTP_200_OK,
            status.HTTP_400_BAD_REQUEST,
            status.HTTP_422_UNPROCESSABLE_ENTITY
        ]


@pytest.mark.unit
@pytest.mark.api
class TestGetBatchStatus:
    """Tests para obtener estado de batch"""
    
    def test_get_batch_status_success(self, client, mock_batch_processor):
        """Test de obtención exitosa de estado"""
        response = client.get("/batch/batch-123")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "batch_id" in data
        assert "status" in data
        assert data["status"] == "processing"
    
    def test_get_batch_status_not_found(self, client, mock_batch_processor):
        """Test cuando el batch no existe"""
        mock_batch_processor.get_batch.return_value = None
        
        response = client.get("/batch/nonexistent")
        
        assert response.status_code == status.HTTP_404_NOT_FOUND


@pytest.mark.unit
@pytest.mark.api
class TestProcessBatch:
    """Tests para procesar batch"""
    
    def test_process_batch_success(self, client, mock_batch_processor):
        """Test de procesamiento exitoso"""
        response = client.post("/batch/batch-123/process")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["message"] == "Batch processing started"
    
    def test_process_batch_not_found(self, client, mock_batch_processor):
        """Test cuando el batch no existe"""
        mock_batch_processor.get_batch.return_value = None
        
        response = client.post("/batch/nonexistent/process")
        
        assert response.status_code == status.HTTP_404_NOT_FOUND


@pytest.mark.unit
@pytest.mark.api
class TestCancelBatch:
    """Tests para cancelar batch"""
    
    def test_cancel_batch_success(self, client, mock_batch_processor):
        """Test de cancelación exitosa"""
        response = client.post("/batch/batch-123/cancel")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["message"] == "Batch cancelled successfully"
    
    def test_cancel_batch_not_found(self, client, mock_batch_processor):
        """Test cuando el batch no existe"""
        mock_batch_processor.get_batch.return_value = None
        
        response = client.post("/batch/nonexistent/cancel")
        
        assert response.status_code == status.HTTP_404_NOT_FOUND


@pytest.mark.integration
@pytest.mark.api
class TestBatchProcessingIntegration:
    """Tests de integración para procesamiento por lotes"""
    
    def test_full_batch_workflow(self, client, mock_batch_processor):
        """Test del flujo completo de procesamiento por lotes"""
        # 1. Crear batch
        create_response = client.post(
            "/batch/create",
            json={
                "items": [{"id": i} for i in range(10)],
                "priority": "high"
            }
        )
        assert create_response.status_code == status.HTTP_200_OK
        batch_id = create_response.json()["batch_id"]
        
        # 2. Obtener estado
        status_response = client.get(f"/batch/{batch_id}")
        assert status_response.status_code == status.HTTP_200_OK
        
        # 3. Procesar batch
        process_response = client.post(f"/batch/{batch_id}/process")
        assert process_response.status_code == status.HTTP_200_OK
        
        # 4. Cancelar batch (opcional)
        cancel_response = client.post(f"/batch/{batch_id}/cancel")
        assert cancel_response.status_code == status.HTTP_200_OK



