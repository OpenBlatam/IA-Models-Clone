"""
Tests para las rutas de A/B Testing
"""

import pytest
from unittest.mock import Mock, patch
from fastapi import status
from fastapi.testclient import TestClient

from api.routes.ab_testing import router
from services.ab_testing import ABTestingService


@pytest.fixture
def mock_ab_testing_service():
    """Mock del servicio de A/B testing"""
    service = Mock(spec=ABTestingService)
    service.create_experiment = Mock(return_value="experiment-123")
    service.assign_variant = Mock(return_value="control")
    service.record_result = Mock(return_value=True)
    service.analyze_experiment = Mock(return_value={
        "control": {"conversions": 100, "total": 1000},
        "variant_a": {"conversions": 120, "total": 1000}
    })
    return service


@pytest.fixture
def client(mock_ab_testing_service):
    """Cliente de prueba con mocks"""
    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(router)
    
    with patch('api.routes.ab_testing.get_ab_testing_service', return_value=mock_ab_testing_service):
        with patch('api.routes.ab_testing.require_role', return_value=lambda: None):
            with patch('api.routes.ab_testing.get_current_user', return_value={"user_id": "test_user"}):
                yield TestClient(app)


@pytest.mark.unit
@pytest.mark.api
class TestCreateExperiment:
    """Tests para crear experimento"""
    
    def test_create_experiment_success(self, client, mock_ab_testing_service):
        """Test de creación exitosa"""
        response = client.post(
            "/ab-testing/experiments",
            json={
                "name": "Test Experiment",
                "variants": ["control", "variant_a"],
                "description": "Test description"
            }
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "experiment_id" in data
        assert data["message"] == "Experiment created successfully"
    
    def test_create_experiment_with_traffic_split(self, client, mock_ab_testing_service):
        """Test con división de tráfico"""
        response = client.post(
            "/ab-testing/experiments",
            json={
                "name": "Test Experiment",
                "variants": ["control", "variant_a"],
                "traffic_split": {"control": 0.5, "variant_a": 0.5}
            }
        )
        
        assert response.status_code == status.HTTP_200_OK


@pytest.mark.unit
@pytest.mark.api
class TestAssignVariant:
    """Tests para asignar variante"""
    
    def test_assign_variant_success(self, client, mock_ab_testing_service):
        """Test de asignación exitosa"""
        response = client.get("/ab-testing/experiments/experiment-123/assign?user_id=user-456")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "variant" in data or "assigned_variant" in data
    
    def test_assign_variant_with_force(self, client, mock_ab_testing_service):
        """Test con variante forzada"""
        response = client.get(
            "/ab-testing/experiments/experiment-123/assign",
            params={"user_id": "user-456", "force_variant": "variant_a"}
        )
        
        assert response.status_code == status.HTTP_200_OK


@pytest.mark.unit
@pytest.mark.api
class TestRecordResult:
    """Tests para registrar resultado"""
    
    def test_record_result_success(self, client, mock_ab_testing_service):
        """Test de registro exitoso"""
        response = client.post(
            "/ab-testing/experiments/experiment-123/results",
            json={
                "user_id": "user-456",
                "variant": "control",
                "converted": True
            }
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "message" in data or "success" in data


@pytest.mark.unit
@pytest.mark.api
class TestAnalyzeExperiment:
    """Tests para analizar experimento"""
    
    def test_analyze_experiment_success(self, client, mock_ab_testing_service):
        """Test de análisis exitoso"""
        response = client.get("/ab-testing/experiments/experiment-123/analyze")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "results" in data or "analysis" in data


@pytest.mark.integration
@pytest.mark.api
class TestABTestingIntegration:
    """Tests de integración para A/B testing"""
    
    def test_full_ab_testing_workflow(self, client, mock_ab_testing_service):
        """Test del flujo completo de A/B testing"""
        # 1. Crear experimento
        create_response = client.post(
            "/ab-testing/experiments",
            json={
                "name": "Full Test Experiment",
                "variants": ["control", "variant_a"]
            }
        )
        assert create_response.status_code == status.HTTP_200_OK
        experiment_id = create_response.json()["experiment_id"]
        
        # 2. Asignar variante
        assign_response = client.get(
            f"/ab-testing/experiments/{experiment_id}/assign",
            params={"user_id": "user-456"}
        )
        assert assign_response.status_code == status.HTTP_200_OK
        
        # 3. Registrar resultado
        record_response = client.post(
            f"/ab-testing/experiments/{experiment_id}/results",
            json={"user_id": "user-456", "variant": "control", "converted": True}
        )
        assert record_response.status_code == status.HTTP_200_OK
        
        # 4. Analizar experimento
        analyze_response = client.get(f"/ab-testing/experiments/{experiment_id}/analyze")
        assert analyze_response.status_code == status.HTTP_200_OK



