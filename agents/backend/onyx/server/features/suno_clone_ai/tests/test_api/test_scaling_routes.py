"""
Tests para las rutas de auto-scaling
"""

import pytest
from unittest.mock import Mock, patch
from fastapi import status
from fastapi.testclient import TestClient

from api.routes.scaling import router
from services.auto_scaler import AutoScaler, ScalingPolicy, ScalingAction


@pytest.fixture
def mock_auto_scaler():
    """Mock del auto-scaler"""
    scaler = Mock(spec=AutoScaler)
    
    # Mock de decisión de escalado
    decision = Mock()
    decision.action = ScalingAction.SCALE_UP
    decision.current_replicas = 2
    decision.target_replicas = 3
    decision.reason = "CPU above threshold"
    
    scaler.add_policy = Mock(return_value=True)
    scaler.evaluate = Mock(return_value=decision)
    scaler.get_stats = Mock(return_value={
        "current_replicas": 2,
        "policies_count": 2
    })
    return scaler


@pytest.fixture
def client(mock_auto_scaler):
    """Cliente de prueba con mocks"""
    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(router)
    
    with patch('api.routes.scaling.get_auto_scaler', return_value=mock_auto_scaler):
        with patch('api.routes.scaling.require_role', return_value=lambda: None):
            yield TestClient(app)


@pytest.mark.unit
@pytest.mark.api
class TestAddScalingPolicy:
    """Tests para agregar política de escalado"""
    
    def test_add_policy_success(self, client, mock_auto_scaler):
        """Test de agregado exitoso"""
        response = client.post(
            "/scaling/policies",
            json={
                "name": "cpu_policy",
                "metric": "cpu",
                "threshold_up": 80.0,
                "threshold_down": 30.0,
                "min_replicas": 1,
                "max_replicas": 10
            }
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "message" in data
        assert "policy" in data
    
    def test_add_policy_different_metrics(self, client, mock_auto_scaler):
        """Test con diferentes métricas"""
        metrics = ["cpu", "memory", "requests", "queue_size"]
        
        for metric in metrics:
            response = client.post(
                "/scaling/policies",
                json={
                    "name": f"{metric}_policy",
                    "metric": metric,
                    "threshold_up": 80.0,
                    "threshold_down": 30.0
                }
            )
            assert response.status_code == status.HTTP_200_OK


@pytest.mark.unit
@pytest.mark.api
class TestEvaluateScaling:
    """Tests para evaluar escalado"""
    
    def test_evaluate_scaling_success(self, client, mock_auto_scaler):
        """Test de evaluación exitosa"""
        response = client.post("/scaling/evaluate")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "decision" in data or "action" in data


@pytest.mark.unit
@pytest.mark.api
class TestGetStats:
    """Tests para obtener estadísticas"""
    
    def test_get_stats_success(self, client, mock_auto_scaler):
        """Test de obtención exitosa"""
        response = client.get("/scaling/stats")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "stats" in data or isinstance(data, dict)


@pytest.mark.integration
@pytest.mark.api
class TestScalingIntegration:
    """Tests de integración para auto-scaling"""
    
    def test_full_scaling_workflow(self, client, mock_auto_scaler):
        """Test del flujo completo"""
        # 1. Agregar política
        policy_response = client.post(
            "/scaling/policies",
            json={
                "name": "test_policy",
                "metric": "cpu",
                "threshold_up": 80.0,
                "threshold_down": 30.0
            }
        )
        assert policy_response.status_code == status.HTTP_200_OK
        
        # 2. Evaluar escalado
        evaluate_response = client.post("/scaling/evaluate")
        assert evaluate_response.status_code == status.HTTP_200_OK
        
        # 3. Obtener estadísticas
        stats_response = client.get("/scaling/stats")
        assert stats_response.status_code == status.HTTP_200_OK



