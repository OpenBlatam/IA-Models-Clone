"""
Tests para las rutas de hyperparameter tuning
"""

import pytest
from unittest.mock import Mock, patch
from fastapi import status
from fastapi.testclient import TestClient

from api.routes.hyperparameter_tuning import router
from services.hyperparameter_tuner import HyperparameterTuner


@pytest.fixture
def mock_hyperparameter_tuner():
    """Mock del servicio de hyperparameter tuning"""
    tuner = Mock(spec=HyperparameterTuner)
    
    # Mock de trial
    trial = Mock()
    trial.hyperparameters = {"learning_rate": 0.001, "batch_size": 32}
    trial.metrics = {"score": 0.95, "loss": 0.05}
    
    tuner.grid_search = Mock(return_value=trial)
    tuner.random_search = Mock(return_value=trial)
    tuner.get_best_trial = Mock(return_value=trial)
    tuner.get_trial_history = Mock(return_value=[trial, trial])
    tuner.trials = [trial, trial]
    return tuner


@pytest.fixture
def client(mock_hyperparameter_tuner):
    """Cliente de prueba con mocks"""
    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(router)
    
    with patch('api.routes.hyperparameter_tuning.get_hyperparameter_tuner', return_value=mock_hyperparameter_tuner):
        with patch('api.routes.hyperparameter_tuning.require_role', return_value=lambda: None):
            yield TestClient(app)


@pytest.mark.unit
@pytest.mark.api
class TestGridSearch:
    """Tests para grid search"""
    
    def test_grid_search_success(self, client, mock_hyperparameter_tuner):
        """Test de grid search exitoso"""
        response = client.post(
            "/hyperparameter-tuning/grid-search",
            json={
                "hyperparameters": {
                    "learning_rate": {
                        "type": "choice",
                        "values": [0.001, 0.01, 0.1]
                    },
                    "batch_size": {
                        "type": "choice",
                        "values": [16, 32, 64]
                    }
                }
            }
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "message" in data
        assert "best_trial" in data
        assert "total_trials" in data


@pytest.mark.unit
@pytest.mark.api
class TestRandomSearch:
    """Tests para random search"""
    
    def test_random_search_success(self, client, mock_hyperparameter_tuner):
        """Test de random search exitoso"""
        response = client.post(
            "/hyperparameter-tuning/random-search",
            json={
                "hyperparameters": {
                    "learning_rate": {
                        "type": "uniform",
                        "min_value": 0.001,
                        "max_value": 0.1
                    }
                },
                "max_trials": 10
            }
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "message" in data
        assert "best_trial" in data


@pytest.mark.unit
@pytest.mark.api
class TestGetBestTrial:
    """Tests para obtener mejor trial"""
    
    def test_get_best_trial_success(self, client, mock_hyperparameter_tuner):
        """Test de obtención exitosa"""
        response = client.get("/hyperparameter-tuning/best-trial")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "trial" in data or "best_trial" in data


@pytest.mark.unit
@pytest.mark.api
class TestGetTrialHistory:
    """Tests para obtener historial de trials"""
    
    def test_get_trial_history_success(self, client, mock_hyperparameter_tuner):
        """Test de obtención exitosa"""
        response = client.get("/hyperparameter-tuning/trials")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "trials" in data or isinstance(data, list)


@pytest.mark.integration
@pytest.mark.api
class TestHyperparameterTuningIntegration:
    """Tests de integración para hyperparameter tuning"""
    
    def test_full_tuning_workflow(self, client, mock_hyperparameter_tuner):
        """Test del flujo completo de tuning"""
        # 1. Grid search
        grid_response = client.post(
            "/hyperparameter-tuning/grid-search",
            json={
                "hyperparameters": {
                    "learning_rate": {
                        "type": "choice",
                        "values": [0.001, 0.01]
                    }
                }
            }
        )
        assert grid_response.status_code == status.HTTP_200_OK
        
        # 2. Obtener mejor trial
        best_response = client.get("/hyperparameter-tuning/best-trial")
        assert best_response.status_code == status.HTTP_200_OK
        
        # 3. Obtener historial
        history_response = client.get("/hyperparameter-tuning/trials")
        assert history_response.status_code == status.HTTP_200_OK



