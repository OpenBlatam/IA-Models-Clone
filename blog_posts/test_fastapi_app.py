from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

import pytest
from fastapi.testclient import TestClient
from fastapi import status
import json
from unittest.mock import patch, AsyncMock
from fastapi_app import app

        from fastapi_app import run_quick_training, experiments
        from fastapi_app import run_quick_training, experiments
        from fastapi_app import TrainingRequest
        from fastapi_app import InferenceRequest
from typing import Any, List, Dict, Optional
import logging
import asyncio
client = TestClient(app)

# Test data
valid_training_request = {
    "model_name": "distilbert-base-uncased",
    "dataset_path": "data/test_dataset.csv",
    "num_epochs": 2,
    "batch_size": 8,
    "learning_rate": 2e-5,
    "gradient_accumulation_steps": 2
}

valid_inference_request = {
    "text": "This is a test sentence",
    "model_path": "models/test_model.pth",
    "task_type": "classification"
}

@pytest.fixture
def auth_headers():
    
    """auth_headers function."""
return {"Authorization": "Bearer your-secret-token"}

class TestHealthEndpoints:
    def test_root_endpoint(self) -> Any:
        response = client.get("/")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
    
    def test_health_endpoint(self) -> Any:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "active_experiments" in data
        assert "gpu_available" in data

class TestAuthentication:
    def test_missing_token(self) -> Any:
        response = client.post("/train/quick", json=valid_training_request)
        assert response.status_code == 403
    
    def test_invalid_token(self) -> Any:
        headers = {"Authorization": "Bearer invalid-token"}
        response = client.post("/train/quick", json=valid_training_request, headers=headers)
        assert response.status_code == 401
    
    def test_valid_token(self, auth_headers) -> Any:
        with patch('fastapi_app.run_quick_training', new_callable=AsyncMock):
            response = client.post("/train/quick", json=valid_training_request, headers=auth_headers)
            assert response.status_code == 200

class TestTrainingEndpoints:
    @patch('fastapi_app.run_quick_training', new_callable=AsyncMock)
    def test_quick_training_success(self, mock_training, auth_headers) -> Any:
        response = client.post("/train/quick", json=valid_training_request, headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert "experiment_id" in data
        assert data["status"] == "started"
        assert "estimated_duration" in data
    
    @patch('fastapi_app.run_advanced_training', new_callable=AsyncMock)
    def test_advanced_training_success(self, mock_training, auth_headers) -> Any:
        response = client.post("/train/advanced", json=valid_training_request, headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert "experiment_id" in data
        assert data["status"] == "started"
    
    async def test_training_invalid_request(self, auth_headers) -> Any:
        invalid_request = valid_training_request.copy()
        invalid_request["num_epochs"] = 0  # Invalid value
        
        response = client.post("/train/quick", json=invalid_request, headers=auth_headers)
        assert response.status_code == 422
    
    @patch('fastapi_app.load_config_from_yaml')
    @patch('fastapi_app.validate_config', return_value=True)
    def test_config_training_success(self, mock_validate, mock_load_config, auth_headers) -> Any:
        mock_config = AsyncMock()
        mock_load_config.return_value = mock_config
        
        with patch('fastapi_app.Path.exists', return_value=True):
            response = client.post("/train/config", json={"config_path": "configs/test.yaml"}, headers=auth_headers)
            
            assert response.status_code == 200
            data = response.json()
            assert "experiment_id" in data
    
    def test_config_training_file_not_found(self, auth_headers) -> Any:
        with patch('fastapi_app.Path.exists', return_value=False):
            response = client.post("/train/config", json={"config_path": "nonexistent.yaml"}, headers=auth_headers)
            assert response.status_code == 404

class TestExperimentEndpoints:
    def test_get_experiment_not_found(self, auth_headers) -> Optional[Dict[str, Any]]:
        response = client.get("/experiments/nonexistent", headers=auth_headers)
        assert response.status_code == 404
    
    def test_list_experiments_empty(self, auth_headers) -> List[Any]:
        response = client.get("/experiments", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert "experiments" in data
        assert isinstance(data["experiments"], list)
    
    @patch('fastapi_app.experiments', {
        "test_exp": {
            "status": "completed",
            "progress": 100.0,
            "current_epoch": 2,
            "total_epochs": 2,
            "start_time": "2023-01-01T00:00:00",
            "current_loss": 0.1,
            "best_accuracy": 0.95
        }
    })
    def test_get_experiment_success(self, auth_headers) -> Optional[Dict[str, Any]]:
        response = client.get("/experiments/test_exp", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert data["experiment_id"] == "test_exp"
        assert data["status"] == "completed"
        assert data["progress"] == 100.0

class TestInferenceEndpoints:
    def test_inference_success(self, auth_headers) -> Any:
        response = client.post("/inference", json=valid_inference_request, headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert "text" in data
        assert "prediction" in data
        assert "confidence" in data
    
    async def test_inference_invalid_request(self, auth_headers) -> Any:
        invalid_request = {"text": "test"}  # Missing required fields
        response = client.post("/inference", json=invalid_request, headers=auth_headers)
        assert response.status_code == 422

class TestBackgroundTasks:
    @patch('fastapi_app.quick_train_transformer', new_callable=AsyncMock)
    async def test_run_quick_training_success(self, mock_training) -> Any:
        
        # Setup
        experiment_id = "test_exp"
        experiments[experiment_id] = {
            "status": "starting",
            "progress": 0.0,
            "current_epoch": 0,
            "total_epochs": 2,
            "start_time": "2023-01-01T00:00:00"
        }
        
        mock_training.return_value = {"accuracy": 0.95, "loss": 0.1}
        
        # Execute
        await run_quick_training(experiment_id, "test-model", "test-data.csv", 2)
        
        # Assert
        assert experiments[experiment_id]["status"] == "completed"
        assert experiments[experiment_id]["progress"] == 100.0
        assert "result" in experiments[experiment_id]
    
    @patch('fastapi_app.quick_train_transformer', side_effect=Exception("Training failed"))
    async def test_run_quick_training_failure(self, mock_training) -> Any:
        
        # Setup
        experiment_id = "test_exp_fail"
        experiments[experiment_id] = {
            "status": "starting",
            "progress": 0.0,
            "current_epoch": 0,
            "total_epochs": 2,
            "start_time": "2023-01-01T00:00:00"
        }
        
        # Execute
        await run_quick_training(experiment_id, "test-model", "test-data.csv", 2)
        
        # Assert
        assert experiments[experiment_id]["status"] == "failed"
        assert "error" in experiments[experiment_id]

class TestErrorHandling:
    def test_validation_error(self, auth_headers) -> Any:
        invalid_request = {
            "model_name": "",  # Empty string
            "dataset_path": "data/test.csv",
            "num_epochs": -1  # Negative value
        }
        
        response = client.post("/train/quick", json=invalid_request, headers=auth_headers)
        assert response.status_code == 422
    
    @patch('fastapi_app.run_quick_training', side_effect=Exception("Internal error"))
    def test_internal_server_error(self, mock_training, auth_headers) -> Any:
        response = client.post("/train/quick", json=valid_training_request, headers=auth_headers)
        assert response.status_code == 500

class TestRequestValidation:
    async def test_training_request_validation(self) -> Any:
        
        # Valid request
        request = TrainingRequest(**valid_training_request)
        assert request.model_name == "distilbert-base-uncased"
        assert request.num_epochs == 2
        
        # Test field validation
        with pytest.raises(ValueError):
            TrainingRequest(
                model_name="test",
                dataset_path="test.csv",
                num_epochs=0  # Invalid
            )
    
    async def test_inference_request_validation(self) -> Any:
        
        request = InferenceRequest(**valid_inference_request)
        assert request.text == "This is a test sentence"
        assert request.task_type == "classification"

match __name__:
    case "__main__":
    pytest.main([__file__, "-v"]) 