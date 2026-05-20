"""
Tests completos para la API TruthGPT
====================================
"""

import pytest
import requests
import numpy as np
import time
from typing import Dict, Any

BASE_URL = "http://localhost:8000"
TIMEOUT = 10

@pytest.fixture(scope="module")
def server_running():
    """Verifica que el servidor esté corriendo."""
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            return True
    except:
        pass
    pytest.skip("Servidor no está corriendo. Ejecuta: python start_server.py")


class TestHealthCheck:
    """Tests para el health check."""
    
    def test_health_endpoint(self, server_running):
        """Test del endpoint /health"""
        response = requests.get(f"{BASE_URL}/health", timeout=TIMEOUT)
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"
        assert "cuda_available" in data
        assert "device" in data
    
    def test_root_endpoint(self, server_running):
        """Test del endpoint raíz"""
        response = requests.get(f"{BASE_URL}/", timeout=TIMEOUT)
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data


class TestModelCreation:
    """Tests para creación de modelos."""
    
    def test_create_model_simple(self, server_running):
        """Test crear un modelo simple"""
        response = requests.post(
            f"{BASE_URL}/models/create",
            json={
                "layers": [
                    {"type": "dense", "params": {"units": 64, "activation": "relu"}},
                    {"type": "dense", "params": {"units": 10, "activation": "softmax"}}
                ]
            },
            timeout=TIMEOUT
        )
        assert response.status_code == 200
        data = response.json()
        assert "model_id" in data
        assert "name" in data
        assert "status" in data
        assert data["status"] == "created"
        return data["model_id"]
    
    def test_create_model_with_name(self, server_running):
        """Test crear modelo con nombre personalizado"""
        response = requests.post(
            f"{BASE_URL}/models/create",
            json={
                "layers": [
                    {"type": "dense", "params": {"units": 32, "activation": "relu"}}
                ],
                "name": "test-model-custom"
            },
            timeout=TIMEOUT
        )
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "test-model-custom"
        return data["model_id"]
    
    def test_create_model_invalid_layer(self, server_running):
        """Test crear modelo con layer inválido"""
        response = requests.post(
            f"{BASE_URL}/models/create",
            json={
                "layers": [
                    {"type": "invalid_layer", "params": {}}
                ]
            },
            timeout=TIMEOUT
        )
        assert response.status_code == 400
    
    def test_create_model_empty_layers(self, server_running):
        """Test crear modelo sin layers"""
        response = requests.post(
            f"{BASE_URL}/models/create",
            json={"layers": []},
            timeout=TIMEOUT
        )
        # Debería crear un modelo vacío o dar error
        assert response.status_code in [200, 400]


class TestModelCompilation:
    """Tests para compilación de modelos."""
    
    def test_compile_model(self, server_running):
        """Test compilar un modelo"""
        # Primero crear un modelo
        create_response = requests.post(
            f"{BASE_URL}/models/create",
            json={
                "layers": [
                    {"type": "dense", "params": {"units": 64, "activation": "relu"}},
                    {"type": "dense", "params": {"units": 10, "activation": "softmax"}}
                ]
            },
            timeout=TIMEOUT
        )
        model_id = create_response.json()["model_id"]
        
        # Compilar el modelo
        compile_response = requests.post(
            f"{BASE_URL}/models/{model_id}/compile",
            json={
                "optimizer": "adam",
                "optimizer_params": {"learning_rate": 0.001},
                "loss": "sparsecategoricalcrossentropy",
                "metrics": ["accuracy"]
            },
            timeout=TIMEOUT
        )
        assert compile_response.status_code == 200
        data = compile_response.json()
        assert data["status"] == "compiled"
    
    def test_compile_nonexistent_model(self, server_running):
        """Test compilar modelo que no existe"""
        response = requests.post(
            f"{BASE_URL}/models/nonexistent-id/compile",
            json={
                "optimizer": "adam",
                "loss": "sparsecategoricalcrossentropy"
            },
            timeout=TIMEOUT
        )
        assert response.status_code == 404


class TestModelTraining:
    """Tests para entrenamiento de modelos."""
    
    def test_train_model(self, server_running):
        """Test entrenar un modelo"""
        # Crear y compilar modelo
        create_response = requests.post(
            f"{BASE_URL}/models/create",
            json={
                "layers": [
                    {"type": "dense", "params": {"units": 32, "activation": "relu"}},
                    {"type": "dense", "params": {"units": 3, "activation": "softmax"}}
                ]
            },
            timeout=TIMEOUT
        )
        model_id = create_response.json()["model_id"]
        
        compile_response = requests.post(
            f"{BASE_URL}/models/{model_id}/compile",
            json={
                "optimizer": "adam",
                "optimizer_params": {"learning_rate": 0.001},
                "loss": "sparsecategoricalcrossentropy",
                "metrics": ["accuracy"]
            },
            timeout=TIMEOUT
        )
        assert compile_response.status_code == 200
        
        # Generar datos de entrenamiento
        x_train = np.random.randn(100, 10).astype(np.float32).tolist()
        y_train = np.random.randint(0, 3, 100).astype(np.int64).tolist()
        
        # Entrenar
        train_response = requests.post(
            f"{BASE_URL}/models/{model_id}/train",
            json={
                "x_train": x_train,
                "y_train": y_train,
                "epochs": 1,
                "batch_size": 32,
                "verbose": 0
            },
            timeout=TIMEOUT * 2  # Training puede tomar más tiempo
        )
        assert train_response.status_code == 200
        data = train_response.json()
        assert data["status"] == "trained"
        assert "history" in data


class TestModelEvaluation:
    """Tests para evaluación de modelos."""
    
    def test_evaluate_model(self, server_running):
        """Test evaluar un modelo"""
        # Crear, compilar y entrenar modelo
        create_response = requests.post(
            f"{BASE_URL}/models/create",
            json={
                "layers": [
                    {"type": "dense", "params": {"units": 32, "activation": "relu"}},
                    {"type": "dense", "params": {"units": 3, "activation": "softmax"}}
                ]
            },
            timeout=TIMEOUT
        )
        model_id = create_response.json()["model_id"]
        
        requests.post(
            f"{BASE_URL}/models/{model_id}/compile",
            json={
                "optimizer": "adam",
                "loss": "sparsecategoricalcrossentropy",
                "metrics": ["accuracy"]
            },
            timeout=TIMEOUT
        )
        
        x_train = np.random.randn(50, 10).astype(np.float32).tolist()
        y_train = np.random.randint(0, 3, 50).astype(np.int64).tolist()
        
        requests.post(
            f"{BASE_URL}/models/{model_id}/train",
            json={
                "x_train": x_train,
                "y_train": y_train,
                "epochs": 1,
                "verbose": 0
            },
            timeout=TIMEOUT * 2
        )
        
        # Evaluar
        x_test = np.random.randn(20, 10).astype(np.float32).tolist()
        y_test = np.random.randint(0, 3, 20).astype(np.int64).tolist()
        
        eval_response = requests.post(
            f"{BASE_URL}/models/{model_id}/evaluate",
            json={
                "x_test": x_test,
                "y_test": y_test,
                "verbose": 0
            },
            timeout=TIMEOUT
        )
        assert eval_response.status_code == 200
        data = eval_response.json()
        assert data["status"] == "evaluated"
        assert "results" in data


class TestModelPrediction:
    """Tests para predicciones."""
    
    def test_predict(self, server_running):
        """Test hacer predicciones"""
        # Crear y compilar modelo
        create_response = requests.post(
            f"{BASE_URL}/models/create",
            json={
                "layers": [
                    {"type": "dense", "params": {"units": 32, "activation": "relu"}},
                    {"type": "dense", "params": {"units": 3, "activation": "softmax"}}
                ]
            },
            timeout=TIMEOUT
        )
        model_id = create_response.json()["model_id"]
        
        requests.post(
            f"{BASE_URL}/models/{model_id}/compile",
            json={
                "optimizer": "adam",
                "loss": "sparsecategoricalcrossentropy"
            },
            timeout=TIMEOUT
        )
        
        # Hacer predicción
        x = np.random.randn(5, 10).astype(np.float32).tolist()
        
        predict_response = requests.post(
            f"{BASE_URL}/models/{model_id}/predict",
            json={"x": x, "verbose": 0},
            timeout=TIMEOUT
        )
        assert predict_response.status_code == 200
        data = predict_response.json()
        assert data["status"] == "predicted"
        assert "predictions" in data
        assert len(data["predictions"]) == 5


class TestModelManagement:
    """Tests para gestión de modelos."""
    
    def test_list_models(self, server_running):
        """Test listar modelos"""
        response = requests.get(f"{BASE_URL}/models", timeout=TIMEOUT)
        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        assert "count" in data
        assert isinstance(data["models"], list)
    
    def test_get_model_info(self, server_running):
        """Test obtener información de un modelo"""
        # Crear modelo
        create_response = requests.post(
            f"{BASE_URL}/models/create",
            json={
                "layers": [
                    {"type": "dense", "params": {"units": 32, "activation": "relu"}}
                ]
            },
            timeout=TIMEOUT
        )
        model_id = create_response.json()["model_id"]
        
        # Obtener info
        info_response = requests.get(
            f"{BASE_URL}/models/{model_id}",
            timeout=TIMEOUT
        )
        assert info_response.status_code == 200
        data = info_response.json()
        assert data["model_id"] == model_id
        assert "name" in data
        assert "compiled" in data
    
    def test_get_nonexistent_model(self, server_running):
        """Test obtener info de modelo que no existe"""
        response = requests.get(
            f"{BASE_URL}/models/nonexistent-id",
            timeout=TIMEOUT
        )
        assert response.status_code == 404
    
    def test_delete_model(self, server_running):
        """Test eliminar un modelo"""
        # Crear modelo
        create_response = requests.post(
            f"{BASE_URL}/models/create",
            json={
                "layers": [
                    {"type": "dense", "params": {"units": 32, "activation": "relu"}}
                ]
            },
            timeout=TIMEOUT
        )
        model_id = create_response.json()["model_id"]
        
        # Eliminar
        delete_response = requests.delete(
            f"{BASE_URL}/models/{model_id}",
            timeout=TIMEOUT
        )
        assert delete_response.status_code == 200
        
        # Verificar que fue eliminado
        info_response = requests.get(
            f"{BASE_URL}/models/{model_id}",
            timeout=TIMEOUT
        )
        assert info_response.status_code == 404


if __name__ == "__main__":
    pytest.main([__file__, "-v"])











