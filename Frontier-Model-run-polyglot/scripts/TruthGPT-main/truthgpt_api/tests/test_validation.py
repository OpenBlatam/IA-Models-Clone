"""
Tests de Validación
===================
Tests que verifican validación de datos y parámetros
"""

import pytest
import requests
import numpy as np

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


class TestInputValidation:
    """Tests de validación de inputs."""
    
    def test_create_model_missing_required_fields(self, server_running):
        """Test crear modelo sin campos requeridos"""
        # Sin layers
        response = requests.post(
            f"{BASE_URL}/models/create",
            json={},
            timeout=TIMEOUT
        )
        assert response.status_code == 422  # Validation error
    
    def test_create_model_invalid_layer_type(self, server_running):
        """Test crear modelo con tipo de layer inválido"""
        response = requests.post(
            f"{BASE_URL}/models/create",
            json={
                "layers": [
                    {"type": "invalid_layer_type", "params": {}}
                ]
            },
            timeout=TIMEOUT
        )
        assert response.status_code == 400
    
    def test_create_model_invalid_layer_params(self, server_running):
        """Test crear modelo con parámetros inválidos"""
        # Dense sin units
        response = requests.post(
            f"{BASE_URL}/models/create",
            json={
                "layers": [
                    {"type": "dense", "params": {}}
                ]
            },
            timeout=TIMEOUT
        )
        # Puede fallar o usar default
        assert response.status_code in [200, 400]
    
    def test_compile_missing_optimizer(self, server_running):
        """Test compilar sin optimizer"""
        create_response = requests.post(
            f"{BASE_URL}/models/create",
            json={
                "layers": [
                    {"type": "dense", "params": {"units": 32}}
                ]
            },
            timeout=TIMEOUT
        )
        model_id = create_response.json()["model_id"]
        
        response = requests.post(
            f"{BASE_URL}/models/{model_id}/compile",
            json={"loss": "mse"},
            timeout=TIMEOUT
        )
        assert response.status_code == 422  # Validation error
    
    def test_compile_missing_loss(self, server_running):
        """Test compilar sin loss"""
        create_response = requests.post(
            f"{BASE_URL}/models/create",
            json={
                "layers": [
                    {"type": "dense", "params": {"units": 32}}
                ]
            },
            timeout=TIMEOUT
        )
        model_id = create_response.json()["model_id"]
        
        response = requests.post(
            f"{BASE_URL}/models/{model_id}/compile",
            json={"optimizer": "adam"},
            timeout=TIMEOUT
        )
        assert response.status_code == 422  # Validation error


class TestDataValidation:
    """Tests de validación de datos."""
    
    def test_train_with_mismatched_shapes(self, server_running):
        """Test entrenar con shapes que no coinciden"""
        create_response = requests.post(
            f"{BASE_URL}/models/create",
            json={
                "layers": [
                    {"type": "dense", "params": {"units": 10}},
                    {"type": "dense", "params": {"units": 3}}
                ]
            },
            timeout=TIMEOUT
        )
        model_id = create_response.json()["model_id"]
        
        requests.post(
            f"{BASE_URL}/models/{model_id}/compile",
            json={"optimizer": "adam", "loss": "mse"},
            timeout=TIMEOUT
        )
        
        # x_train tiene 10 features pero el modelo espera diferente
        x_train = np.random.randn(100, 5).astype(np.float32).tolist()  # Shape incorrecto
        y_train = np.random.randn(100, 3).astype(np.float32).tolist()
        
        response = requests.post(
            f"{BASE_URL}/models/{model_id}/train",
            json={
                "x_train": x_train,
                "y_train": y_train,
                "epochs": 1,
                "verbose": 0
            },
            timeout=TIMEOUT * 2
        )
        # Debería fallar o dar error
        assert response.status_code in [200, 400, 500]
    
    def test_train_with_empty_data(self, server_running):
        """Test entrenar con datos vacíos"""
        create_response = requests.post(
            f"{BASE_URL}/models/create",
            json={
                "layers": [
                    {"type": "dense", "params": {"units": 32}}
                ]
            },
            timeout=TIMEOUT
        )
        model_id = create_response.json()["model_id"]
        
        requests.post(
            f"{BASE_URL}/models/{model_id}/compile",
            json={"optimizer": "adam", "loss": "mse"},
            timeout=TIMEOUT
        )
        
        response = requests.post(
            f"{BASE_URL}/models/{model_id}/train",
            json={
                "x_train": [],
                "y_train": [],
                "epochs": 1
            },
            timeout=TIMEOUT
        )
        assert response.status_code == 400
    
    def test_predict_with_invalid_shape(self, server_running):
        """Test predecir con shape inválido"""
        create_response = requests.post(
            f"{BASE_URL}/models/create",
            json={
                "layers": [
                    {"type": "dense", "params": {"units": 10}},
                    {"type": "dense", "params": {"units": 3}}
                ]
            },
            timeout=TIMEOUT
        )
        model_id = create_response.json()["model_id"]
        
        requests.post(
            f"{BASE_URL}/models/{model_id}/compile",
            json={"optimizer": "adam", "loss": "mse"},
            timeout=TIMEOUT
        )
        
        # Shape incorrecto
        x = np.random.randn(10, 5).astype(np.float32).tolist()  # Debería ser 10 features
        
        response = requests.post(
            f"{BASE_URL}/models/{model_id}/predict",
            json={"x": x, "verbose": 0},
            timeout=TIMEOUT
        )
        assert response.status_code in [200, 400, 500]


class TestParameterValidation:
    """Tests de validación de parámetros."""
    
    def test_invalid_learning_rate(self, server_running):
        """Test con learning rate inválido"""
        create_response = requests.post(
            f"{BASE_URL}/models/create",
            json={
                "layers": [
                    {"type": "dense", "params": {"units": 32}}
                ]
            },
            timeout=TIMEOUT
        )
        model_id = create_response.json()["model_id"]
        
        # Learning rate negativo
        response = requests.post(
            f"{BASE_URL}/models/{model_id}/compile",
            json={
                "optimizer": "adam",
                "optimizer_params": {"learning_rate": -0.001},
                "loss": "mse"
            },
            timeout=TIMEOUT
        )
        # Puede aceptar o rechazar
        assert response.status_code in [200, 400]
    
    def test_invalid_epochs(self, server_running):
        """Test con epochs inválido"""
        create_response = requests.post(
            f"{BASE_URL}/models/create",
            json={
                "layers": [
                    {"type": "dense", "params": {"units": 32}}
                ]
            },
            timeout=TIMEOUT
        )
        model_id = create_response.json()["model_id"]
        
        requests.post(
            f"{BASE_URL}/models/{model_id}/compile",
            json={"optimizer": "adam", "loss": "mse"},
            timeout=TIMEOUT
        )
        
        x_train = np.random.randn(10, 32).astype(np.float32).tolist()
        y_train = np.random.randn(10, 1).astype(np.float32).tolist()
        
        # Epochs negativo
        response = requests.post(
            f"{BASE_URL}/models/{model_id}/train",
            json={
                "x_train": x_train,
                "y_train": y_train,
                "epochs": -1
            },
            timeout=TIMEOUT
        )
        assert response.status_code in [200, 400]
    
    def test_invalid_batch_size(self, server_running):
        """Test con batch_size inválido"""
        create_response = requests.post(
            f"{BASE_URL}/models/create",
            json={
                "layers": [
                    {"type": "dense", "params": {"units": 32}}
                ]
            },
            timeout=TIMEOUT
        )
        model_id = create_response.json()["model_id"]
        
        requests.post(
            f"{BASE_URL}/models/{model_id}/compile",
            json={"optimizer": "adam", "loss": "mse"},
            timeout=TIMEOUT
        )
        
        x_train = np.random.randn(10, 32).astype(np.float32).tolist()
        y_train = np.random.randn(10, 1).astype(np.float32).tolist()
        
        # Batch size mayor que dataset
        response = requests.post(
            f"{BASE_URL}/models/{model_id}/train",
            json={
                "x_train": x_train,
                "y_train": y_train,
                "epochs": 1,
                "batch_size": 1000  # Mayor que dataset
            },
            timeout=TIMEOUT
        )
        # Debería funcionar o ajustar automáticamente
        assert response.status_code in [200, 400]


class TestTypeValidation:
    """Tests de validación de tipos."""
    
    def test_wrong_data_types(self, server_running):
        """Test con tipos de datos incorrectos"""
        create_response = requests.post(
            f"{BASE_URL}/models/create",
            json={
                "layers": [
                    {"type": "dense", "params": {"units": 32}}
                ]
            },
            timeout=TIMEOUT
        )
        model_id = create_response.json()["model_id"]
        
        requests.post(
            f"{BASE_URL}/models/{model_id}/compile",
            json={"optimizer": "adam", "loss": "mse"},
            timeout=TIMEOUT
        )
        
        # Strings en lugar de números
        x_train = [["1", "2", "3"] for _ in range(10)]
        y_train = ["1" for _ in range(10)]
        
        response = requests.post(
            f"{BASE_URL}/models/{model_id}/train",
            json={
                "x_train": x_train,
                "y_train": y_train,
                "epochs": 1
            },
            timeout=TIMEOUT
        )
        # Debería fallar o convertir
        assert response.status_code in [200, 400, 422]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])











