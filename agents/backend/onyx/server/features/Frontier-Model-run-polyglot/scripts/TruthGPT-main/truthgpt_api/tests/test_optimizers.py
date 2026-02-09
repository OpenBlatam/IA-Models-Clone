"""
Tests para diferentes optimizers
================================
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


@pytest.fixture
def simple_model(server_running):
    """Crea un modelo simple para tests."""
    response = requests.post(
        f"{BASE_URL}/models/create",
        json={
            "layers": [
                {"type": "dense", "params": {"units": 32, "activation": "relu"}},
                {"type": "dense", "params": {"units": 10, "activation": "softmax"}}
            ]
        },
        timeout=TIMEOUT
    )
    assert response.status_code == 200
    return response.json()["model_id"]


class TestAdamOptimizer:
    """Tests para optimizer Adam."""
    
    def test_adam_basic(self, server_running, simple_model):
        """Test Adam básico"""
        response = requests.post(
            f"{BASE_URL}/models/{simple_model}/compile",
            json={
                "optimizer": "adam",
                "loss": "sparsecategoricalcrossentropy"
            },
            timeout=TIMEOUT
        )
        assert response.status_code == 200
    
    def test_adam_with_learning_rate(self, server_running, simple_model):
        """Test Adam con learning rate"""
        response = requests.post(
            f"{BASE_URL}/models/{simple_model}/compile",
            json={
                "optimizer": "adam",
                "optimizer_params": {"learning_rate": 0.001},
                "loss": "sparsecategoricalcrossentropy"
            },
            timeout=TIMEOUT
        )
        assert response.status_code == 200
    
    def test_adam_with_all_params(self, server_running, simple_model):
        """Test Adam con todos los parámetros"""
        response = requests.post(
            f"{BASE_URL}/models/{simple_model}/compile",
            json={
                "optimizer": "adam",
                "optimizer_params": {
                    "learning_rate": 0.001,
                    "beta_1": 0.9,
                    "beta_2": 0.999
                },
                "loss": "sparsecategoricalcrossentropy"
            },
            timeout=TIMEOUT
        )
        assert response.status_code == 200


class TestSGDOptimizer:
    """Tests para optimizer SGD."""
    
    def test_sgd_basic(self, server_running, simple_model):
        """Test SGD básico"""
        response = requests.post(
            f"{BASE_URL}/models/{simple_model}/compile",
            json={
                "optimizer": "sgd",
                "loss": "sparsecategoricalcrossentropy"
            },
            timeout=TIMEOUT
        )
        assert response.status_code == 200
    
    def test_sgd_with_momentum(self, server_running, simple_model):
        """Test SGD con momentum"""
        response = requests.post(
            f"{BASE_URL}/models/{simple_model}/compile",
            json={
                "optimizer": "sgd",
                "optimizer_params": {
                    "learning_rate": 0.01,
                    "momentum": 0.9
                },
                "loss": "sparsecategoricalcrossentropy"
            },
            timeout=TIMEOUT
        )
        assert response.status_code == 200


class TestRMSpropOptimizer:
    """Tests para optimizer RMSprop."""
    
    def test_rmsprop_basic(self, server_running, simple_model):
        """Test RMSprop básico"""
        response = requests.post(
            f"{BASE_URL}/models/{simple_model}/compile",
            json={
                "optimizer": "rmsprop",
                "loss": "sparsecategoricalcrossentropy"
            },
            timeout=TIMEOUT
        )
        assert response.status_code == 200
    
    def test_rmsprop_with_params(self, server_running, simple_model):
        """Test RMSprop con parámetros"""
        response = requests.post(
            f"{BASE_URL}/models/{simple_model}/compile",
            json={
                "optimizer": "rmsprop",
                "optimizer_params": {
                    "learning_rate": 0.001
                },
                "loss": "sparsecategoricalcrossentropy"
            },
            timeout=TIMEOUT
        )
        assert response.status_code == 200


class TestAdagradOptimizer:
    """Tests para optimizer Adagrad."""
    
    def test_adagrad_basic(self, server_running, simple_model):
        """Test Adagrad básico"""
        response = requests.post(
            f"{BASE_URL}/models/{simple_model}/compile",
            json={
                "optimizer": "adagrad",
                "loss": "sparsecategoricalcrossentropy"
            },
            timeout=TIMEOUT
        )
        assert response.status_code == 200


class TestAdamWOptimizer:
    """Tests para optimizer AdamW."""
    
    def test_adamw_basic(self, server_running, simple_model):
        """Test AdamW básico"""
        response = requests.post(
            f"{BASE_URL}/models/{simple_model}/compile",
            json={
                "optimizer": "adamw",
                "loss": "sparsecategoricalcrossentropy"
            },
            timeout=TIMEOUT
        )
        assert response.status_code == 200


class TestInvalidOptimizers:
    """Tests para optimizers inválidos."""
    
    def test_invalid_optimizer(self, server_running, simple_model):
        """Test optimizer inválido"""
        response = requests.post(
            f"{BASE_URL}/models/{simple_model}/compile",
            json={
                "optimizer": "invalid_optimizer",
                "loss": "sparsecategoricalcrossentropy"
            },
            timeout=TIMEOUT
        )
        assert response.status_code == 400


if __name__ == "__main__":
    pytest.main([__file__, "-v"])











