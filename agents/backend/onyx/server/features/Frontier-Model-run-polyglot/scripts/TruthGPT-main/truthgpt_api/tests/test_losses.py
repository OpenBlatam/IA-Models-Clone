"""
Tests para diferentes loss functions
=====================================
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


class TestSparseCategoricalCrossentropy:
    """Tests para SparseCategoricalCrossentropy."""
    
    def test_sparse_categorical_basic(self, server_running, simple_model):
        """Test básico"""
        response = requests.post(
            f"{BASE_URL}/models/{simple_model}/compile",
            json={
                "optimizer": "adam",
                "loss": "sparsecategoricalcrossentropy"
            },
            timeout=TIMEOUT
        )
        assert response.status_code == 200
    
    def test_sparse_categorical_with_from_logits(self, server_running, simple_model):
        """Test con from_logits"""
        response = requests.post(
            f"{BASE_URL}/models/{simple_model}/compile",
            json={
                "optimizer": "adam",
                "loss": "sparsecategoricalcrossentropy",
                "loss_params": {"from_logits": False}
            },
            timeout=TIMEOUT
        )
        assert response.status_code == 200


class TestCategoricalCrossentropy:
    """Tests para CategoricalCrossentropy."""
    
    def test_categorical_basic(self, server_running, simple_model):
        """Test básico"""
        response = requests.post(
            f"{BASE_URL}/models/{simple_model}/compile",
            json={
                "optimizer": "adam",
                "loss": "categoricalcrossentropy"
            },
            timeout=TIMEOUT
        )
        assert response.status_code == 200


class TestBinaryCrossentropy:
    """Tests para BinaryCrossentropy."""
    
    def test_binary_basic(self, server_running, simple_model):
        """Test básico"""
        response = requests.post(
            f"{BASE_URL}/models/{simple_model}/compile",
            json={
                "optimizer": "adam",
                "loss": "binarycrossentropy"
            },
            timeout=TIMEOUT
        )
        assert response.status_code == 200


class TestMeanSquaredError:
    """Tests para MeanSquaredError."""
    
    def test_mse_basic(self, server_running, simple_model):
        """Test básico"""
        response = requests.post(
            f"{BASE_URL}/models/{simple_model}/compile",
            json={
                "optimizer": "adam",
                "loss": "mse"
            },
            timeout=TIMEOUT
        )
        assert response.status_code == 200
    
    def test_mse_alternative_name(self, server_running, simple_model):
        """Test con nombre alternativo"""
        response = requests.post(
            f"{BASE_URL}/models/{simple_model}/compile",
            json={
                "optimizer": "adam",
                "loss": "meansquarederror"
            },
            timeout=TIMEOUT
        )
        assert response.status_code == 200


class TestMeanAbsoluteError:
    """Tests para MeanAbsoluteError."""
    
    def test_mae_basic(self, server_running, simple_model):
        """Test básico"""
        response = requests.post(
            f"{BASE_URL}/models/{simple_model}/compile",
            json={
                "optimizer": "adam",
                "loss": "mae"
            },
            timeout=TIMEOUT
        )
        assert response.status_code == 200
    
    def test_mae_alternative_name(self, server_running, simple_model):
        """Test con nombre alternativo"""
        response = requests.post(
            f"{BASE_URL}/models/{simple_model}/compile",
            json={
                "optimizer": "adam",
                "loss": "meanabsoluteerror"
            },
            timeout=TIMEOUT
        )
        assert response.status_code == 200


class TestInvalidLosses:
    """Tests para loss functions inválidos."""
    
    def test_invalid_loss(self, server_running, simple_model):
        """Test loss inválido"""
        response = requests.post(
            f"{BASE_URL}/models/{simple_model}/compile",
            json={
                "optimizer": "adam",
                "loss": "invalid_loss"
            },
            timeout=TIMEOUT
        )
        assert response.status_code == 400


if __name__ == "__main__":
    pytest.main([__file__, "-v"])











