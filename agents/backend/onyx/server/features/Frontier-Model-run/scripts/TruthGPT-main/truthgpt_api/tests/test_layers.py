"""
Tests para diferentes tipos de layers
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


class TestDenseLayers:
    """Tests para layers Dense."""
    
    def test_dense_basic(self, server_running):
        """Test layer Dense básico"""
        response = requests.post(
            f"{BASE_URL}/models/create",
            json={
                "layers": [
                    {"type": "dense", "params": {"units": 64}}
                ]
            },
            timeout=TIMEOUT
        )
        assert response.status_code == 200
    
    def test_dense_with_activation(self, server_running):
        """Test Dense con activación"""
        response = requests.post(
            f"{BASE_URL}/models/create",
            json={
                "layers": [
                    {"type": "dense", "params": {"units": 128, "activation": "relu"}},
                    {"type": "dense", "params": {"units": 64, "activation": "sigmoid"}},
                    {"type": "dense", "params": {"units": 10, "activation": "softmax"}}
                ]
            },
            timeout=TIMEOUT
        )
        assert response.status_code == 200
    
    def test_dense_without_bias(self, server_running):
        """Test Dense sin bias"""
        response = requests.post(
            f"{BASE_URL}/models/create",
            json={
                "layers": [
                    {"type": "dense", "params": {"units": 64, "use_bias": False}}
                ]
            },
            timeout=TIMEOUT
        )
        assert response.status_code == 200


class TestConv2DLayers:
    """Tests para layers Conv2D."""
    
    def test_conv2d_basic(self, server_running):
        """Test Conv2D básico"""
        response = requests.post(
            f"{BASE_URL}/models/create",
            json={
                "layers": [
                    {"type": "conv2d", "params": {"filters": 32, "kernel_size": 3}}
                ]
            },
            timeout=TIMEOUT
        )
        assert response.status_code == 200
    
    def test_conv2d_with_activation(self, server_running):
        """Test Conv2D con activación"""
        response = requests.post(
            f"{BASE_URL}/models/create",
            json={
                "layers": [
                    {"type": "conv2d", "params": {"filters": 32, "kernel_size": 3, "activation": "relu"}},
                    {"type": "maxpooling2d", "params": {"pool_size": 2}},
                    {"type": "conv2d", "params": {"filters": 64, "kernel_size": 3, "activation": "relu"}},
                    {"type": "flatten", "params": {}},
                    {"type": "dense", "params": {"units": 10, "activation": "softmax"}}
                ]
            },
            timeout=TIMEOUT
        )
        assert response.status_code == 200
    
    def test_conv2d_with_strides(self, server_running):
        """Test Conv2D con strides personalizados"""
        response = requests.post(
            f"{BASE_URL}/models/create",
            json={
                "layers": [
                    {"type": "conv2d", "params": {"filters": 32, "kernel_size": 3, "strides": [2, 2]}}
                ]
            },
            timeout=TIMEOUT
        )
        assert response.status_code == 200


class TestLSTMLayers:
    """Tests para layers LSTM."""
    
    def test_lstm_basic(self, server_running):
        """Test LSTM básico"""
        response = requests.post(
            f"{BASE_URL}/models/create",
            json={
                "layers": [
                    {"type": "lstm", "params": {"units": 64}}
                ]
            },
            timeout=TIMEOUT
        )
        assert response.status_code == 200
    
    def test_lstm_return_sequences(self, server_running):
        """Test LSTM con return_sequences"""
        response = requests.post(
            f"{BASE_URL}/models/create",
            json={
                "layers": [
                    {"type": "lstm", "params": {"units": 64, "return_sequences": True}},
                    {"type": "lstm", "params": {"units": 32, "return_sequences": False}},
                    {"type": "dense", "params": {"units": 10, "activation": "softmax"}}
                ]
            },
            timeout=TIMEOUT
        )
        assert response.status_code == 200


class TestGRULayers:
    """Tests para layers GRU."""
    
    def test_gru_basic(self, server_running):
        """Test GRU básico"""
        response = requests.post(
            f"{BASE_URL}/models/create",
            json={
                "layers": [
                    {"type": "gru", "params": {"units": 64}}
                ]
            },
            timeout=TIMEOUT
        )
        assert response.status_code == 200


class TestDropoutLayers:
    """Tests para layers Dropout."""
    
    def test_dropout_basic(self, server_running):
        """Test Dropout básico"""
        response = requests.post(
            f"{BASE_URL}/models/create",
            json={
                "layers": [
                    {"type": "dense", "params": {"units": 128, "activation": "relu"}},
                    {"type": "dropout", "params": {"rate": 0.5}},
                    {"type": "dense", "params": {"units": 10, "activation": "softmax"}}
                ]
            },
            timeout=TIMEOUT
        )
        assert response.status_code == 200
    
    def test_dropout_different_rates(self, server_running):
        """Test Dropout con diferentes rates"""
        response = requests.post(
            f"{BASE_URL}/models/create",
            json={
                "layers": [
                    {"type": "dense", "params": {"units": 128, "activation": "relu"}},
                    {"type": "dropout", "params": {"rate": 0.2}},
                    {"type": "dense", "params": {"units": 64, "activation": "relu"}},
                    {"type": "dropout", "params": {"rate": 0.3}},
                    {"type": "dense", "params": {"units": 10, "activation": "softmax"}}
                ]
            },
            timeout=TIMEOUT
        )
        assert response.status_code == 200


class TestPoolingLayers:
    """Tests para layers de pooling."""
    
    def test_maxpooling2d(self, server_running):
        """Test MaxPooling2D"""
        response = requests.post(
            f"{BASE_URL}/models/create",
            json={
                "layers": [
                    {"type": "conv2d", "params": {"filters": 32, "kernel_size": 3}},
                    {"type": "maxpooling2d", "params": {"pool_size": 2}}
                ]
            },
            timeout=TIMEOUT
        )
        assert response.status_code == 200
    
    def test_averagepooling2d(self, server_running):
        """Test AveragePooling2D"""
        response = requests.post(
            f"{BASE_URL}/models/create",
            json={
                "layers": [
                    {"type": "conv2d", "params": {"filters": 32, "kernel_size": 3}},
                    {"type": "averagepooling2d", "params": {"pool_size": 2}}
                ]
            },
            timeout=TIMEOUT
        )
        assert response.status_code == 200


class TestUtilityLayers:
    """Tests para layers de utilidad."""
    
    def test_flatten(self, server_running):
        """Test Flatten"""
        response = requests.post(
            f"{BASE_URL}/models/create",
            json={
                "layers": [
                    {"type": "conv2d", "params": {"filters": 32, "kernel_size": 3}},
                    {"type": "flatten", "params": {}},
                    {"type": "dense", "params": {"units": 10}}
                ]
            },
            timeout=TIMEOUT
        )
        assert response.status_code == 200
    
    def test_reshape(self, server_running):
        """Test Reshape"""
        response = requests.post(
            f"{BASE_URL}/models/create",
            json={
                "layers": [
                    {"type": "dense", "params": {"units": 64}},
                    {"type": "reshape", "params": {"target_shape": [8, 8]}}
                ]
            },
            timeout=TIMEOUT
        )
        assert response.status_code == 200


class TestComplexArchitectures:
    """Tests para arquitecturas complejas."""
    
    def test_cnn_architecture(self, server_running):
        """Test arquitectura CNN completa"""
        response = requests.post(
            f"{BASE_URL}/models/create",
            json={
                "layers": [
                    {"type": "conv2d", "params": {"filters": 32, "kernel_size": 3, "activation": "relu"}},
                    {"type": "maxpooling2d", "params": {"pool_size": 2}},
                    {"type": "conv2d", "params": {"filters": 64, "kernel_size": 3, "activation": "relu"}},
                    {"type": "maxpooling2d", "params": {"pool_size": 2}},
                    {"type": "flatten", "params": {}},
                    {"type": "dense", "params": {"units": 128, "activation": "relu"}},
                    {"type": "dropout", "params": {"rate": 0.5}},
                    {"type": "dense", "params": {"units": 10, "activation": "softmax"}}
                ],
                "name": "cnn-model"
            },
            timeout=TIMEOUT
        )
        assert response.status_code == 200
    
    def test_rnn_architecture(self, server_running):
        """Test arquitectura RNN completa"""
        response = requests.post(
            f"{BASE_URL}/models/create",
            json={
                "layers": [
                    {"type": "lstm", "params": {"units": 128, "return_sequences": True}},
                    {"type": "dropout", "params": {"rate": 0.2}},
                    {"type": "lstm", "params": {"units": 64, "return_sequences": False}},
                    {"type": "dense", "params": {"units": 32, "activation": "relu"}},
                    {"type": "dense", "params": {"units": 10, "activation": "softmax"}}
                ],
                "name": "rnn-model"
            },
            timeout=TIMEOUT
        )
        assert response.status_code == 200
    
    def test_hybrid_architecture(self, server_running):
        """Test arquitectura híbrida"""
        response = requests.post(
            f"{BASE_URL}/models/create",
            json={
                "layers": [
                    {"type": "conv2d", "params": {"filters": 32, "kernel_size": 3}},
                    {"type": "flatten", "params": {}},
                    {"type": "dense", "params": {"units": 64, "activation": "relu"}},
                    {"type": "lstm", "params": {"units": 32}},
                    {"type": "dense", "params": {"units": 10, "activation": "softmax"}}
                ],
                "name": "hybrid-model"
            },
            timeout=TIMEOUT
        )
        assert response.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])











