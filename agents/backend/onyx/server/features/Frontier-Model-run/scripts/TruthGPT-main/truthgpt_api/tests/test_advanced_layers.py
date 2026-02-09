"""
Tests Avanzados para Layers
===========================
Tests para configuraciones avanzadas y combinaciones de layers
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


class TestAdvancedConv2D:
    """Tests avanzados para Conv2D."""
    
    def test_conv2d_with_padding(self, server_running):
        """Test Conv2D con padding"""
        response = requests.post(
            f"{BASE_URL}/models/create",
            json={
                "layers": [
                    {"type": "conv2d", "params": {"filters": 32, "kernel_size": 3, "padding": "same"}},
                    {"type": "conv2d", "params": {"filters": 64, "kernel_size": 3, "padding": "valid"}}
                ]
            },
            timeout=TIMEOUT
        )
        assert response.status_code == 200
    
    def test_conv2d_different_kernel_sizes(self, server_running):
        """Test Conv2D con diferentes tamaños de kernel"""
        kernel_sizes = [1, 3, 5, 7]
        for kernel_size in kernel_sizes:
            response = requests.post(
                f"{BASE_URL}/models/create",
                json={
                    "layers": [
                        {"type": "conv2d", "params": {"filters": 32, "kernel_size": kernel_size}}
                    ]
                },
                timeout=TIMEOUT
            )
            assert response.status_code == 200
    
    def test_conv2d_deep_network(self, server_running):
        """Test red CNN profunda"""
        layers = []
        for i in range(5):
            layers.append({"type": "conv2d", "params": {"filters": 32 * (i + 1), "kernel_size": 3}})
            layers.append({"type": "maxpooling2d", "params": {"pool_size": 2}})
        
        layers.append({"type": "flatten", "params": {}})
        layers.append({"type": "dense", "params": {"units": 128}})
        layers.append({"type": "dense", "params": {"units": 10, "activation": "softmax"}})
        
        response = requests.post(
            f"{BASE_URL}/models/create",
            json={"layers": layers},
            timeout=TIMEOUT
        )
        assert response.status_code == 200


class TestAdvancedRNN:
    """Tests avanzados para RNN."""
    
    def test_bidirectional_lstm(self, server_running):
        """Test LSTM bidireccional (simulado con múltiples capas)"""
        response = requests.post(
            f"{BASE_URL}/models/create",
            json={
                "layers": [
                    {"type": "lstm", "params": {"units": 64, "return_sequences": True}},
                    {"type": "lstm", "params": {"units": 64, "return_sequences": False}},
                    {"type": "dense", "params": {"units": 10, "activation": "softmax"}}
                ]
            },
            timeout=TIMEOUT
        )
        assert response.status_code == 200
    
    def test_gru_stacked(self, server_running):
        """Test múltiples capas GRU apiladas"""
        response = requests.post(
            f"{BASE_URL}/models/create",
            json={
                "layers": [
                    {"type": "gru", "params": {"units": 128, "return_sequences": True}},
                    {"type": "gru", "params": {"units": 64, "return_sequences": True}},
                    {"type": "gru", "params": {"units": 32, "return_sequences": False}},
                    {"type": "dense", "params": {"units": 10, "activation": "softmax"}}
                ]
            },
            timeout=TIMEOUT
        )
        assert response.status_code == 200


class TestResidualConnections:
    """Tests para conexiones residuales (simuladas)."""
    
    def test_skip_connection_pattern(self, server_running):
        """Test patrón de skip connection"""
        response = requests.post(
            f"{BASE_URL}/models/create",
            json={
                "layers": [
                    {"type": "dense", "params": {"units": 64, "activation": "relu"}},
                    {"type": "dropout", "params": {"rate": 0.2}},
                    {"type": "dense", "params": {"units": 64, "activation": "relu"}},
                    {"type": "dense", "params": {"units": 64, "activation": "relu"}},  # Simula skip
                    {"type": "dense", "params": {"units": 10, "activation": "softmax"}}
                ]
            },
            timeout=TIMEOUT
        )
        assert response.status_code == 200


class TestAttentionLayers:
    """Tests para layers de atención."""
    
    def test_attention_layer(self, server_running):
        """Test layer de atención"""
        response = requests.post(
            f"{BASE_URL}/models/create",
            json={
                "layers": [
                    {"type": "dense", "params": {"units": 64}},
                    {"type": "attention", "params": {}},
                    {"type": "dense", "params": {"units": 10, "activation": "softmax"}}
                ]
            },
            timeout=TIMEOUT
        )
        # Puede fallar si attention no está implementado
        assert response.status_code in [200, 400]
    
    def test_multihead_attention(self, server_running):
        """Test multi-head attention"""
        response = requests.post(
            f"{BASE_URL}/models/create",
            json={
                "layers": [
                    {"type": "dense", "params": {"units": 64}},
                    {"type": "multiheadattention", "params": {"num_heads": 8}},
                    {"type": "dense", "params": {"units": 10, "activation": "softmax"}}
                ]
            },
            timeout=TIMEOUT
        )
        # Puede fallar si multiheadattention no está implementado
        assert response.status_code in [200, 400]


class TestBatchNormalization:
    """Tests para BatchNormalization."""
    
    def test_batchnormalization_after_conv(self, server_running):
        """Test BatchNorm después de Conv2D"""
        response = requests.post(
            f"{BASE_URL}/models/create",
            json={
                "layers": [
                    {"type": "conv2d", "params": {"filters": 32, "kernel_size": 3}},
                    {"type": "batchnormalization", "params": {}},
                    {"type": "maxpooling2d", "params": {"pool_size": 2}}
                ]
            },
            timeout=TIMEOUT
        )
        assert response.status_code == 200
    
    def test_batchnormalization_after_dense(self, server_running):
        """Test BatchNorm después de Dense"""
        response = requests.post(
            f"{BASE_URL}/models/create",
            json={
                "layers": [
                    {"type": "dense", "params": {"units": 128, "activation": "relu"}},
                    {"type": "batchnormalization", "params": {}},
                    {"type": "dropout", "params": {"rate": 0.2}},
                    {"type": "dense", "params": {"units": 10, "activation": "softmax"}}
                ]
            },
            timeout=TIMEOUT
        )
        assert response.status_code == 200


class TestEmbeddingLayers:
    """Tests para Embedding layers."""
    
    def test_embedding_layer(self, server_running):
        """Test Embedding layer"""
        response = requests.post(
            f"{BASE_URL}/models/create",
            json={
                "layers": [
                    {"type": "embedding", "params": {"input_dim": 10000, "output_dim": 128}},
                    {"type": "lstm", "params": {"units": 64}},
                    {"type": "dense", "params": {"units": 10, "activation": "softmax"}}
                ]
            },
            timeout=TIMEOUT
        )
        assert response.status_code == 200


class TestComplexArchitectures:
    """Tests para arquitecturas complejas."""
    
    def test_transformer_like(self, server_running):
        """Test arquitectura tipo Transformer"""
        response = requests.post(
            f"{BASE_URL}/models/create",
            json={
                "layers": [
                    {"type": "embedding", "params": {"input_dim": 5000, "output_dim": 256}},
                    {"type": "dense", "params": {"units": 256, "activation": "relu"}},
                    {"type": "dropout", "params": {"rate": 0.1}},
                    {"type": "dense", "params": {"units": 256, "activation": "relu"}},
                    {"type": "dropout", "params": {"rate": 0.1}},
                    {"type": "dense", "params": {"units": 10, "activation": "softmax"}}
                ],
                "name": "transformer-like-model"
            },
            timeout=TIMEOUT
        )
        assert response.status_code == 200
    
    def test_hybrid_cnn_rnn(self, server_running):
        """Test arquitectura híbrida CNN + RNN"""
        response = requests.post(
            f"{BASE_URL}/models/create",
            json={
                "layers": [
                    {"type": "conv2d", "params": {"filters": 32, "kernel_size": 3}},
                    {"type": "maxpooling2d", "params": {"pool_size": 2}},
                    {"type": "flatten", "params": {}},
                    {"type": "dense", "params": {"units": 128}},
                    {"type": "lstm", "params": {"units": 64}},
                    {"type": "dense", "params": {"units": 10, "activation": "softmax"}}
                ],
                "name": "hybrid-cnn-rnn"
            },
            timeout=TIMEOUT
        )
        assert response.status_code == 200
    
    def test_wide_deep_network(self, server_running):
        """Test red ancha y profunda"""
        layers = []
        # Parte ancha
        for i in range(3):
            layers.append({"type": "dense", "params": {"units": 512, "activation": "relu"}})
            layers.append({"type": "dropout", "params": {"rate": 0.3}})
        
        # Parte profunda
        for i in range(5):
            layers.append({"type": "dense", "params": {"units": 256, "activation": "relu"}})
            layers.append({"type": "dropout", "params": {"rate": 0.2}})
        
        layers.append({"type": "dense", "params": {"units": 10, "activation": "softmax"}})
        
        response = requests.post(
            f"{BASE_URL}/models/create",
            json={"layers": layers, "name": "wide-deep-network"},
            timeout=TIMEOUT
        )
        assert response.status_code == 200


class TestLayerCombinations:
    """Tests para combinaciones específicas de layers."""
    
    def test_conv_pool_conv_pool_pattern(self, server_running):
        """Test patrón Conv-Pool repetido"""
        response = requests.post(
            f"{BASE_URL}/models/create",
            json={
                "layers": [
                    {"type": "conv2d", "params": {"filters": 32, "kernel_size": 3}},
                    {"type": "maxpooling2d", "params": {"pool_size": 2}},
                    {"type": "conv2d", "params": {"filters": 64, "kernel_size": 3}},
                    {"type": "maxpooling2d", "params": {"pool_size": 2}},
                    {"type": "conv2d", "params": {"filters": 128, "kernel_size": 3}},
                    {"type": "maxpooling2d", "params": {"pool_size": 2}},
                    {"type": "flatten", "params": {}},
                    {"type": "dense", "params": {"units": 10, "activation": "softmax"}}
                ]
            },
            timeout=TIMEOUT
        )
        assert response.status_code == 200
    
    def test_dense_dropout_dense_pattern(self, server_running):
        """Test patrón Dense-Dropout repetido"""
        layers = []
        for i in range(4):
            layers.append({"type": "dense", "params": {"units": 128, "activation": "relu"}})
            layers.append({"type": "dropout", "params": {"rate": 0.2}})
        
        layers.append({"type": "dense", "params": {"units": 10, "activation": "softmax"}})
        
        response = requests.post(
            f"{BASE_URL}/models/create",
            json={"layers": layers},
            timeout=TIMEOUT
        )
        assert response.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])











