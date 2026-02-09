"""
Tests para Diferentes Tipos de Datos
====================================
Tests que verifican el manejo de diferentes tipos y formatos de datos
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


class TestNumericDataTypes:
    """Tests para diferentes tipos numéricos."""
    
    def test_float32_data(self, server_running):
        """Test con datos float32"""
        create_response = requests.post(
            f"{BASE_URL}/models/create",
            json={
                "layers": [
                    {"type": "dense", "params": {"units": 32}},
                    {"type": "dense", "params": {"units": 10}}
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
        
        # Datos float32
        x_train = np.random.randn(50, 32).astype(np.float32).tolist()
        y_train = np.random.randn(50, 10).astype(np.float32).tolist()
        
        response = requests.post(
            f"{BASE_URL}/models/{model_id}/train",
            json={"x_train": x_train, "y_train": y_train, "epochs": 1, "verbose": 0},
            timeout=TIMEOUT * 2
        )
        assert response.status_code == 200
    
    def test_float64_data(self, server_running):
        """Test con datos float64"""
        create_response = requests.post(
            f"{BASE_URL}/models/create",
            json={
                "layers": [
                    {"type": "dense", "params": {"units": 32}},
                    {"type": "dense", "params": {"units": 10}}
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
        
        # Datos float64 (se convertirán a float32)
        x_train = np.random.randn(50, 32).astype(np.float64).tolist()
        y_train = np.random.randn(50, 10).astype(np.float64).tolist()
        
        response = requests.post(
            f"{BASE_URL}/models/{model_id}/train",
            json={"x_train": x_train, "y_train": y_train, "epochs": 1, "verbose": 0},
            timeout=TIMEOUT * 2
        )
        assert response.status_code == 200
    
    def test_int32_labels(self, server_running):
        """Test con labels int32"""
        create_response = requests.post(
            f"{BASE_URL}/models/create",
            json={
                "layers": [
                    {"type": "dense", "params": {"units": 32}},
                    {"type": "dense", "params": {"units": 10, "activation": "softmax"}}
                ]
            },
            timeout=TIMEOUT
        )
        model_id = create_response.json()["model_id"]
        
        requests.post(
            f"{BASE_URL}/models/{model_id}/compile",
            json={"optimizer": "adam", "loss": "sparsecategoricalcrossentropy"},
            timeout=TIMEOUT
        )
        
        x_train = np.random.randn(50, 32).astype(np.float32).tolist()
        y_train = np.random.randint(0, 10, 50).astype(np.int32).tolist()
        
        response = requests.post(
            f"{BASE_URL}/models/{model_id}/train",
            json={"x_train": x_train, "y_train": y_train, "epochs": 1, "verbose": 0},
            timeout=TIMEOUT * 2
        )
        assert response.status_code == 200
    
    def test_int64_labels(self, server_running):
        """Test con labels int64"""
        create_response = requests.post(
            f"{BASE_URL}/models/create",
            json={
                "layers": [
                    {"type": "dense", "params": {"units": 32}},
                    {"type": "dense", "params": {"units": 10, "activation": "softmax"}}
                ]
            },
            timeout=TIMEOUT
        )
        model_id = create_response.json()["model_id"]
        
        requests.post(
            f"{BASE_URL}/models/{model_id}/compile",
            json={"optimizer": "adam", "loss": "sparsecategoricalcrossentropy"},
            timeout=TIMEOUT
        )
        
        x_train = np.random.randn(50, 32).astype(np.float32).tolist()
        y_train = np.random.randint(0, 10, 50).astype(np.int64).tolist()
        
        response = requests.post(
            f"{BASE_URL}/models/{model_id}/train",
            json={"x_train": x_train, "y_train": y_train, "epochs": 1, "verbose": 0},
            timeout=TIMEOUT * 2
        )
        assert response.status_code == 200


class TestDataShapes:
    """Tests para diferentes formas de datos."""
    
    def test_1d_input(self, server_running):
        """Test con input 1D"""
        create_response = requests.post(
            f"{BASE_URL}/models/create",
            json={
                "layers": [
                    {"type": "dense", "params": {"units": 10}},
                    {"type": "dense", "params": {"units": 1}}
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
        
        # Input 1D (será reshapeado a 2D)
        x_train = np.random.randn(50, 10).astype(np.float32).tolist()
        y_train = np.random.randn(50).astype(np.float32).tolist()
        
        response = requests.post(
            f"{BASE_URL}/models/{model_id}/train",
            json={"x_train": x_train, "y_train": y_train, "epochs": 1, "verbose": 0},
            timeout=TIMEOUT * 2
        )
        assert response.status_code in [200, 400]  # Puede requerir reshape
    
    def test_2d_input(self, server_running):
        """Test con input 2D estándar"""
        create_response = requests.post(
            f"{BASE_URL}/models/create",
            json={
                "layers": [
                    {"type": "dense", "params": {"units": 32}},
                    {"type": "dense", "params": {"units": 10}}
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
        
        # Input 2D estándar
        x_train = np.random.randn(100, 32).astype(np.float32).tolist()
        y_train = np.random.randn(100, 10).astype(np.float32).tolist()
        
        response = requests.post(
            f"{BASE_URL}/models/{model_id}/train",
            json={"x_train": x_train, "y_train": y_train, "epochs": 1, "verbose": 0},
            timeout=TIMEOUT * 2
        )
        assert response.status_code == 200


class TestDataRanges:
    """Tests para diferentes rangos de datos."""
    
    def test_normalized_data(self, server_running):
        """Test con datos normalizados (0-1)"""
        create_response = requests.post(
            f"{BASE_URL}/models/create",
            json={
                "layers": [
                    {"type": "dense", "params": {"units": 32}},
                    {"type": "dense", "params": {"units": 10}}
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
        
        # Datos normalizados
        x_train = np.random.rand(50, 32).astype(np.float32).tolist()  # 0-1
        y_train = np.random.rand(50, 10).astype(np.float32).tolist()
        
        response = requests.post(
            f"{BASE_URL}/models/{model_id}/train",
            json={"x_train": x_train, "y_train": y_train, "epochs": 1, "verbose": 0},
            timeout=TIMEOUT * 2
        )
        assert response.status_code == 200
    
    def test_large_values(self, server_running):
        """Test con valores grandes"""
        create_response = requests.post(
            f"{BASE_URL}/models/create",
            json={
                "layers": [
                    {"type": "dense", "params": {"units": 32}},
                    {"type": "dense", "params": {"units": 10}}
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
        
        # Valores grandes
        x_train = (np.random.randn(50, 32) * 100).astype(np.float32).tolist()
        y_train = (np.random.randn(50, 10) * 100).astype(np.float32).tolist()
        
        response = requests.post(
            f"{BASE_URL}/models/{model_id}/train",
            json={"x_train": x_train, "y_train": y_train, "epochs": 1, "verbose": 0},
            timeout=TIMEOUT * 2
        )
        assert response.status_code in [200, 400]  # Puede funcionar o dar warning
    
    def test_small_values(self, server_running):
        """Test con valores pequeños"""
        create_response = requests.post(
            f"{BASE_URL}/models/create",
            json={
                "layers": [
                    {"type": "dense", "params": {"units": 32}},
                    {"type": "dense", "params": {"units": 10}}
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
        
        # Valores pequeños
        x_train = (np.random.randn(50, 32) * 0.001).astype(np.float32).tolist()
        y_train = (np.random.randn(50, 10) * 0.001).astype(np.float32).tolist()
        
        response = requests.post(
            f"{BASE_URL}/models/{model_id}/train",
            json={"x_train": x_train, "y_train": y_train, "epochs": 1, "verbose": 0},
            timeout=TIMEOUT * 2
        )
        assert response.status_code == 200


class TestCategoricalData:
    """Tests para datos categóricos."""
    
    def test_binary_classification(self, server_running):
        """Test clasificación binaria"""
        create_response = requests.post(
            f"{BASE_URL}/models/create",
            json={
                "layers": [
                    {"type": "dense", "params": {"units": 32}},
                    {"type": "dense", "params": {"units": 1, "activation": "sigmoid"}}
                ]
            },
            timeout=TIMEOUT
        )
        model_id = create_response.json()["model_id"]
        
        requests.post(
            f"{BASE_URL}/models/{model_id}/compile",
            json={"optimizer": "adam", "loss": "binarycrossentropy"},
            timeout=TIMEOUT
        )
        
        x_train = np.random.randn(50, 32).astype(np.float32).tolist()
        y_train = np.random.randint(0, 2, 50).astype(np.int64).tolist()
        
        response = requests.post(
            f"{BASE_URL}/models/{model_id}/train",
            json={"x_train": x_train, "y_train": y_train, "epochs": 1, "verbose": 0},
            timeout=TIMEOUT * 2
        )
        assert response.status_code == 200
    
    def test_multi_class_classification(self, server_running):
        """Test clasificación multiclase"""
        num_classes = [3, 5, 10, 20]
        
        for n_classes in num_classes:
            create_response = requests.post(
                f"{BASE_URL}/models/create",
                json={
                    "layers": [
                        {"type": "dense", "params": {"units": 32}},
                        {"type": "dense", "params": {"units": n_classes, "activation": "softmax"}}
                    ]
                },
                timeout=TIMEOUT
            )
            model_id = create_response.json()["model_id"]
            
            requests.post(
                f"{BASE_URL}/models/{model_id}/compile",
                json={"optimizer": "adam", "loss": "sparsecategoricalcrossentropy"},
                timeout=TIMEOUT
            )
            
            x_train = np.random.randn(50, 32).astype(np.float32).tolist()
            y_train = np.random.randint(0, n_classes, 50).astype(np.int64).tolist()
            
            response = requests.post(
                f"{BASE_URL}/models/{model_id}/train",
                json={"x_train": x_train, "y_train": y_train, "epochs": 1, "verbose": 0},
                timeout=TIMEOUT * 2
            )
            assert response.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])











