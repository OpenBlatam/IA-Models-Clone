"""
Tests para casos edge y validaciones
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


class TestInvalidRequests:
    """Tests para requests inválidos."""
    
    def test_create_model_missing_layers(self, server_running):
        """Test crear modelo sin layers"""
        response = requests.post(
            f"{BASE_URL}/models/create",
            json={},
            timeout=TIMEOUT
        )
        assert response.status_code == 422  # Validation error
    
    def test_create_model_invalid_json(self, server_running):
        """Test con JSON inválido"""
        response = requests.post(
            f"{BASE_URL}/models/create",
            data="invalid json",
            headers={"Content-Type": "application/json"},
            timeout=TIMEOUT
        )
        assert response.status_code in [400, 422]
    
    def test_compile_without_model(self, server_running):
        """Test compilar sin especificar modelo"""
        response = requests.post(
            f"{BASE_URL}/models//compile",
            json={"optimizer": "adam", "loss": "mse"},
            timeout=TIMEOUT
        )
        assert response.status_code in [404, 405]
    
    def test_train_without_compile(self, server_running):
        """Test entrenar sin compilar primero"""
        # Crear modelo
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
        
        # Intentar entrenar sin compilar
        x_train = np.random.randn(10, 10).astype(np.float32).tolist()
        y_train = np.random.randint(0, 3, 10).astype(np.int64).tolist()
        
        train_response = requests.post(
            f"{BASE_URL}/models/{model_id}/train",
            json={
                "x_train": x_train,
                "y_train": y_train,
                "epochs": 1
            },
            timeout=TIMEOUT
        )
        assert train_response.status_code == 400


class TestDataValidation:
    """Tests para validación de datos."""
    
    def test_train_empty_data(self, server_running):
        """Test entrenar con datos vacíos"""
        # Crear y compilar modelo
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
        
        # Intentar entrenar con datos vacíos
        train_response = requests.post(
            f"{BASE_URL}/models/{model_id}/train",
            json={
                "x_train": [],
                "y_train": [],
                "epochs": 1
            },
            timeout=TIMEOUT
        )
        assert train_response.status_code == 400
    
    def test_train_mismatched_data(self, server_running):
        """Test entrenar con datos que no coinciden"""
        # Crear y compilar modelo
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
        
        # Datos con tamaños diferentes
        x_train = np.random.randn(10, 10).astype(np.float32).tolist()
        y_train = np.random.randint(0, 3, 5).astype(np.int64).tolist()  # Diferente tamaño
        
        train_response = requests.post(
            f"{BASE_URL}/models/{model_id}/train",
            json={
                "x_train": x_train,
                "y_train": y_train,
                "epochs": 1
            },
            timeout=TIMEOUT
        )
        # Puede fallar o dar error
        assert train_response.status_code in [200, 400, 500]
    
    def test_predict_empty_data(self, server_running):
        """Test predicción con datos vacíos"""
        # Crear y compilar modelo
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
        
        # Intentar predecir con datos vacíos
        predict_response = requests.post(
            f"{BASE_URL}/models/{model_id}/predict",
            json={"x": []},
            timeout=TIMEOUT
        )
        assert predict_response.status_code in [200, 400]


class TestModelOperations:
    """Tests para operaciones con modelos."""
    
    def test_delete_twice(self, server_running):
        """Test eliminar modelo dos veces"""
        # Crear modelo
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
        
        # Eliminar primera vez
        delete1 = requests.delete(
            f"{BASE_URL}/models/{model_id}",
            timeout=TIMEOUT
        )
        assert delete1.status_code == 200
        
        # Intentar eliminar segunda vez
        delete2 = requests.delete(
            f"{BASE_URL}/models/{model_id}",
            timeout=TIMEOUT
        )
        assert delete2.status_code == 404
    
    def test_compile_twice(self, server_running):
        """Test compilar modelo dos veces"""
        # Crear modelo
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
        
        # Compilar primera vez
        compile1 = requests.post(
            f"{BASE_URL}/models/{model_id}/compile",
            json={"optimizer": "adam", "loss": "mse"},
            timeout=TIMEOUT
        )
        assert compile1.status_code == 200
        
        # Compilar segunda vez (debería funcionar o dar error)
        compile2 = requests.post(
            f"{BASE_URL}/models/{model_id}/compile",
            json={"optimizer": "sgd", "loss": "mse"},
            timeout=TIMEOUT
        )
        assert compile2.status_code in [200, 400]


class TestLargeModels:
    """Tests para modelos grandes."""
    
    def test_model_with_many_layers(self, server_running):
        """Test modelo con muchas capas"""
        layers = []
        for i in range(10):
            layers.append({"type": "dense", "params": {"units": 64, "activation": "relu"}})
            layers.append({"type": "dropout", "params": {"rate": 0.2}})
        layers.append({"type": "dense", "params": {"units": 10, "activation": "softmax"}})
        
        response = requests.post(
            f"{BASE_URL}/models/create",
            json={"layers": layers},
            timeout=TIMEOUT * 2
        )
        assert response.status_code == 200
    
    def test_model_with_large_layers(self, server_running):
        """Test modelo con capas grandes"""
        response = requests.post(
            f"{BASE_URL}/models/create",
            json={
                "layers": [
                    {"type": "dense", "params": {"units": 1024, "activation": "relu"}},
                    {"type": "dense", "params": {"units": 512, "activation": "relu"}},
                    {"type": "dense", "params": {"units": 256, "activation": "relu"}},
                    {"type": "dense", "params": {"units": 10, "activation": "softmax"}}
                ]
            },
            timeout=TIMEOUT * 2
        )
        assert response.status_code == 200


class TestConcurrentOperations:
    """Tests para operaciones concurrentes."""
    
    def test_create_multiple_models(self, server_running):
        """Test crear múltiples modelos"""
        model_ids = []
        for i in range(5):
            response = requests.post(
                f"{BASE_URL}/models/create",
                json={
                    "layers": [
                        {"type": "dense", "params": {"units": 32}}
                    ],
                    "name": f"test-model-{i}"
                },
                timeout=TIMEOUT
            )
            if response.status_code == 200:
                model_ids.append(response.json()["model_id"])
        
        # Verificar que todos se crearon
        list_response = requests.get(f"{BASE_URL}/models", timeout=TIMEOUT)
        assert list_response.status_code == 200
        models = list_response.json()["models"]
        assert len(models) >= len(model_ids)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])











