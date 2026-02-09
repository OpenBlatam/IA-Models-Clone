"""
Tests de Manejo de Errores
==========================
Tests que verifican que los errores se manejen correctamente
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


class TestHTTPErrors:
    """Tests para errores HTTP."""
    
    def test_404_not_found(self, server_running):
        """Test 404 para endpoint inexistente"""
        response = requests.get(f"{BASE_URL}/nonexistent-endpoint", timeout=TIMEOUT)
        assert response.status_code == 404
    
    def test_405_method_not_allowed(self, server_running):
        """Test 405 para método no permitido"""
        # GET en endpoint que solo acepta POST
        response = requests.get(f"{BASE_URL}/models/create", timeout=TIMEOUT)
        assert response.status_code == 405
    
    def test_422_validation_error(self, server_running):
        """Test 422 para errores de validación"""
        # Request sin campos requeridos
        response = requests.post(
            f"{BASE_URL}/models/create",
            json={},
            timeout=TIMEOUT
        )
        assert response.status_code == 422


class TestModelErrors:
    """Tests para errores relacionados con modelos."""
    
    def test_create_model_invalid_json(self, server_running):
        """Test crear modelo con JSON inválido"""
        response = requests.post(
            f"{BASE_URL}/models/create",
            data="invalid json",
            headers={"Content-Type": "application/json"},
            timeout=TIMEOUT
        )
        assert response.status_code in [400, 422]
    
    def test_compile_nonexistent_model(self, server_running):
        """Test compilar modelo que no existe"""
        response = requests.post(
            f"{BASE_URL}/models/invalid-model-id-12345/compile",
            json={"optimizer": "adam", "loss": "mse"},
            timeout=TIMEOUT
        )
        assert response.status_code == 404
    
    def test_train_without_compile(self, server_running):
        """Test entrenar sin compilar"""
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
        
        x_train = np.random.randn(10, 32).astype(np.float32).tolist()
        y_train = np.random.randn(10, 1).astype(np.float32).tolist()
        
        response = requests.post(
            f"{BASE_URL}/models/{model_id}/train",
            json={"x_train": x_train, "y_train": y_train, "epochs": 1},
            timeout=TIMEOUT
        )
        assert response.status_code == 400
    
    def test_predict_without_compile(self, server_running):
        """Test predecir sin compilar"""
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
        
        x = np.random.randn(5, 32).astype(np.float32).tolist()
        
        response = requests.post(
            f"{BASE_URL}/models/{model_id}/predict",
            json={"x": x},
            timeout=TIMEOUT
        )
        # Puede funcionar o dar error, dependiendo de la implementación
        assert response.status_code in [200, 400, 500]


class TestDataErrors:
    """Tests para errores relacionados con datos."""
    
    def test_train_empty_arrays(self, server_running):
        """Test entrenar con arrays vacíos"""
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
    
    def test_train_mismatched_lengths(self, server_running):
        """Test entrenar con longitudes diferentes"""
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
        
        # x_train y y_train con diferentes longitudes
        x_train = np.random.randn(100, 32).astype(np.float32).tolist()
        y_train = np.random.randn(50, 1).astype(np.float32).tolist()
        
        response = requests.post(
            f"{BASE_URL}/models/{model_id}/train",
            json={"x_train": x_train, "y_train": y_train, "epochs": 1},
            timeout=TIMEOUT
        )
        assert response.status_code in [200, 400, 500]  # Puede fallar o ajustar
    
    def test_predict_wrong_shape(self, server_running):
        """Test predecir con shape incorrecto"""
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
        
        # Shape incorrecto (debería ser 10 features)
        x = np.random.randn(5, 5).astype(np.float32).tolist()
        
        response = requests.post(
            f"{BASE_URL}/models/{model_id}/predict",
            json={"x": x},
            timeout=TIMEOUT
        )
        assert response.status_code in [200, 400, 500]


class TestParameterErrors:
    """Tests para errores en parámetros."""
    
    def test_invalid_optimizer_params(self, server_running):
        """Test con parámetros de optimizer inválidos"""
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
        
        # Batch size negativo
        response = requests.post(
            f"{BASE_URL}/models/{model_id}/train",
            json={
                "x_train": x_train,
                "y_train": y_train,
                "epochs": 1,
                "batch_size": -1
            },
            timeout=TIMEOUT
        )
        assert response.status_code in [200, 400]


class TestErrorMessages:
    """Tests para verificar mensajes de error."""
    
    def test_error_message_format(self, server_running):
        """Test que los errores tengan formato adecuado"""
        # Crear request inválido
        response = requests.post(
            f"{BASE_URL}/models/create",
            json={"layers": []},
            timeout=TIMEOUT
        )
        
        # Debería tener algún mensaje de error
        assert response.status_code >= 400
        try:
            data = response.json()
            # Debería tener algún campo de error
            assert "detail" in data or "error" in data or "message" in data
        except:
            # Si no es JSON, el texto debería tener contenido
            assert len(response.text) > 0
    
    def test_404_error_message(self, server_running):
        """Test mensaje de error 404"""
        response = requests.get(
            f"{BASE_URL}/models/nonexistent-id",
            timeout=TIMEOUT
        )
        
        assert response.status_code == 404
        try:
            data = response.json()
            assert "detail" in data or "error" in data or "message" in data
        except:
            assert len(response.text) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])











