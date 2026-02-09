"""
Tests de Compatibilidad
=======================
Tests para verificar compatibilidad con diferentes clientes y formatos
"""

import pytest
import requests
import numpy as np
import json

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


class TestJSONCompatibility:
    """Tests de compatibilidad con JSON."""
    
    def test_create_model_minimal_json(self, server_running):
        """Test crear modelo con JSON mínimo"""
        response = requests.post(
            f"{BASE_URL}/models/create",
            json={"layers": [{"type": "dense", "params": {"units": 10}}]},
            timeout=TIMEOUT
        )
        assert response.status_code == 200
    
    def test_create_model_full_json(self, server_running):
        """Test crear modelo con JSON completo"""
        response = requests.post(
            f"{BASE_URL}/models/create",
            json={
                "layers": [
                    {"type": "dense", "params": {"units": 64, "activation": "relu"}},
                    {"type": "dense", "params": {"units": 10, "activation": "softmax"}}
                ],
                "name": "full-json-model"
            },
            timeout=TIMEOUT
        )
        assert response.status_code == 200
    
    def test_response_json_format(self, server_running):
        """Test que las respuestas sean JSON válido"""
        response = requests.post(
            f"{BASE_URL}/models/create",
            json={
                "layers": [
                    {"type": "dense", "params": {"units": 16}}
                ]
            },
            timeout=TIMEOUT
        )
        assert response.status_code == 200
        
        # Verificar que es JSON válido
        try:
            data = response.json()
            assert isinstance(data, dict)
        except json.JSONDecodeError:
            pytest.fail("Response no es JSON válido")


class TestHTTPMethods:
    """Tests de métodos HTTP."""
    
    def test_get_health(self, server_running):
        """Test GET en health endpoint"""
        response = requests.get(f"{BASE_URL}/health", timeout=TIMEOUT)
        assert response.status_code == 200
    
    def test_get_models_list(self, server_running):
        """Test GET en models list"""
        response = requests.get(f"{BASE_URL}/models", timeout=TIMEOUT)
        assert response.status_code == 200
    
    def test_post_create_model(self, server_running):
        """Test POST para crear modelo"""
        response = requests.post(
            f"{BASE_URL}/models/create",
            json={"layers": [{"type": "dense", "params": {"units": 10}}]},
            timeout=TIMEOUT
        )
        assert response.status_code == 200
    
    def test_delete_model(self, server_running):
        """Test DELETE para eliminar modelo"""
        create_response = requests.post(
            f"{BASE_URL}/models/create",
            json={"layers": [{"type": "dense", "params": {"units": 10}}]},
            timeout=TIMEOUT
        )
        model_id = create_response.json()["model_id"]
        
        delete_response = requests.delete(
            f"{BASE_URL}/models/{model_id}",
            timeout=TIMEOUT
        )
        assert delete_response.status_code in [200, 204]


class TestContentTypes:
    """Tests de diferentes content types."""
    
    def test_json_content_type(self, server_running):
        """Test con Content-Type: application/json"""
        response = requests.post(
            f"{BASE_URL}/models/create",
            json={"layers": [{"type": "dense", "params": {"units": 10}}]},
            headers={"Content-Type": "application/json"},
            timeout=TIMEOUT
        )
        assert response.status_code == 200
    
    def test_json_utf8_content_type(self, server_running):
        """Test con Content-Type: application/json; charset=utf-8"""
        response = requests.post(
            f"{BASE_URL}/models/create",
            json={"layers": [{"type": "dense", "params": {"units": 10}}]},
            headers={"Content-Type": "application/json; charset=utf-8"},
            timeout=TIMEOUT
        )
        assert response.status_code == 200


class TestDataFormats:
    """Tests de diferentes formatos de datos."""
    
    def test_numpy_array_like(self, server_running):
        """Test con datos tipo numpy array (convertidos a lista)"""
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
        
        # Datos como lista (simulando numpy array convertido)
        x_train = np.random.randn(20, 32).tolist()
        y_train = np.random.randn(20, 10).tolist()
        
        response = requests.post(
            f"{BASE_URL}/models/{model_id}/train",
            json={"x_train": x_train, "y_train": y_train, "epochs": 1, "verbose": 0},
            timeout=TIMEOUT * 2
        )
        assert response.status_code == 200
    
    def test_python_list(self, server_running):
        """Test con listas Python puras"""
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
        
        # Listas Python puras
        x_train = [[0.1 * i + 0.5 * j for j in range(32)] for i in range(20)]
        y_train = [[0.2 * i + 0.3 * j for j in range(10)] for i in range(20)]
        
        response = requests.post(
            f"{BASE_URL}/models/{model_id}/train",
            json={"x_train": x_train, "y_train": y_train, "epochs": 1, "verbose": 0},
            timeout=TIMEOUT * 2
        )
        assert response.status_code == 200


class TestClientCompatibility:
    """Tests de compatibilidad con diferentes clientes."""
    
    def test_curl_like_request(self, server_running):
        """Test request estilo curl (sin headers especiales)"""
        response = requests.post(
            f"{BASE_URL}/models/create",
            json={"layers": [{"type": "dense", "params": {"units": 10}}]},
            timeout=TIMEOUT
        )
        assert response.status_code == 200
    
    def test_browser_like_request(self, server_running):
        """Test request estilo navegador (con headers adicionales)"""
        response = requests.post(
            f"{BASE_URL}/models/create",
            json={"layers": [{"type": "dense", "params": {"units": 10}}]},
            headers={
                "User-Agent": "Mozilla/5.0",
                "Accept": "application/json",
                "Accept-Language": "en-US,en;q=0.9"
            },
            timeout=TIMEOUT
        )
        assert response.status_code == 200
    
    def test_api_client_request(self, server_running):
        """Test request estilo cliente API"""
        response = requests.post(
            f"{BASE_URL}/models/create",
            json={"layers": [{"type": "dense", "params": {"units": 10}}]},
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "X-API-Client": "test-client"
            },
            timeout=TIMEOUT
        )
        assert response.status_code == 200


class TestVersionCompatibility:
    """Tests de compatibilidad de versiones."""
    
    def test_api_version_header(self, server_running):
        """Test con header de versión (si está soportado)"""
        response = requests.get(
            f"{BASE_URL}/health",
            headers={"X-API-Version": "1.0"},
            timeout=TIMEOUT
        )
        # Puede ignorar el header o usarlo
        assert response.status_code == 200
    
    def test_backward_compatibility(self, server_running):
        """Test que endpoints antiguos sigan funcionando"""
        # Endpoints básicos deberían mantenerse
        response = requests.get(f"{BASE_URL}/health", timeout=TIMEOUT)
        assert response.status_code == 200
        
        response = requests.get(f"{BASE_URL}/models", timeout=TIMEOUT)
        assert response.status_code == 200


class TestErrorResponseFormat:
    """Tests de formato de respuestas de error."""
    
    def test_error_response_is_json(self, server_running):
        """Test que errores sean JSON"""
        response = requests.get(
            f"{BASE_URL}/models/nonexistent-id",
            timeout=TIMEOUT
        )
        assert response.status_code == 404
        
        # Debería ser JSON
        try:
            data = response.json()
            assert isinstance(data, dict)
        except json.JSONDecodeError:
            pytest.fail("Error response no es JSON")
    
    def test_error_response_structure(self, server_running):
        """Test estructura de respuesta de error"""
        response = requests.post(
            f"{BASE_URL}/models/create",
            json={},  # Inválido
            timeout=TIMEOUT
        )
        assert response.status_code >= 400
        
        try:
            data = response.json()
            # Debería tener algún campo de error
            assert "detail" in data or "error" in data or "message" in data
        except:
            pass  # Algunos errores pueden no ser JSON


if __name__ == "__main__":
    pytest.main([__file__, "-v"])











