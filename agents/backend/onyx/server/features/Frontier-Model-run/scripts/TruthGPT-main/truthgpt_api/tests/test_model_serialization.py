"""
Tests de Serialización de Modelos
=================================
Tests para guardar, cargar y serializar modelos
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


class TestModelInfo:
    """Tests para obtener información de modelos."""
    
    def test_get_model_info(self, server_running):
        """Test obtener información de un modelo"""
        # Crear modelo
        create_response = requests.post(
            f"{BASE_URL}/models/create",
            json={
                "layers": [
                    {"type": "dense", "params": {"units": 64, "activation": "relu"}},
                    {"type": "dense", "params": {"units": 10, "activation": "softmax"}}
                ],
                "name": "test-model-info"
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
        assert "model_id" in data
        assert "name" in data
        assert "layers" in data or "spec" in data
    
    def test_get_nonexistent_model_info(self, server_running):
        """Test obtener info de modelo inexistente"""
        response = requests.get(
            f"{BASE_URL}/models/nonexistent-id-12345",
            timeout=TIMEOUT
        )
        assert response.status_code == 404
    
    def test_get_model_info_after_compile(self, server_running):
        """Test obtener info después de compilar"""
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
        
        # Compilar
        requests.post(
            f"{BASE_URL}/models/{model_id}/compile",
            json={"optimizer": "adam", "loss": "mse"},
            timeout=TIMEOUT
        )
        
        # Obtener info
        info_response = requests.get(
            f"{BASE_URL}/models/{model_id}",
            timeout=TIMEOUT
        )
        
        assert info_response.status_code == 200
        data = info_response.json()
        # Debería tener información de compilación
        assert "model_id" in data


class TestModelList:
    """Tests para listar modelos."""
    
    def test_list_empty_models(self, server_running):
        """Test listar cuando no hay modelos"""
        response = requests.get(f"{BASE_URL}/models", timeout=TIMEOUT)
        assert response.status_code == 200
        data = response.json()
        assert "count" in data
        assert "models" in data
    
    def test_list_models_after_creation(self, server_running):
        """Test listar después de crear modelos"""
        # Crear algunos modelos
        model_ids = []
        for i in range(3):
            response = requests.post(
                f"{BASE_URL}/models/create",
                json={
                    "layers": [
                        {"type": "dense", "params": {"units": 16}}
                    ],
                    "name": f"list-test-{i}"
                },
                timeout=TIMEOUT
            )
            if response.status_code == 200:
                model_ids.append(response.json()["model_id"])
        
        # Listar
        list_response = requests.get(f"{BASE_URL}/models", timeout=TIMEOUT)
        assert list_response.status_code == 200
        data = list_response.json()
        assert data["count"] >= len(model_ids)
        assert len(data["models"]) >= len(model_ids)
    
    def test_list_models_with_pagination(self, server_running):
        """Test listar modelos (simulando paginación)"""
        # Crear varios modelos
        for i in range(5):
            requests.post(
                f"{BASE_URL}/models/create",
                json={
                    "layers": [
                        {"type": "dense", "params": {"units": 8}}
                    ],
                    "name": f"pagination-test-{i}"
                },
                timeout=TIMEOUT
            )
        
        # Listar
        response = requests.get(f"{BASE_URL}/models", timeout=TIMEOUT)
        assert response.status_code == 200
        data = response.json()
        assert data["count"] >= 5


class TestModelDeletion:
    """Tests para eliminar modelos."""
    
    def test_delete_model(self, server_running):
        """Test eliminar un modelo"""
        # Crear modelo
        create_response = requests.post(
            f"{BASE_URL}/models/create",
            json={
                "layers": [
                    {"type": "dense", "params": {"units": 16}}
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
        
        # Debería eliminarse exitosamente
        assert delete_response.status_code in [200, 204]
        
        # Verificar que ya no existe
        info_response = requests.get(
            f"{BASE_URL}/models/{model_id}",
            timeout=TIMEOUT
        )
        assert info_response.status_code == 404
    
    def test_delete_nonexistent_model(self, server_running):
        """Test eliminar modelo inexistente"""
        response = requests.delete(
            f"{BASE_URL}/models/nonexistent-id-12345",
            timeout=TIMEOUT
        )
        assert response.status_code == 404
    
    def test_delete_compiled_model(self, server_running):
        """Test eliminar modelo compilado"""
        # Crear y compilar
        create_response = requests.post(
            f"{BASE_URL}/models/create",
            json={
                "layers": [
                    {"type": "dense", "params": {"units": 16}}
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
        
        # Eliminar
        delete_response = requests.delete(
            f"{BASE_URL}/models/{model_id}",
            timeout=TIMEOUT
        )
        assert delete_response.status_code in [200, 204]


class TestModelState:
    """Tests para el estado de modelos."""
    
    def test_model_state_before_compile(self, server_running):
        """Test estado antes de compilar"""
        create_response = requests.post(
            f"{BASE_URL}/models/create",
            json={
                "layers": [
                    {"type": "dense", "params": {"units": 16}}
                ]
            },
            timeout=TIMEOUT
        )
        model_id = create_response.json()["model_id"]
        
        info_response = requests.get(
            f"{BASE_URL}/models/{model_id}",
            timeout=TIMEOUT
        )
        assert info_response.status_code == 200
    
    def test_model_state_after_compile(self, server_running):
        """Test estado después de compilar"""
        create_response = requests.post(
            f"{BASE_URL}/models/create",
            json={
                "layers": [
                    {"type": "dense", "params": {"units": 16}}
                ]
            },
            timeout=TIMEOUT
        )
        model_id = create_response.json()["model_id"]
        
        compile_response = requests.post(
            f"{BASE_URL}/models/{model_id}/compile",
            json={"optimizer": "adam", "loss": "mse"},
            timeout=TIMEOUT
        )
        assert compile_response.status_code == 200
        
        info_response = requests.get(
            f"{BASE_URL}/models/{model_id}",
            timeout=TIMEOUT
        )
        assert info_response.status_code == 200
    
    def test_model_state_after_training(self, server_running):
        """Test estado después de entrenar"""
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
        
        x_train = np.random.randn(20, 32).astype(np.float32).tolist()
        y_train = np.random.randn(20, 10).astype(np.float32).tolist()
        
        requests.post(
            f"{BASE_URL}/models/{model_id}/train",
            json={"x_train": x_train, "y_train": y_train, "epochs": 1, "verbose": 0},
            timeout=TIMEOUT * 2
        )
        
        info_response = requests.get(
            f"{BASE_URL}/models/{model_id}",
            timeout=TIMEOUT
        )
        assert info_response.status_code == 200


class TestModelHistory:
    """Tests para historial de entrenamiento."""
    
    def test_training_history_available(self, server_running):
        """Test que el historial esté disponible después de entrenar"""
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
            json={"optimizer": "adam", "loss": "mse", "metrics": ["accuracy"]},
            timeout=TIMEOUT
        )
        
        x_train = np.random.randn(20, 32).astype(np.float32).tolist()
        y_train = np.random.randn(20, 10).astype(np.float32).tolist()
        
        train_response = requests.post(
            f"{BASE_URL}/models/{model_id}/train",
            json={"x_train": x_train, "y_train": y_train, "epochs": 3, "verbose": 0},
            timeout=TIMEOUT * 2
        )
        
        assert train_response.status_code == 200
        data = train_response.json()
        # Debería tener información de historial
        assert "history" in data or "loss" in data or "epochs" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])











