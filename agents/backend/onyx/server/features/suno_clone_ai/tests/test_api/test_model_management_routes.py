"""
Tests para las rutas de gestión de modelos
"""

import pytest
from unittest.mock import Mock, patch
from fastapi import status
from fastapi.testclient import TestClient

from api.routes.model_management import router
from services.model_optimizer import ModelOptimizer


@pytest.fixture
def mock_model_optimizer():
    """Mock del optimizador de modelos"""
    optimizer = Mock(spec=ModelOptimizer)
    optimizer.optimize_model = Mock(return_value="/path/to/optimized_model")
    optimizer.save_model_version = Mock(return_value="/path/to/versioned_model")
    optimizer.list_versions = Mock(return_value=["v1.0", "v1.1", "v2.0"])
    optimizer.compare_models = Mock(return_value={"difference": 0.05})
    return optimizer


@pytest.fixture
def client(mock_model_optimizer):
    """Cliente de prueba con mocks"""
    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(router)
    
    with patch('api.routes.model_management.get_model_optimizer', return_value=mock_model_optimizer):
        with patch('api.routes.model_management.require_role', return_value=lambda: None):
            yield TestClient(app)


@pytest.mark.unit
@pytest.mark.api
class TestOptimizeModel:
    """Tests para optimizar modelo"""
    
    def test_optimize_model_success(self, client, mock_model_optimizer):
        """Test de optimización exitosa"""
        response = client.post(
            "/models/optimize",
            json={
                "model_path": "/path/to/model",
                "optimizations": ["quantize", "compile"],
                "save_path": "/path/to/optimized"
            }
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "message" in data
        assert data["model_path"] == "/path/to/model"
        assert "optimizations" in data
    
    def test_optimize_model_different_optimizations(self, client, mock_model_optimizer):
        """Test con diferentes optimizaciones"""
        optimization_sets = [
            ["quantize"],
            ["compile"],
            ["quantize", "compile"],
            ["prune", "quantize"]
        ]
        
        for optimizations in optimization_sets:
            response = client.post(
                "/models/optimize",
                json={
                    "model_path": "/path/to/model",
                    "optimizations": optimizations
                }
            )
            assert response.status_code == status.HTTP_200_OK


@pytest.mark.unit
@pytest.mark.api
class TestSaveModelVersion:
    """Tests para guardar versión de modelo"""
    
    def test_save_model_version_success(self, client, mock_model_optimizer):
        """Test de guardado exitoso de versión"""
        response = client.post(
            "/models/versions",
            json={
                "version": "v1.0",
                "model_path": "/path/to/model",
                "metadata": {"accuracy": 0.95}
            }
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "message" in data
        assert data["version"] == "v1.0"
        assert "path" in data


@pytest.mark.unit
@pytest.mark.api
class TestListVersions:
    """Tests para listar versiones"""
    
    def test_list_versions_success(self, client, mock_model_optimizer):
        """Test de listado exitoso"""
        response = client.get("/models/versions")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "versions" in data or isinstance(data, list)


@pytest.mark.unit
@pytest.mark.api
class TestCompareModels:
    """Tests para comparar modelos"""
    
    def test_compare_models_success(self, client, mock_model_optimizer):
        """Test de comparación exitosa"""
        response = client.post(
            "/models/compare",
            json={
                "model1_path": "/path/to/model1",
                "model2_path": "/path/to/model2"
            }
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert isinstance(data, dict)


@pytest.mark.integration
@pytest.mark.api
class TestModelManagementIntegration:
    """Tests de integración para gestión de modelos"""
    
    def test_full_model_management_workflow(self, client, mock_model_optimizer):
        """Test del flujo completo de gestión de modelos"""
        # 1. Optimizar modelo
        optimize_response = client.post(
            "/models/optimize",
            json={
                "model_path": "/path/to/model",
                "optimizations": ["quantize"]
            }
        )
        assert optimize_response.status_code == status.HTTP_200_OK
        
        # 2. Guardar versión
        version_response = client.post(
            "/models/versions",
            json={
                "version": "v1.0",
                "model_path": "/path/to/model"
            }
        )
        assert version_response.status_code == status.HTTP_200_OK
        
        # 3. Listar versiones
        list_response = client.get("/models/versions")
        assert list_response.status_code == status.HTTP_200_OK
        
        # 4. Comparar modelos
        compare_response = client.post(
            "/models/compare",
            json={
                "model1_path": "/path/to/model1",
                "model2_path": "/path/to/model2"
            }
        )
        assert compare_response.status_code == status.HTTP_200_OK



