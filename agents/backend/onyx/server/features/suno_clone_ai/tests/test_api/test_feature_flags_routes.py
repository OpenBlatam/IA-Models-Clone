"""
Tests para las rutas de feature flags
"""

import pytest
from unittest.mock import Mock, patch
from fastapi import status
from fastapi.testclient import TestClient

from api.routes.feature_flags import router
from utils.feature_flags import FeatureFlagService, FlagType


@pytest.fixture
def mock_feature_flag_service():
    """Mock del servicio de feature flags"""
    service = Mock(spec=FeatureFlagService)
    service.is_enabled = Mock(return_value=True)
    service.list_flags = Mock(return_value=[
        {"name": "new_feature", "type": "boolean", "enabled": True},
        {"name": "beta_feature", "type": "percentage", "enabled": False}
    ])
    service.create_flag = Mock(return_value="flag-123")
    service.update_flag = Mock(return_value=True)
    service.delete_flag = Mock(return_value=True)
    return service


@pytest.fixture
def client(mock_feature_flag_service):
    """Cliente de prueba con mocks"""
    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(router)
    
    with patch('api.routes.feature_flags.get_feature_flag_service', return_value=mock_feature_flag_service):
        with patch('api.routes.feature_flags.get_current_user', return_value={"user_id": "test_user"}):
            with patch('api.routes.feature_flags.require_role', return_value=lambda: None):
                yield TestClient(app)


@pytest.mark.unit
@pytest.mark.api
class TestCheckFeatureFlag:
    """Tests para verificar feature flag"""
    
    def test_check_feature_flag_success(self, client, mock_feature_flag_service):
        """Test de verificación exitosa"""
        response = client.get("/feature-flags/check/new_feature")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "flag_name" in data
        assert "enabled" in data
        assert data["flag_name"] == "new_feature"
    
    def test_check_feature_flag_disabled(self, client, mock_feature_flag_service):
        """Test con flag deshabilitado"""
        mock_feature_flag_service.is_enabled.return_value = False
        
        response = client.get("/feature-flags/check/disabled_feature")
        
        assert response.status_code == status.HTTP_200_OK
        assert response.json()["enabled"] is False


@pytest.mark.unit
@pytest.mark.api
class TestListFeatureFlags:
    """Tests para listar feature flags"""
    
    def test_list_feature_flags_success(self, client, mock_feature_flag_service):
        """Test de listado exitoso"""
        response = client.get("/feature-flags/list")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "flags" in data or isinstance(data, list)


@pytest.mark.unit
@pytest.mark.api
class TestCreateFeatureFlag:
    """Tests para crear feature flag"""
    
    def test_create_feature_flag_success(self, client, mock_feature_flag_service):
        """Test de creación exitosa"""
        response = client.post(
            "/feature-flags",
            json={
                "name": "new_flag",
                "type": "boolean",
                "enabled": True
            }
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "flag_id" in data or "message" in data


@pytest.mark.unit
@pytest.mark.api
class TestUpdateFeatureFlag:
    """Tests para actualizar feature flag"""
    
    def test_update_feature_flag_success(self, client, mock_feature_flag_service):
        """Test de actualización exitosa"""
        response = client.put(
            "/feature-flags/new_feature",
            json={"enabled": False}
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "message" in data


@pytest.mark.integration
@pytest.mark.api
class TestFeatureFlagsIntegration:
    """Tests de integración para feature flags"""
    
    def test_full_feature_flags_workflow(self, client, mock_feature_flag_service):
        """Test del flujo completo de feature flags"""
        # 1. Verificar flag
        check_response = client.get("/feature-flags/check/new_feature")
        assert check_response.status_code == status.HTTP_200_OK
        
        # 2. Listar flags
        list_response = client.get("/feature-flags/list")
        assert list_response.status_code == status.HTTP_200_OK
        
        # 3. Crear flag
        create_response = client.post(
            "/feature-flags",
            json={"name": "test_flag", "type": "boolean", "enabled": True}
        )
        assert create_response.status_code == status.HTTP_200_OK
        
        # 4. Actualizar flag
        update_response = client.put(
            "/feature-flags/test_flag",
            json={"enabled": False}
        )
        assert update_response.status_code == status.HTTP_200_OK



