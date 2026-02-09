"""
Tests refactorizados para utilidades de versionado de API
Usando clases base y helpers para eliminar duplicación
"""

import pytest
from unittest.mock import Mock
from fastapi import Request, HTTPException, status
from fastapi.testclient import TestClient
from fastapi import FastAPI, APIRouter

from api.versioning import (
    VersionedAPIRoute,
    get_api_version,
    validate_api_version,
    create_versioned_router,
    SUPPORTED_VERSIONS
)
from test_helpers import BaseServiceTestCase, StandardTestMixin


class TestVersionedAPIRouteRefactored(BaseServiceTestCase, StandardTestMixin):
    """Tests refactorizados para VersionedAPIRoute"""
    
    def test_versioned_route_creation(self):
        """Test de creación de ruta versionada"""
        route = VersionedAPIRoute(
            path="/test",
            endpoint=lambda: "test",
            version="v1"
        )
        
        assert route.version == "v1"
    
    def test_versioned_route_no_version(self):
        """Test de ruta sin versión"""
        route = VersionedAPIRoute(
            path="/test",
            endpoint=lambda: "test"
        )
        
        assert route.version is None


class TestGetAPIVersionRefactored(BaseServiceTestCase, StandardTestMixin):
    """Tests refactorizados para get_api_version"""
    
    @pytest.mark.parametrize("header_version,path,expected", [
        ("v2", "/api/test", "v2"),
        (None, "/v2/api/test", "v2"),
        (None, "/api/test", "v1"),
        ("v1", "/v2/api/test", "v1")  # Header tiene prioridad
    ])
    def test_get_api_version(self, header_version, path, expected):
        """Test de obtención de versión con diferentes escenarios"""
        request = Mock(spec=Request)
        request.url.path = path
        
        version = get_api_version(request, api_version=header_version)
        
        assert version == expected


class TestValidateAPIVersionRefactored(BaseServiceTestCase, StandardTestMixin):
    """Tests refactorizados para validate_api_version"""
    
    @pytest.mark.parametrize("version,supported,should_raise", [
        ("v1", ["v1", "v2"], False),
        ("v2", ["v1", "v2"], False),
        ("v3", ["v1", "v2"], True),
        ("v1", None, False)  # Default supported
    ])
    def test_validate_api_version(self, version, supported, should_raise):
        """Test de validación de versión"""
        if should_raise:
            with pytest.raises(HTTPException) as exc_info:
                validate_api_version(version, supported)
            
            assert exc_info.value.status_code == status.HTTP_400_BAD_REQUEST
        else:
            result = validate_api_version(version, supported)
            assert result == version


class TestCreateVersionedRouterRefactored(BaseServiceTestCase, StandardTestMixin):
    """Tests refactorizados para create_versioned_router"""
    
    @pytest.mark.parametrize("version,prefix,expected_prefix", [
        ("v1", "", "/v1"),
        ("v2", "/api", "/v2/api"),
        ("v1", "/songs", "/v1/songs")
    ])
    def test_create_versioned_router(self, version, prefix, expected_prefix):
        """Test de creación de router versionado"""
        router = create_versioned_router(version, prefix=prefix)
        
        assert isinstance(router, APIRouter)
        assert router.prefix == expected_prefix
        assert f"API {version.upper()}" in router.tags


class TestVersionedRouterIntegrationRefactored(BaseServiceTestCase, StandardTestMixin):
    """Tests de integración refactorizados para routers versionados"""
    
    def test_versioned_router_endpoint(self):
        """Test de endpoint en router versionado"""
        router = create_versioned_router("v1")
        
        @router.get("/test")
        async def test_endpoint():
            return {"version": "v1"}
        
        app = FastAPI()
        app.include_router(router)
        
        client = TestClient(app)
        response = client.get("/v1/test")
        
        self.assert_success(response)
        assert response.json()["version"] == "v1"
    
    def test_versioned_router_multiple_versions(self):
        """Test de múltiples versiones en la misma app"""
        router_v1 = create_versioned_router("v1")
        router_v2 = create_versioned_router("v2")
        
        @router_v1.get("/test")
        async def test_v1():
            return {"version": "v1"}
        
        @router_v2.get("/test")
        async def test_v2():
            return {"version": "v2"}
        
        app = FastAPI()
        app.include_router(router_v1)
        app.include_router(router_v2)
        
        client = TestClient(app)
        
        response_v1 = client.get("/v1/test")
        self.assert_success(response_v1)
        assert response_v1.json()["version"] == "v1"
        
        response_v2 = client.get("/v2/test")
        self.assert_success(response_v2)
        assert response_v2.json()["version"] == "v2"



