"""
Clases base para tests que eliminan duplicación de código
"""

import pytest
from typing import Dict, Any, Optional, List
from unittest.mock import Mock, AsyncMock, patch
from fastapi import FastAPI, APIRouter
from fastapi.testclient import TestClient


class BaseAPITestCase:
    """Clase base para tests de API que elimina duplicación"""
    
    router: Optional[APIRouter] = None
    router_path: Optional[str] = None
    
    def setup_method(self):
        """Setup común para todos los tests"""
        self.app = FastAPI()
        if self.router:
            self.app.include_router(self.router)
        self.patches = []
    
    def teardown_method(self):
        """Cleanup común para todos los tests"""
        for patch_obj in self.patches:
            try:
                patch_obj.stop()
            except Exception:
                pass
        self.patches.clear()
    
    def create_client(
        self,
        mocks: Dict[str, Any],
        dependencies: Optional[Dict[str, Any]] = None
    ) -> TestClient:
        """
        Crea un TestClient con mocks configurados.
        
        Args:
            mocks: Diccionario de mocks a aplicar
            dependencies: Dependencias adicionales
            
        Returns:
            TestClient configurado
        """
        # Aplicar patches
        all_mocks = {**mocks}
        if dependencies:
            all_mocks.update(dependencies)
        
        for path, mock_value in all_mocks.items():
            patch_obj = patch(path, return_value=mock_value)
            patch_obj.start()
            self.patches.append(patch_obj)
        
        return TestClient(self.app)
    
    def assert_success_response(self, response, expected_status: int = 200):
        """Verifica que una respuesta sea exitosa"""
        assert response.status_code == expected_status, \
            f"Expected status {expected_status}, got {response.status_code}: {response.text}"
    
    def assert_error_response(self, response, expected_status: int = 400):
        """Verifica que una respuesta sea un error"""
        assert response.status_code == expected_status, \
            f"Expected error status {expected_status}, got {response.status_code}"
        data = response.json()
        assert "detail" in data or "message" in data
    
    def assert_response_contains_keys(self, response, keys: List[str]):
        """Verifica que una respuesta contenga las claves especificadas"""
        data = response.json()
        for key in keys:
            assert key in data, f"Missing key '{key}' in response"


class BaseServiceTestCase:
    """Clase base para tests de servicios"""
    
    service_class = None
    
    def setup_method(self):
        """Setup común para tests de servicios"""
        self.patches = []
    
    def teardown_method(self):
        """Cleanup común para tests de servicios"""
        for patch_obj in self.patches:
            try:
                patch_obj.stop()
            except Exception:
                pass
        self.patches.clear()
    
    def create_mock_service(self, methods: Optional[Dict[str, Any]] = None) -> Mock:
        """Crea un mock del servicio"""
        service = Mock(spec=self.service_class)
        if methods:
            for method_name, return_value in methods.items():
                if callable(return_value):
                    setattr(service, method_name, return_value)
                else:
                    setattr(service, method_name, Mock(return_value=return_value))
        return service
    
    def create_async_mock_service(self, methods: Optional[Dict[str, Any]] = None) -> AsyncMock:
        """Crea un mock asíncrono del servicio"""
        service = AsyncMock(spec=self.service_class)
        if methods:
            for method_name, return_value in methods.items():
                if callable(return_value):
                    setattr(service, method_name, return_value)
                else:
                    setattr(service, method_name, AsyncMock(return_value=return_value))
        return service


class BaseRouteTestMixin:
    """Mixin para tests de rutas que proporciona helpers comunes"""
    
    def assert_route_exists(self, client: TestClient, method: str, path: str):
        """Verifica que una ruta exista"""
        methods = {
            "GET": client.get,
            "POST": client.post,
            "PUT": client.put,
            "DELETE": client.delete,
            "PATCH": client.patch
        }
        
        if method not in methods:
            pytest.fail(f"Unsupported HTTP method: {method}")
        
        # Intentar hacer una request (puede fallar por validación, pero la ruta debe existir)
        try:
            response = methods[method](path)
            # Si es 404, la ruta no existe
            assert response.status_code != 404, f"Route {method} {path} does not exist"
        except Exception as e:
            # Si hay excepción, verificar que no sea 404
            if "404" in str(e) or "Not Found" in str(e):
                pytest.fail(f"Route {method} {path} does not exist")
    
    def assert_requires_auth(self, client: TestClient, method: str, path: str):
        """Verifica que una ruta requiera autenticación"""
        methods = {
            "GET": client.get,
            "POST": client.post,
            "PUT": client.put,
            "DELETE": client.delete
        }
        
        if method not in methods:
            pytest.fail(f"Unsupported HTTP method: {method}")
        
        response = methods[method](path)
        # Debería retornar 401 o 403 sin autenticación
        assert response.status_code in [401, 403, 422], \
            f"Route {method} {path} should require authentication"


@pytest.fixture
def base_api_test_case():
    """Fixture que proporciona una instancia de BaseAPITestCase"""
    return BaseAPITestCase()


@pytest.fixture
def base_service_test_case():
    """Fixture que proporciona una instancia de BaseServiceTestCase"""
    return BaseServiceTestCase()



