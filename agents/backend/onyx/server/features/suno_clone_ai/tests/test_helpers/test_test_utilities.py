"""
Utilidades avanzadas para tests refactorizados
"""

import pytest
from typing import Dict, Any, Optional, List, Type, Callable
from unittest.mock import Mock, AsyncMock, patch
from fastapi import FastAPI, APIRouter, Depends
from fastapi.testclient import TestClient
from dataclasses import dataclass
import functools
import time


@dataclass
class TestConfig:
    """Configuración para tests"""
    router: APIRouter
    mocks: Dict[str, Any]
    dependencies: Optional[Dict[str, Any]] = None
    auth_user: Optional[Dict[str, Any]] = None


class TestClientBuilder:
    """Builder para crear TestClients con configuración compleja"""
    
    def __init__(self, router: APIRouter):
        self.router = router
        self.mocks = {}
        self.dependencies = {}
        self.auth_user = {"user_id": "test_user"}
    
    def with_mock(self, path: str, mock_value: Any) -> 'TestClientBuilder':
        """Agrega un mock"""
        self.mocks[path] = mock_value
        return self
    
    def with_dependency(self, path: str, dependency: Any) -> 'TestClientBuilder':
        """Agrega una dependencia"""
        self.dependencies[path] = dependency
        return self
    
    def with_auth(self, user: Dict[str, Any]) -> 'TestClientBuilder':
        """Configura usuario autenticado"""
        self.auth_user = user
        return self
    
    def build(self) -> TestClient:
        """Construye el TestClient"""
        app = FastAPI()
        app.include_router(self.router)
        
        all_mocks = {**self.mocks}
        if self.auth_user:
            all_mocks["api.routes.get_current_user"] = self.auth_user
        if self.dependencies:
            all_mocks.update(self.dependencies)
        
        patches = []
        for path, mock_value in all_mocks.items():
            patch_obj = patch(path, return_value=mock_value)
            patch_obj.start()
            patches.append(patch_obj)
        
        client = TestClient(app)
        client._patches = patches
        client._builder = self
        
        return client


def create_test_client_builder(router: APIRouter) -> TestClientBuilder:
    """Crea un builder para TestClient"""
    return TestClientBuilder(router)


def retry_on_failure(max_retries: int = 3, delay: float = 0.1):
    """
    Decorator para reintentar tests que fallan.
    
    Args:
        max_retries: Número máximo de reintentos
        delay: Delay entre reintentos en segundos
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except AssertionError as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        time.sleep(delay)
                        continue
                    raise
            if last_exception:
                raise last_exception
        return wrapper
    return decorator


def parametrize_http_methods(func):
    """
    Decorator para parametrizar tests con diferentes métodos HTTP.
    
    Usage:
        @parametrize_http_methods
        def test_my_endpoint(client, method):
            response = getattr(client, method.lower())("/endpoint")
            assert response.status_code == 200
    """
    return pytest.mark.parametrize("method", ["GET", "POST", "PUT", "DELETE"])(func)


def skip_if_service_unavailable(service_path: str):
    """
    Decorator para saltar tests si un servicio no está disponible.
    
    Usage:
        @skip_if_service_unavailable("services.my_service")
        def test_my_service():
            # test code
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                # Intentar importar el servicio
                import importlib
                module_path, service_name = service_path.rsplit(".", 1)
                module = importlib.import_module(module_path)
                service = getattr(module, service_name, None)
                if service is None:
                    pytest.skip(f"Service {service_path} not available")
            except (ImportError, AttributeError):
                pytest.skip(f"Service {service_path} not available")
            return func(*args, **kwargs)
        return wrapper
    return decorator


class TestDataGenerator:
    """Generador de datos de prueba estandarizados"""
    
    @staticmethod
    def generate_song_data(**overrides) -> Dict[str, Any]:
        """Genera datos de canción"""
        defaults = {
            "song_id": "song-123",
            "user_id": "user-123",
            "prompt": "Test song",
            "status": "completed",
            "duration": 30.0
        }
        defaults.update(overrides)
        return defaults
    
    @staticmethod
    def generate_playlist_data(**overrides) -> Dict[str, Any]:
        """Genera datos de playlist"""
        defaults = {
            "playlist_id": "playlist-123",
            "user_id": "user-123",
            "name": "Test Playlist",
            "songs": []
        }
        defaults.update(overrides)
        return defaults
    
    @staticmethod
    def generate_user_data(**overrides) -> Dict[str, Any]:
        """Genera datos de usuario"""
        defaults = {
            "user_id": "user-123",
            "email": "user@example.com",
            "role": "user"
        }
        defaults.update(overrides)
        return defaults


@pytest.fixture
def test_client_builder():
    """Factory para crear builders de TestClient"""
    return create_test_client_builder


@pytest.fixture
def test_data_generator():
    """Factory para generar datos de prueba"""
    return TestDataGenerator


@pytest.fixture
def standard_test_config():
    """Configuración estándar para tests"""
    return {
        "auth_user": {"user_id": "test_user", "role": "user"},
        "default_timeout": 5.0
    }



