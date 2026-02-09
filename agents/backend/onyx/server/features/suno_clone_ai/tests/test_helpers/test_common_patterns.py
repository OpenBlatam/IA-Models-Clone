"""
Patrones comunes refactorizados para tests
"""

import pytest
from typing import Dict, Any, Optional, List, Callable
from unittest.mock import Mock, patch
from fastapi import FastAPI, APIRouter
from fastapi.testclient import TestClient
from contextlib import contextmanager


@contextmanager
def mock_dependencies(mocks: Dict[str, Any]):
    """
    Context manager para mockear dependencias.
    
    Usage:
        with mock_dependencies({"path.to.service": mock_service}):
            # código que usa el mock
            pass
    """
    patches = []
    try:
        for path, mock_value in mocks.items():
            patch_obj = patch(path, return_value=mock_value)
            patch_obj.start()
            patches.append(patch_obj)
        yield
    finally:
        for patch_obj in patches:
            patch_obj.stop()


def create_router_client(
    router: APIRouter,
    mocks: Optional[Dict[str, Any]] = None,
    dependencies: Optional[Dict[str, Any]] = None
) -> TestClient:
    """
    Crea un TestClient para un router con mocks configurados.
    
    Args:
        router: Router de FastAPI
        mocks: Diccionario de mocks a aplicar
        dependencies: Dependencias adicionales
        
    Returns:
        TestClient configurado
    """
    app = FastAPI()
    app.include_router(router)
    
    all_mocks = {}
    if mocks:
        all_mocks.update(mocks)
    if dependencies:
        all_mocks.update(dependencies)
    
    patches = []
    for path, mock_value in all_mocks.items():
        patch_obj = patch(path, return_value=mock_value)
        patch_obj.start()
        patches.append(patch_obj)
    
    client = TestClient(app)
    client._patches = patches
    
    return client


def create_service_mock(
    service_class,
    methods: Optional[Dict[str, Any]] = None,
    async_methods: Optional[List[str]] = None
) -> Mock:
    """
    Crea un mock de servicio con métodos configurados.
    
    Args:
        service_class: Clase del servicio
        methods: Métodos síncronos y sus valores de retorno
        async_methods: Lista de métodos que deben ser asíncronos
        
    Returns:
        Mock del servicio
    """
    service = Mock(spec=service_class)
    
    if methods:
        for method_name, return_value in methods.items():
            if method_name in (async_methods or []):
                setattr(service, method_name, AsyncMock(return_value=return_value))
            elif callable(return_value):
                setattr(service, method_name, return_value)
            else:
                setattr(service, method_name, Mock(return_value=return_value))
    
    return service


def assert_standard_response(
    response,
    expected_status: int = 200,
    required_keys: Optional[List[str]] = None,
    optional_keys: Optional[List[str]] = None
):
    """
    Verifica una respuesta estándar.
    
    Args:
        response: Respuesta HTTP
        expected_status: Status code esperado
        required_keys: Claves requeridas en la respuesta
        optional_keys: Claves opcionales en la respuesta
    """
    assert response.status_code == expected_status, \
        f"Expected status {expected_status}, got {response.status_code}: {response.text}"
    
    if required_keys:
        data = response.json()
        for key in required_keys:
            assert key in data, f"Missing required key '{key}' in response"
    
    if optional_keys:
        data = response.json()
        # Solo verificar que las claves opcionales existan si están presentes
        for key in optional_keys:
            if key in data:
                assert data[key] is not None, f"Optional key '{key}' is None"


def assert_paginated_response(
    response,
    expected_status: int = 200,
    min_items: int = 0
):
    """
    Verifica una respuesta paginada.
    
    Args:
        response: Respuesta HTTP
        expected_status: Status code esperado
        min_items: Número mínimo de items esperados
    """
    assert response.status_code == expected_status
    data = response.json()
    
    # Puede ser una lista directa o un objeto con items
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict) and "items" in data:
        items = data["items"]
        assert "limit" in data or "page" in data, "Missing pagination metadata"
    else:
        items = []
    
    assert len(items) >= min_items, \
        f"Expected at least {min_items} items, got {len(items)}"


def assert_error_response(
    response,
    expected_status: int = 400,
    error_key: str = "detail"
):
    """
    Verifica una respuesta de error.
    
    Args:
        response: Respuesta HTTP
        expected_status: Status code esperado
        error_key: Clave que contiene el mensaje de error
    """
    assert response.status_code == expected_status, \
        f"Expected error status {expected_status}, got {response.status_code}"
    
    data = response.json()
    assert error_key in data or "message" in data, \
        f"Error response missing '{error_key}' or 'message'"


def create_test_data_factory(entity_type: str):
    """
    Crea una factory para datos de prueba.
    
    Args:
        entity_type: Tipo de entidad (song, playlist, user, etc.)
        
    Returns:
        Función factory
    """
    factories = {
        "song": lambda **kwargs: {
            "song_id": kwargs.get("song_id", "song-123"),
            "user_id": kwargs.get("user_id", "user-123"),
            "prompt": kwargs.get("prompt", "Test song"),
            "status": kwargs.get("status", "completed"),
            **kwargs
        },
        "playlist": lambda **kwargs: {
            "playlist_id": kwargs.get("playlist_id", "playlist-123"),
            "user_id": kwargs.get("user_id", "user-123"),
            "name": kwargs.get("name", "Test Playlist"),
            "songs": kwargs.get("songs", []),
            **kwargs
        },
        "user": lambda **kwargs: {
            "user_id": kwargs.get("user_id", "user-123"),
            "email": kwargs.get("email", "user@example.com"),
            "role": kwargs.get("role", "user"),
            **kwargs
        }
    }
    
    return factories.get(entity_type, lambda **kwargs: kwargs)


@pytest.fixture
def router_client_factory():
    """Factory para crear clientes de router"""
    return create_router_client


@pytest.fixture
def service_mock_factory():
    """Factory para crear mocks de servicios"""
    return create_service_mock


@pytest.fixture
def test_data_factory():
    """Factory para crear datos de prueba"""
    return create_test_data_factory



