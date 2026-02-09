"""
Patrones refactorizados comunes para tests
Elimina duplicación y proporciona utilidades reutilizables
"""

import pytest
from typing import Dict, Any, Optional, List, Callable
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from fastapi import FastAPI, APIRouter
from fastapi.testclient import TestClient
from contextlib import contextmanager
import io
import numpy as np


class RefactoredTestClient:
    """Cliente de test refactorizado que maneja mocks automáticamente"""
    
    def __init__(self, router: APIRouter, mocks: Dict[str, Any] = None):
        self.router = router
        self.mocks = mocks or {}
        self.app = FastAPI()
        self.app.include_router(router)
        self.patches = []
        self._setup_mocks()
    
    def _setup_mocks(self):
        """Configura los mocks"""
        for path, mock_value in self.mocks.items():
            patch_obj = patch(path, return_value=mock_value)
            patch_obj.start()
            self.patches.append(patch_obj)
    
    def __enter__(self):
        """Context manager entry"""
        return TestClient(self.app)
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - limpia patches"""
        for patch_obj in self.patches:
            try:
                patch_obj.stop()
            except Exception:
                pass
        self.patches.clear()


def create_refactored_client(
    router: APIRouter,
    mocks: Optional[Dict[str, Any]] = None,
    dependencies: Optional[Dict[str, Any]] = None
) -> RefactoredTestClient:
    """
    Crea un cliente refactorizado con mocks configurados.
    
    Args:
        router: Router de FastAPI
        mocks: Diccionario de mocks
        dependencies: Dependencias adicionales
        
    Returns:
        RefactoredTestClient
    """
    all_mocks = {}
    if mocks:
        all_mocks.update(mocks)
    if dependencies:
        all_mocks.update(dependencies)
    
    return RefactoredTestClient(router, all_mocks)


@contextmanager
def mock_service_context(service_path: str, service_mock: Mock):
    """
    Context manager para mockear un servicio.
    
    Usage:
        with mock_service_context("path.to.service", mock_service):
            # código que usa el servicio
            pass
    """
    with patch(service_path, return_value=service_mock):
        yield


@contextmanager
def mock_multiple_services(services: Dict[str, Any]):
    """
    Context manager para mockear múltiples servicios.
    
    Usage:
        with mock_multiple_services({
            "path.to.service1": mock1,
            "path.to.service2": mock2
        }):
            # código que usa los servicios
            pass
    """
    patches = []
    try:
        for path, mock_value in services.items():
            patch_obj = patch(path, return_value=mock_value)
            patch_obj.start()
            patches.append(patch_obj)
        yield
    finally:
        for patch_obj in patches:
            patch_obj.stop()


def create_standard_mock_service(
    service_class,
    return_values: Dict[str, Any] = None,
    async_methods: List[str] = None
) -> Mock:
    """
    Crea un mock estándar de servicio con valores de retorno predefinidos.
    
    Args:
        service_class: Clase del servicio
        return_values: Valores de retorno para métodos
        async_methods: Lista de métodos que son asíncronos
        
    Returns:
        Mock del servicio
    """
    service = Mock(spec=service_class)
    
    if return_values:
        for method_name, return_value in return_values.items():
            if method_name in (async_methods or []):
                setattr(service, method_name, AsyncMock(return_value=return_value))
            elif callable(return_value):
                setattr(service, method_name, return_value)
            else:
                setattr(service, method_name, Mock(return_value=return_value))
    
    return service


def assert_standard_api_response(
    response,
    expected_status: int = 200,
    required_fields: Optional[List[str]] = None,
    response_type: Optional[str] = None
):
    """
    Verifica una respuesta de API estándar.
    
    Args:
        response: Respuesta HTTP
        expected_status: Status code esperado
        required_fields: Campos requeridos en la respuesta
        response_type: Tipo de respuesta esperado (json, stream, etc.)
    """
    assert response.status_code == expected_status, \
        f"Expected status {expected_status}, got {response.status_code}: {response.text}"
    
    if required_fields:
        if response_type == "json" or response.headers.get("content-type", "").startswith("application/json"):
            data = response.json()
            for field in required_fields:
                assert field in data, f"Missing required field '{field}' in response"
        else:
            # Para otros tipos, verificar que la respuesta no esté vacía
            assert len(response.content) > 0, "Response content is empty"


def assert_list_response_structure(
    response,
    min_items: int = 0,
    item_validator: Optional[Callable] = None
):
    """
    Verifica la estructura de una respuesta de lista.
    
    Args:
        response: Respuesta HTTP
        min_items: Número mínimo de items
        item_validator: Función para validar cada item
    """
    assert response.status_code == 200
    
    data = response.json()
    
    # Puede ser lista directa o objeto con items
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        items = data.get("items", data.get("results", []))
    else:
        items = []
    
    assert len(items) >= min_items, \
        f"Expected at least {min_items} items, got {len(items)}"
    
    if item_validator:
        for item in items:
            item_validator(item)


def create_test_audio_data(
    duration: float = 1.0,
    sample_rate: int = 44100,
    channels: int = 1,
    dtype: type = np.float32
) -> np.ndarray:
    """
    Crea datos de audio de prueba estandarizados.
    
    Args:
        duration: Duración en segundos
        sample_rate: Sample rate
        channels: Número de canales
        dtype: Tipo de dato
        
    Returns:
        Array de audio
    """
    samples = int(sample_rate * duration)
    if channels == 1:
        return np.random.randn(samples).astype(dtype)
    else:
        return np.random.randn(channels, samples).astype(dtype)


def create_test_audio_file_bytes(
    duration: float = 1.0,
    format: str = "wav"
) -> io.BytesIO:
    """
    Crea un archivo de audio en memoria para tests.
    
    Args:
        duration: Duración en segundos
        format: Formato del archivo
        
    Returns:
        BytesIO con datos de audio
    """
    # Simular archivo de audio (simplificado para tests)
    file_data = io.BytesIO()
    file_data.write(b"fake audio content for testing")
    file_data.seek(0)
    return file_data


class StandardTestMixin:
    """Mixin con métodos estándar para tests"""
    
    def assert_success(self, response, expected_status: int = 200):
        """Verifica respuesta exitosa"""
        assert response.status_code == expected_status, \
            f"Expected {expected_status}, got {response.status_code}: {response.text}"
    
    def assert_error(self, response, expected_status: int = 400):
        """Verifica respuesta de error"""
        assert response.status_code == expected_status, \
            f"Expected error {expected_status}, got {response.status_code}"
        data = response.json()
        assert "detail" in data or "message" in data
    
    def assert_contains_keys(self, response, keys: List[str]):
        """Verifica que respuesta contenga claves"""
        data = response.json()
        for key in keys:
            assert key in data, f"Missing key '{key}'"
    
    def assert_valid_uuid(self, uuid_string: str):
        """Verifica UUID válido"""
        import uuid
        try:
            uuid.UUID(uuid_string)
        except ValueError:
            pytest.fail(f"'{uuid_string}' is not a valid UUID")


@pytest.fixture
def refactored_client_factory():
    """Factory para crear clientes refactorizados"""
    return create_refactored_client


@pytest.fixture
def standard_mock_service_factory():
    """Factory para crear mocks estándar de servicios"""
    return create_standard_mock_service


@pytest.fixture
def test_audio_data_factory():
    """Factory para crear datos de audio de prueba"""
    return create_test_audio_data


@pytest.fixture
def test_audio_file_factory():
    """Factory para crear archivos de audio de prueba"""
    return create_test_audio_file_bytes



