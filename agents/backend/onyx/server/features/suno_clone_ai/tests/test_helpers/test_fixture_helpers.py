"""
Helpers para crear fixtures avanzadas y reutilizables
"""

import pytest
from typing import Any, Dict, List, Optional, Callable, Generator
from unittest.mock import Mock, AsyncMock, MagicMock
from fastapi import FastAPI, TestClient
from pathlib import Path
import tempfile
import shutil
import numpy as np
import io


def create_fastapi_app_with_router(router, dependencies: Dict[str, Any] = None) -> FastAPI:
    """
    Crea una app FastAPI con un router y dependencias mockeadas.
    
    Args:
        router: Router de FastAPI
        dependencies: Diccionario de dependencias a mockear
        
    Returns:
        FastAPI app configurada
    """
    app = FastAPI()
    app.include_router(router)
    
    if dependencies:
        for dep_path, mock_value in dependencies.items():
            # Aplicar patches si es necesario
            pass
    
    return app


def create_test_client_with_dependencies(
    router,
    mocks: Dict[str, Any],
    dependencies: Dict[str, Any] = None
) -> TestClient:
    """
    Crea un TestClient con mocks y dependencias configuradas.
    
    Args:
        router: Router de FastAPI
        mocks: Diccionario de mocks a aplicar
        dependencies: Dependencias adicionales
        
    Returns:
        TestClient configurado
    """
    from unittest.mock import patch
    
    app = FastAPI()
    app.include_router(router)
    
    patches = []
    for path, mock_value in mocks.items():
        patches.append(patch(path, return_value=mock_value))
    
    if dependencies:
        for path, dep_value in dependencies.items():
            patches.append(patch(path, return_value=dep_value))
    
    for p in patches:
        p.start()
    
    client = TestClient(app)
    client._patches = patches
    
    return client


@pytest.fixture
def temp_test_dir() -> Generator[Path, None, None]:
    """Crea un directorio temporal para tests"""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def temp_audio_file(temp_test_dir: Path) -> Path:
    """Crea un archivo de audio temporal"""
    audio_file = temp_test_dir / "test_audio.wav"
    # Crear archivo de audio simulado
    audio_file.write_bytes(b"fake audio content")
    return audio_file


@pytest.fixture
def sample_audio_array() -> np.ndarray:
    """Genera un array de audio de prueba"""
    duration = 1.0
    sample_rate = 44100
    samples = int(sample_rate * duration)
    return np.random.randn(samples).astype(np.float32)


@pytest.fixture
def sample_audio_bytes() -> bytes:
    """Genera bytes de audio de prueba"""
    return b"fake audio content for testing"


@pytest.fixture
def sample_audio_file_io() -> io.BytesIO:
    """Crea un BytesIO con datos de audio"""
    return io.BytesIO(b"fake audio content")


@pytest.fixture
def mock_user_factory():
    """Factory para crear mocks de usuario"""
    def _create_user(user_id: str = "test-user-123", role: str = "user"):
        return {
            "user_id": user_id,
            "email": f"{user_id}@example.com",
            "role": role,
            "sub": user_id
        }
    return _create_user


@pytest.fixture
def mock_song_factory():
    """Factory para crear mocks de canción"""
    def _create_song(
        song_id: str = "song-123",
        user_id: str = "user-123",
        status: str = "completed"
    ):
        return {
            "song_id": song_id,
            "user_id": user_id,
            "prompt": "Test song",
            "status": status,
            "file_path": f"/tmp/{song_id}.wav",
            "duration": 30.0
        }
    return _create_song


@pytest.fixture
def mock_playlist_factory():
    """Factory para crear mocks de playlist"""
    def _create_playlist(
        playlist_id: str = "playlist-123",
        user_id: str = "user-123",
        song_count: int = 5
    ):
        return {
            "playlist_id": playlist_id,
            "user_id": user_id,
            "name": "Test Playlist",
            "songs": [f"song-{i}" for i in range(song_count)]
        }
    return _create_playlist


@pytest.fixture
def mock_service_factory():
    """Factory para crear mocks de servicios"""
    def _create_service(service_class, methods: Dict[str, Any] = None):
        service = Mock(spec=service_class)
        if methods:
            for method_name, return_value in methods.items():
                if callable(return_value):
                    setattr(service, method_name, return_value)
                else:
                    setattr(service, method_name, Mock(return_value=return_value))
        return service
    return _create_service


@pytest.fixture
def mock_async_service_factory():
    """Factory para crear mocks asíncronos de servicios"""
    def _create_async_service(service_class, methods: Dict[str, Any] = None):
        service = AsyncMock(spec=service_class)
        if methods:
            for method_name, return_value in methods.items():
                if callable(return_value):
                    setattr(service, method_name, return_value)
                else:
                    setattr(service, method_name, AsyncMock(return_value=return_value))
        return service
    return _create_async_service


@pytest.fixture
def assert_response_helper():
    """Helper para aserciones de respuestas"""
    def _assert_response(
        response,
        expected_status: int = 200,
        expected_keys: Optional[List[str]] = None,
        expected_type: Optional[str] = None
    ):
        assert response.status_code == expected_status, \
            f"Expected status {expected_status}, got {response.status_code}"
        
        if expected_keys:
            data = response.json() if hasattr(response, 'json') else response
            for key in expected_keys:
                assert key in data, f"Missing key '{key}' in response"
        
        if expected_type:
            content_type = response.headers.get("content-type", "")
            assert expected_type in content_type.lower(), \
                f"Expected content type '{expected_type}' not found"
    
    return _assert_response


@pytest.fixture
def cleanup_helper():
    """Helper para limpiar recursos después de tests"""
    cleanup_items = []
    
    def _register_cleanup(item: Any, cleanup_func: Callable):
        cleanup_items.append((item, cleanup_func))
    
    yield _register_cleanup
    
    # Ejecutar cleanup
    for item, cleanup_func in cleanup_items:
        try:
            cleanup_func(item)
        except Exception as e:
            print(f"Error during cleanup: {e}")



