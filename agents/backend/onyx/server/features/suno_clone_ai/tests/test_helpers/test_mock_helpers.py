"""
Helpers mejorados para crear mocks y fixtures
"""

import pytest
from unittest.mock import Mock, AsyncMock, MagicMock, patch
from typing import Any, Dict, List, Optional, Callable
from fastapi import FastAPI, TestClient
from fastapi.testclient import TestClient as FastAPITestClient
import numpy as np
import io


def create_mock_service(service_class, methods: Dict[str, Any] = None) -> Mock:
    """
    Crea un mock de servicio con métodos predefinidos.
    
    Args:
        service_class: Clase del servicio a mockear
        methods: Diccionario de métodos y sus valores de retorno
        
    Returns:
        Mock del servicio
    """
    service = Mock(spec=service_class)
    if methods:
        for method_name, return_value in methods.items():
            if isinstance(return_value, Callable):
                setattr(service, method_name, return_value)
            else:
                setattr(service, method_name, Mock(return_value=return_value))
    return service


def create_async_mock_service(service_class, methods: Dict[str, Any] = None) -> AsyncMock:
    """
    Crea un mock asíncrono de servicio.
    
    Args:
        service_class: Clase del servicio a mockear
        methods: Diccionario de métodos y sus valores de retorno
        
    Returns:
        AsyncMock del servicio
    """
    service = AsyncMock(spec=service_class)
    if methods:
        for method_name, return_value in methods.items():
            if isinstance(return_value, Callable):
                setattr(service, method_name, return_value)
            else:
                setattr(service, method_name, AsyncMock(return_value=return_value))
    return service


def create_test_client_with_mocks(
    router,
    mocks: Dict[str, Any],
    dependencies: Dict[str, Any] = None
) -> FastAPITestClient:
    """
    Crea un cliente de prueba FastAPI con mocks configurados.
    
    Args:
        router: Router de FastAPI
        mocks: Diccionario de mocks a aplicar
        dependencies: Dependencias adicionales
        
    Returns:
        TestClient configurado
    """
    app = FastAPI()
    app.include_router(router)
    
    patches = []
    for path, mock_value in mocks.items():
        patches.append(patch(path, return_value=mock_value))
    
    if dependencies:
        for path, dep_value in dependencies.items():
            patches.append(patch(path, return_value=dep_value))
    
    # Aplicar todos los patches
    for p in patches:
        p.start()
    
    client = FastAPITestClient(app)
    
    # Guardar patches para cleanup
    client._patches = patches
    
    return client


def create_sample_audio_data(
    duration: float = 1.0,
    sample_rate: int = 44100,
    channels: int = 1
) -> np.ndarray:
    """
    Crea datos de audio de prueba.
    
    Args:
        duration: Duración en segundos
        sample_rate: Sample rate
        channels: Número de canales
        
    Returns:
        Array de audio
    """
    samples = int(sample_rate * duration)
    if channels == 1:
        return np.random.randn(samples).astype(np.float32)
    else:
        return np.random.randn(channels, samples).astype(np.float32)


def create_sample_audio_file(
    duration: float = 1.0,
    sample_rate: int = 44100,
    format: str = "wav"
) -> io.BytesIO:
    """
    Crea un archivo de audio de prueba en memoria.
    
    Args:
        duration: Duración en segundos
        sample_rate: Sample rate
        format: Formato del archivo
        
    Returns:
        BytesIO con datos de audio
    """
    audio_data = create_sample_audio_data(duration, sample_rate)
    
    # Simular archivo de audio (simplificado)
    file_data = io.BytesIO()
    file_data.write(b"fake audio content")
    file_data.seek(0)
    
    return file_data


def create_mock_user(user_id: str = "test-user-123", role: str = "user") -> Dict[str, Any]:
    """
    Crea un mock de usuario.
    
    Args:
        user_id: ID del usuario
        role: Rol del usuario
        
    Returns:
        Diccionario con datos de usuario
    """
    return {
        "user_id": user_id,
        "email": f"{user_id}@example.com",
        "role": role,
        "sub": user_id
    }


def create_mock_song(
    song_id: str = "song-123",
    user_id: str = "user-123",
    status: str = "completed"
) -> Dict[str, Any]:
    """
    Crea un mock de canción.
    
    Args:
        song_id: ID de la canción
        user_id: ID del usuario
        status: Estado de la canción
        
    Returns:
        Diccionario con datos de canción
    """
    return {
        "song_id": song_id,
        "user_id": user_id,
        "prompt": "Test song",
        "status": status,
        "file_path": f"/tmp/{song_id}.wav",
        "duration": 30.0,
        "created_at": "2024-01-01T00:00:00"
    }


def create_mock_playlist(
    playlist_id: str = "playlist-123",
    user_id: str = "user-123",
    song_count: int = 5
) -> Dict[str, Any]:
    """
    Crea un mock de playlist.
    
    Args:
        playlist_id: ID de la playlist
        user_id: ID del usuario
        song_count: Número de canciones
        
    Returns:
        Diccionario con datos de playlist
    """
    return {
        "playlist_id": playlist_id,
        "user_id": user_id,
        "name": "Test Playlist",
        "songs": [f"song-{i}" for i in range(song_count)],
        "created_at": "2024-01-01T00:00:00"
    }


def assert_dict_contains(dict_obj: Dict[str, Any], keys: List[str]) -> None:
    """
    Verifica que un diccionario contenga las claves especificadas.
    
    Args:
        dict_obj: Diccionario a verificar
        keys: Lista de claves requeridas
    """
    for key in keys:
        assert key in dict_obj, f"Missing key '{key}' in dictionary"


def assert_list_not_empty(list_obj: List[Any]) -> None:
    """
    Verifica que una lista no esté vacía.
    
    Args:
        list_obj: Lista a verificar
    """
    assert len(list_obj) > 0, "List is empty"


def assert_response_contains_keys(response: Any, keys: List[str]) -> None:
    """
    Verifica que una respuesta contenga las claves especificadas.
    
    Args:
        response: Respuesta (puede ser Response o dict)
        keys: Lista de claves requeridas
    """
    if hasattr(response, 'json'):
        data = response.json()
    else:
        data = response
    
    assert_dict_contains(data, keys)


def create_mock_error_response(
    status_code: int = 400,
    detail: str = "Test error"
) -> Dict[str, Any]:
    """
    Crea una respuesta de error mock.
    
    Args:
        status_code: Código de estado HTTP
        detail: Detalle del error
        
    Returns:
        Diccionario con respuesta de error
    """
    return {
        "detail": detail,
        "status_code": status_code
    }


@pytest.fixture
def mock_user_factory():
    """Factory para crear mocks de usuario"""
    return create_mock_user


@pytest.fixture
def mock_song_factory():
    """Factory para crear mocks de canción"""
    return create_mock_song


@pytest.fixture
def mock_playlist_factory():
    """Factory para crear mocks de playlist"""
    return create_mock_playlist


@pytest.fixture
def sample_audio_data_factory():
    """Factory para crear datos de audio de prueba"""
    return create_sample_audio_data


@pytest.fixture
def sample_audio_file_factory():
    """Factory para crear archivos de audio de prueba"""
    return create_sample_audio_file



