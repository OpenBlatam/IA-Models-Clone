"""
Helpers y utilidades para tests
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, Any, List
import json
import time
import os


class TestHelpers:
    """Clase de helpers para tests"""
    
    @staticmethod
    def create_mock_response(status_code=200, data=None):
        """Crear un mock de respuesta HTTP"""
        mock_response = Mock()
        mock_response.status_code = status_code
        mock_response.json.return_value = data or {}
        mock_response.text = json.dumps(data) if data else ""
        return mock_response
    
    @staticmethod
    def create_mock_spotify_track(track_id="test_123", name="Test Track"):
        """Crear un mock de track de Spotify"""
        return {
            "id": track_id,
            "name": name,
            "artists": [{"name": "Test Artist"}],
            "album": {"name": "Test Album"},
            "duration_ms": 180000,
            "popularity": 50
        }
    
    @staticmethod
    def create_mock_audio_features():
        """Crear mock de características de audio"""
        return {
            "key": 0,
            "mode": 1,
            "tempo": 120.0,
            "time_signature": 4,
            "energy": 0.8,
            "danceability": 0.7,
            "valence": 0.6,
            "acousticness": 0.3,
            "instrumentalness": 0.2,
            "liveness": 0.1,
            "speechiness": 0.05,
            "loudness": -5.5
        }
    
    @staticmethod
    def create_mock_analysis_result():
        """Crear mock de resultado de análisis"""
        return {
            "track_basic_info": {
                "name": "Test Track",
                "artists": ["Test Artist"],
                "duration_ms": 180000
            },
            "musical_analysis": {
                "key": "C",
                "mode": "Major",
                "tempo": 120.0
            },
            "technical_analysis": {
                "energy": 0.8,
                "danceability": 0.7
            }
        }
    
    @staticmethod
    def assert_dict_contains(actual: Dict, expected: Dict):
        """Verificar que un diccionario contiene las claves esperadas"""
        for key, value in expected.items():
            assert key in actual, f"Key '{key}' not found in actual dict"
            assert actual[key] == value, f"Value for '{key}' doesn't match"
    
    @staticmethod
    def assert_response_structure(response: Dict, required_fields: List[str]):
        """Verificar estructura de respuesta"""
        for field in required_fields:
            assert field in response, f"Required field '{field}' missing in response"
    
    @staticmethod
    def wait_for_condition(condition_func, timeout=5, interval=0.1):
        """Esperar hasta que una condición se cumpla"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if condition_func():
                return True
            time.sleep(interval)
        return False


@pytest.fixture
def test_helpers():
    """Fixture para helpers de test"""
    return TestHelpers


class MockDataFactory:
    """Factory para crear datos mock"""
    
    @staticmethod
    def create_user(user_id="user123", email="user@example.com"):
        """Crear usuario mock"""
        return {
            "id": user_id,
            "email": email,
            "name": "Test User",
            "created_at": time.time()
        }
    
    @staticmethod
    def create_playlist(playlist_id="playlist123", name="Test Playlist"):
        """Crear playlist mock"""
        return {
            "id": playlist_id,
            "name": name,
            "tracks": [],
            "created_at": time.time()
        }
    
    @staticmethod
    def create_favorite(user_id="user123", track_id="track123"):
        """Crear favorito mock"""
        return {
            "user_id": user_id,
            "track_id": track_id,
            "created_at": time.time()
        }


@pytest.fixture
def mock_data_factory():
    """Fixture para factory de datos mock"""
    return MockDataFactory


def retry_on_failure(max_retries=3, delay=0.1):
    """Decorator para reintentar tests que fallan"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except AssertionError as e:
                    if attempt == max_retries - 1:
                        raise
                    time.sleep(delay)
        return wrapper
    return decorator


def skip_if_no_env(env_var):
    """Decorator para saltar tests si falta variable de entorno"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            if not os.getenv(env_var):
                pytest.skip(f"{env_var} not set")
            return func(*args, **kwargs)
        return wrapper
    return decorator

