"""
Helpers mejorados para aserciones en tests
"""

import pytest
from typing import Any, Dict, List, Optional
from fastapi import Response


def assert_response_structure(response: Response, expected_keys: List[str]) -> None:
    """Verifica que la respuesta tenga las claves esperadas"""
    data = response.json()
    for key in expected_keys:
        assert key in data, f"Missing key '{key}' in response: {data}"


def assert_response_status(response: Response, expected_status: int) -> None:
    """Verifica el status code de la respuesta"""
    assert response.status_code == expected_status, \
        f"Expected status {expected_status}, got {response.status_code}: {response.text}"


def assert_valid_uuid(uuid_string: str) -> None:
    """Verifica que un string sea un UUID válido"""
    import uuid
    try:
        uuid.UUID(uuid_string)
    except ValueError:
        pytest.fail(f"'{uuid_string}' is not a valid UUID")


def assert_valid_timestamp(timestamp: str) -> None:
    """Verifica que un string sea un timestamp válido"""
    from datetime import datetime
    try:
        datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
    except (ValueError, AttributeError):
        pytest.fail(f"'{timestamp}' is not a valid timestamp")


def assert_audio_response(response: Response) -> None:
    """Verifica que una respuesta sea de audio válida"""
    assert response.status_code == 200
    assert "audio" in response.headers.get("content-type", "").lower()
    assert len(response.content) > 0, "Audio response is empty"


def assert_pagination_response(response: Response) -> None:
    """Verifica que una respuesta tenga estructura de paginación"""
    data = response.json()
    required_keys = ["limit", "offset"]
    for key in required_keys:
        assert key in data, f"Missing pagination key '{key}'"
    assert isinstance(data.get("limit"), int)
    assert isinstance(data.get("offset"), int)


def assert_error_response(response: Response, expected_status: int = 400) -> None:
    """Verifica que una respuesta sea un error"""
    assert response.status_code == expected_status, \
        f"Expected error status {expected_status}, got {response.status_code}"
    data = response.json()
    assert "detail" in data or "message" in data, "Error response missing detail/message"


def assert_song_data_structure(song_data: Dict[str, Any]) -> None:
    """Verifica que los datos de canción tengan la estructura correcta"""
    required_keys = ["song_id", "user_id"]
    for key in required_keys:
        assert key in song_data, f"Missing song key '{key}'"
    assert_valid_uuid(song_data["song_id"])


def assert_playlist_data_structure(playlist_data: Dict[str, Any]) -> None:
    """Verifica que los datos de playlist tengan la estructura correcta"""
    required_keys = ["playlist_id", "name", "user_id"]
    for key in required_keys:
        assert key in playlist_data, f"Missing playlist key '{key}'"
    assert isinstance(playlist_data.get("songs"), list)


def assert_rating_valid(rating: float) -> None:
    """Verifica que un rating sea válido (0-5)"""
    assert 0 <= rating <= 5, f"Rating {rating} is out of valid range [0, 5]"


def assert_bpm_valid(bpm: Optional[float]) -> None:
    """Verifica que un BPM sea válido"""
    if bpm is not None:
        assert 0 < bpm <= 300, f"BPM {bpm} is out of valid range (0, 300]"


def assert_volume_valid(volume: float) -> None:
    """Verifica que un volumen sea válido (0.0-1.0)"""
    assert 0.0 <= volume <= 1.0, f"Volume {volume} is out of valid range [0.0, 1.0]"


def assert_duration_valid(duration: float) -> None:
    """Verifica que una duración sea válida"""
    assert duration > 0, f"Duration {duration} must be positive"
    assert duration <= 600, f"Duration {duration} exceeds maximum (600 seconds)"


def assert_list_response(response: Response, min_items: int = 0) -> None:
    """Verifica que una respuesta sea una lista válida"""
    data = response.json()
    assert isinstance(data, (list, dict)), "Response is not a list or dict"
    
    if isinstance(data, dict) and "items" in data:
        items = data["items"]
    elif isinstance(data, list):
        items = data
    else:
        items = []
    
    assert len(items) >= min_items, f"Expected at least {min_items} items, got {len(items)}"


def assert_metadata_structure(metadata: Dict[str, Any], expected_keys: Optional[List[str]] = None) -> None:
    """Verifica que metadata tenga la estructura correcta"""
    assert isinstance(metadata, dict), "Metadata must be a dictionary"
    if expected_keys:
        for key in expected_keys:
            assert key in metadata, f"Missing metadata key '{key}'"



