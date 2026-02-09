"""
Helpers mejorados y extendidos para aserciones en tests
"""

import pytest
from typing import Any, Dict, List, Optional, Union
from fastapi import Response
from datetime import datetime
import uuid
import numpy as np


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
    try:
        uuid.UUID(uuid_string)
    except ValueError:
        pytest.fail(f"'{uuid_string}' is not a valid UUID")


def assert_valid_timestamp(timestamp: str) -> None:
    """Verifica que un string sea un timestamp válido"""
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


def assert_audio_data_valid(audio_data: np.ndarray) -> None:
    """Verifica que datos de audio sean válidos"""
    assert isinstance(audio_data, np.ndarray), "Audio data must be numpy array"
    assert len(audio_data) > 0, "Audio data is empty"
    assert np.all(np.isfinite(audio_data)), "Audio data contains NaN or Inf values"
    assert audio_data.dtype in [np.float32, np.float64], "Audio data must be float32 or float64"


def assert_list_not_empty(list_obj: List[Any], min_length: int = 1) -> None:
    """Verifica que una lista no esté vacía y tenga mínimo longitud"""
    assert isinstance(list_obj, list), "Object is not a list"
    assert len(list_obj) >= min_length, f"List has less than {min_length} items"


def assert_dict_contains(dict_obj: Dict[str, Any], keys: List[str]) -> None:
    """Verifica que un diccionario contenga las claves especificadas"""
    for key in keys:
        assert key in dict_obj, f"Missing key '{key}' in dictionary"


def assert_response_contains_keys(response: Union[Response, Dict], keys: List[str]) -> None:
    """Verifica que una respuesta contenga las claves especificadas"""
    if isinstance(response, Response):
        data = response.json()
    else:
        data = response
    
    assert_dict_contains(data, keys)


def assert_numeric_range(value: float, min_val: float, max_val: float, inclusive: bool = True) -> None:
    """Verifica que un valor numérico esté en un rango"""
    if inclusive:
        assert min_val <= value <= max_val, \
            f"Value {value} is not in range [{min_val}, {max_val}]"
    else:
        assert min_val < value < max_val, \
            f"Value {value} is not in range ({min_val}, {max_val})"


def assert_string_length(string: str, min_length: int = 0, max_length: Optional[int] = None) -> None:
    """Verifica que un string tenga una longitud válida"""
    length = len(string)
    assert length >= min_length, f"String length {length} is less than minimum {min_length}"
    if max_length is not None:
        assert length <= max_length, f"String length {length} exceeds maximum {max_length}"


def assert_email_valid(email: str) -> None:
    """Verifica que un email sea válido"""
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    assert bool(re.match(pattern, email)), f"'{email}' is not a valid email"


def assert_url_valid(url: str) -> None:
    """Verifica que una URL sea válida"""
    import re
    pattern = r'^https?://(?:[-\w.])+(?:\:[0-9]+)?(?:/(?:[\w/_.])*)?(?:\?(?:[\w&=%.])*)?(?:#(?:\w)*)?$'
    assert bool(re.match(pattern, url)), f"'{url}' is not a valid URL"


def assert_status_in_list(status: str, valid_statuses: List[str]) -> None:
    """Verifica que un status esté en la lista de status válidos"""
    assert status in valid_statuses, f"Status '{status}' is not in valid list: {valid_statuses}"


def assert_response_time_acceptable(response_time: float, max_time: float = 5.0) -> None:
    """Verifica que el tiempo de respuesta sea aceptable"""
    assert response_time <= max_time, \
        f"Response time {response_time}s exceeds maximum {max_time}s"


def assert_file_exists(file_path: str) -> None:
    """Verifica que un archivo exista"""
    from pathlib import Path
    path = Path(file_path)
    assert path.exists(), f"File does not exist: {file_path}"


def assert_file_size_valid(file_path: str, min_size: int = 0, max_size: Optional[int] = None) -> None:
    """Verifica que el tamaño de un archivo sea válido"""
    from pathlib import Path
    path = Path(file_path)
    assert path.exists(), f"File does not exist: {file_path}"
    
    size = path.stat().st_size
    assert size >= min_size, f"File size {size} is less than minimum {min_size}"
    if max_size is not None:
        assert size <= max_size, f"File size {size} exceeds maximum {max_size}"


def assert_json_structure(data: Dict[str, Any], schema: Dict[str, type]) -> None:
    """Verifica que un diccionario tenga la estructura de tipos esperada"""
    for key, expected_type in schema.items():
        assert key in data, f"Missing key '{key}'"
        assert isinstance(data[key], expected_type), \
            f"Key '{key}' has type {type(data[key])}, expected {expected_type}"


def assert_array_shape(array: np.ndarray, expected_shape: tuple) -> None:
    """Verifica que un array numpy tenga la forma esperada"""
    assert isinstance(array, np.ndarray), "Object is not a numpy array"
    assert array.shape == expected_shape, \
        f"Array shape {array.shape} does not match expected {expected_shape}"


def assert_array_dtype(array: np.ndarray, expected_dtype: type) -> None:
    """Verifica que un array numpy tenga el tipo de dato esperado"""
    assert isinstance(array, np.ndarray), "Object is not a numpy array"
    assert array.dtype == expected_dtype, \
        f"Array dtype {array.dtype} does not match expected {expected_dtype}"


def assert_no_duplicates(items: List[Any]) -> None:
    """Verifica que una lista no tenga duplicados"""
    assert len(items) == len(set(items)), "List contains duplicate items"


def assert_sorted(items: List[Any], reverse: bool = False) -> None:
    """Verifica que una lista esté ordenada"""
    sorted_items = sorted(items, reverse=reverse)
    assert items == sorted_items, "List is not sorted"


def assert_all_items_type(items: List[Any], expected_type: type) -> None:
    """Verifica que todos los items de una lista sean del tipo esperado"""
    for item in items:
        assert isinstance(item, expected_type), \
            f"Item {item} has type {type(item)}, expected {expected_type}"


def assert_response_headers(response: Response, headers: Dict[str, str]) -> None:
    """Verifica que una respuesta tenga los headers esperados"""
    for header_name, expected_value in headers.items():
        actual_value = response.headers.get(header_name)
        assert actual_value == expected_value, \
            f"Header '{header_name}' has value '{actual_value}', expected '{expected_value}'"


def assert_response_content_type(response: Response, expected_type: str) -> None:
    """Verifica que una respuesta tenga el content-type esperado"""
    content_type = response.headers.get("content-type", "")
    assert expected_type in content_type.lower(), \
        f"Content-Type '{content_type}' does not contain '{expected_type}'"



